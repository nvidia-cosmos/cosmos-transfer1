# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import einops
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from einops import rearrange

from cosmos_transfer1.auxiliary.guardrail.common.io_utils import save_video
from cosmos_transfer1.checkpoints import (
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.config.transfer.augmentors import BilateralOnlyBlurAugmentorConfig
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.config_helper import get_config_module, override
from cosmos_transfer1.utils.io import load_from_fileobj

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

DEFAULT_AUGMENT_SIGMA = 0.001
NUM_MAX_FRAMES = 5000
VIDEO_RES_SIZE_INFO = {
    "1,1": (960, 960),
    "4,3": (960, 704),
    "3,4": (704, 960),
    "16,9": (1280, 704),
    "9,16": (704, 1280),
}

# Default model names for each control type
default_model_names = {
    "vis": VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "seg": SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "edge": EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "depth": DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "keypoint": KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "upscale": UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    "hdmap": HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    "lidar": LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
}


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> _IncompatibleKeys:
    """Load a model checkpoint with non-strict matching, handling shape mismatches.

    Args:
        model (torch.nn.Module): Model to load weights into
        checkpoint_state_dict (dict): State dict from checkpoint

    Returns:
        _IncompatibleKeys: Named tuple containing:
            - missing_keys: Keys present in model but missing from checkpoint
            - unexpected_keys: Keys present in checkpoint but not in model
            - incorrect_shapes: Keys with mismatched tensor shapes

    The function handles special cases like:
    - Uninitialized parameters
    - Quantization observers
    - TransformerEngine FP8 states
    """
    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            if "_extra_state" in k:  # Key introduced by TransformerEngine for FP8
                log.debug(f"Skipping key {k} introduced by TransformerEngine for FP8 in the checkpoint.")
                continue
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(model_param, torch.nn.parameter.UninitializedParameter):
                continue
            if not isinstance(model_param, torch.Tensor):
                raise ValueError(
                    f"Find non-tensor parameter {k} in the model. type: {type(model_param)} {type(checkpoint_state_dict[k])}, please check if this key is safe to skip or not."
                )

            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(model: torch.nn.Module, key: str) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    # Remove keys with "_extra_state" suffix, which are non-parameter items introduced by TransformerEngine for FP8 handling
    missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    unexpected_keys = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )


@contextmanager
def skip_init_linear():
    # skip init of nn.Linear
    orig_reset_parameters = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda x: x
    xavier_uniform_ = torch.nn.init.xavier_uniform_
    torch.nn.init.xavier_uniform_ = lambda x: x
    yield
    torch.nn.Linear.reset_parameters = orig_reset_parameters
    torch.nn.init.xavier_uniform_ = xavier_uniform_


def load_model_by_config(
    config_job_name,
    config_file="projects/cosmos_video/config/config.py",
    model_class=DiffusionT2WModel,
    base_checkpoint_dir="",
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={config_job_name}"])
    if base_checkpoint_dir != "" and hasattr(config.model, "base_load_from"):
        if hasattr(config.model.base_load_from, "load_path"):
            if config.model.base_load_from.load_path != "":
                config.model.base_load_from.load_path = config.model.base_load_from.load_path.replace(
                    "checkpoints", base_checkpoint_dir
                )
                log.info(
                    f"Model need to load a base model weight, change the loading path from default folder to the {base_checkpoint_dir}"
                )

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore

    # Initialize model
    with skip_init_linear():
        model = model_class(config.model)
    return model


def load_network_model(model: DiffusionT2WModel, ckpt_path: str):
    with skip_init_linear():
        model.set_up_model()
    net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # , weights_only=True)
    non_strict_load_model(model.model, net_state_dict)
    model.cuda()


def load_tokenizer_model(model: DiffusionT2WModel, tokenizer_dir: str):
    with skip_init_linear():
        model.set_up_tokenizer(tokenizer_dir)
    model.cuda()


def prepare_data_batch(
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    prompt_embedding: torch.Tensor,
    negative_prompt_embedding: Optional[torch.Tensor] = None,
):
    """Prepare input batch tensors for video generation.

    Args:
        height (int): Height of video frames
        width (int): Width of video frames
        num_frames (int): Number of frames to generate
        fps (int): Frames per second
        prompt_embedding (torch.Tensor): Encoded text prompt embeddings
        negative_prompt_embedding (torch.Tensor, optional): Encoded negative prompt embeddings

    Returns:
        dict: Batch dictionary containing:
            - video: Zero tensor of target video shape
            - t5_text_mask: Attention mask for text embeddings
            - image_size: Target frame dimensions
            - fps: Target frame rate
            - num_frames: Number of frames
            - padding_mask: Frame padding mask
            - t5_text_embeddings: Prompt embeddings
            - neg_t5_text_embeddings: Negative prompt embeddings (if provided)
            - neg_t5_text_mask: Mask for negative embeddings (if provided)
    """
    # Create base data batch
    data_batch = {
        "video": torch.zeros((1, 3, num_frames, height, width), dtype=torch.uint8).cuda(),
        "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16).cuda(),
        "image_size": torch.tensor([[height, width, height, width]] * 1, dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([fps] * 1, dtype=torch.bfloat16).cuda(),
        "num_frames": torch.tensor([num_frames] * 1, dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((1, 1, height, width), dtype=torch.bfloat16).cuda(),
    }

    # Handle text embeddings

    t5_embed = prompt_embedding.to(dtype=torch.bfloat16).cuda()
    data_batch["t5_text_embeddings"] = t5_embed

    if negative_prompt_embedding is not None:
        neg_t5_embed = negative_prompt_embedding.to(dtype=torch.bfloat16).cuda()
        data_batch["neg_t5_text_embeddings"] = neg_t5_embed
        data_batch["neg_t5_text_mask"] = torch.ones(1, 512, dtype=torch.bfloat16).cuda()

    return data_batch


def get_video_batch(model, prompt_embedding, negative_prompt_embedding, height, width, fps, num_video_frames):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance
        prompt_embedding (torch.Tensor): Text prompt embeddings
        negative_prompt_embedding (torch.Tensor): Negative prompt embeddings
        height (int): Output video height
        width (int): Output video width
        fps (int): Output video frame rate
        num_video_frames (int): Number of frames to generate

    Returns:
        tuple:
            - data_batch (dict): Complete model input batch
            - state_shape (list): Shape of latent state [C,T,H,W] accounting for VAE compression
    """
    raw_video_batch = prepare_data_batch(
        height=height,
        width=width,
        num_frames=num_video_frames,
        fps=fps,
        prompt_embedding=prompt_embedding,
        negative_prompt_embedding=negative_prompt_embedding,
    )
    state_shape = [
        model.tokenizer.channel,
        model.tokenizer.get_latent_num_frames(num_video_frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    ]
    return raw_video_batch, state_shape


def resize_video(video_np, h, w, interpolation=cv2.INTER_AREA):
    """Resize video frames to the specified height and width."""
    video_np = video_np[0].transpose((1, 2, 3, 0))  # Convert to T x H x W x C
    t = video_np.shape[0]
    resized_video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        resized_video[i] = cv2.resize(video_np[i], (w, h), interpolation=interpolation)
    return resized_video.transpose((3, 0, 1, 2))[None]  # Convert back to B x C x T x H x W


def detect_aspect_ratio(img_size: tuple[int]):
    """Function for detecting the closest aspect ratio."""

    _aspect_ratios = np.array([(16 / 9), (4 / 3), 1, (3 / 4), (9 / 16)])
    _aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    w, h = img_size
    current_ratio = w / h
    closest_aspect_ratio = np.argmin((_aspect_ratios - current_ratio) ** 2)
    return _aspect_ratio_keys[closest_aspect_ratio]


def get_upscale_size(orig_size: tuple[int], aspect_ratio: str, upscale_factor: int = 3, patch_overlap: int = 256):
    patch_w, patch_h = orig_size
    if aspect_ratio == "16,9" or aspect_ratio == "4,3":
        ratio = int(aspect_ratio.split(",")[1]) / int(aspect_ratio.split(",")[0])
        target_w = patch_w * upscale_factor - patch_overlap
        target_h = patch_h * upscale_factor - int(patch_overlap * ratio)
    elif aspect_ratio == "9,16" or aspect_ratio == "3,4":
        ratio = int(aspect_ratio.split(",")[0]) / int(aspect_ratio.split(",")[1])
        target_h = patch_h * upscale_factor - patch_overlap
        target_w = patch_w * upscale_factor - int(patch_overlap * ratio)
    else:
        target_h = patch_h * upscale_factor - patch_overlap
        target_w = patch_w * upscale_factor - patch_overlap
    return target_w, target_h


def read_and_resize_input(input_control_path, num_total_frames, interpolation):
    control_input, fps = read_video_or_image_into_frames_BCTHW(
        input_control_path,
        normalize=False,  # s.t. output range is [0, 255]
        max_frames=num_total_frames,
        also_return_fps=True,
    )  # BCTHW
    aspect_ratio = detect_aspect_ratio((control_input.shape[-1], control_input.shape[-2]))
    w, h = VIDEO_RES_SIZE_INFO[aspect_ratio]
    control_input = resize_video(control_input, h, w, interpolation=interpolation)  # BCTHW, range [0, 255]
    control_input = torch.from_numpy(control_input[0])  # CTHW, range [0, 255]
    return control_input, fps, aspect_ratio


def get_ctrl_batch(
    model, data_batch, num_video_frames, input_video_path, control_inputs, blur_strength, canny_threshold
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance

    Returns:
        - data_batch (dict): Complete model input batch
    """
    state_shape = model.state_shape

    H, W = (
        state_shape[-2] * model.tokenizer.spatial_compression_factor,
        state_shape[-1] * model.tokenizer.spatial_compression_factor,
    )

    # Initialize control input dictionary
    control_input_dict = {k: v for k, v in data_batch.items()}
    num_total_frames = NUM_MAX_FRAMES
    if input_video_path:
        input_frames, fps, aspect_ratio = read_and_resize_input(
            input_video_path, num_total_frames=num_total_frames, interpolation=cv2.INTER_AREA
        )
        _, num_total_frames, H, W = input_frames.shape
        control_input_dict["video"] = input_frames.numpy()  # CTHW
        data_batch["input_video"] = input_frames.bfloat16()[None] / 255 * 2 - 1  # BCTHW
    else:
        data_batch["input_video"] = None
    target_w, target_h = W, H

    control_weights = []
    for hint_key, control_info in control_inputs.items():
        if "input_control" in control_info:
            in_file = control_info["input_control"]
            interpolation = cv2.INTER_NEAREST if hint_key == "seg" else cv2.INTER_LINEAR
            log.info(f"reading control input {in_file} for hint {hint_key}")
            control_input_dict[f"control_input_{hint_key}"], fps, aspect_ratio = read_and_resize_input(
                in_file, num_total_frames=num_total_frames, interpolation=interpolation
            )  # CTHW
            num_total_frames = min(num_total_frames, control_input_dict[f"control_input_{hint_key}"].shape[1])
            target_h, target_w = H, W = control_input_dict[f"control_input_{hint_key}"].shape[2:]
        if hint_key == "upscale":
            orig_size = (W, H)
            target_w, target_h = get_upscale_size(orig_size, aspect_ratio, upscale_factor=3)
            input_resized = resize_video(
                input_frames[None].numpy(),
                target_h,
                target_w,
                interpolation=cv2.INTER_LINEAR,
            )  # BCTHW
            control_input_dict["control_input_upscale"] = split_video_into_patches(
                torch.from_numpy(input_resized), H, W
            )
            data_batch["input_video"] = control_input_dict["control_input_upscale"].bfloat16() / 255 * 2 - 1
        control_weights.append(control_info["control_weight"])

    # Trim all control videos and input video to be the same length.
    log.info(f"Making all control and input videos to be length of {num_total_frames} frames.")
    if len(control_inputs) > 1:
        for hint_key in control_inputs.keys():
            cur_key = f"control_input_{hint_key}"
            if cur_key in control_input_dict:
                control_input_dict[cur_key] = control_input_dict[cur_key][:, :num_total_frames]
    if input_video_path:
        control_input_dict["video"] = control_input_dict["video"][:, :num_total_frames]
        data_batch["input_video"] = data_batch["input_video"][:, :, :num_total_frames]

    hint_key = "control_input_" + "_".join(control_inputs.keys())
    add_control_input = get_augmentor_for_eval(
        input_key="video",
        output_key=hint_key,
        preset_blur_strength=blur_strength,
        preset_canny_threshold=canny_threshold,
        blur_config=BilateralOnlyBlurAugmentorConfig[blur_strength],
    )

    if len(control_input_dict):
        control_input = add_control_input(control_input_dict)[hint_key]
        if control_input.ndim == 4:
            control_input = control_input[None]
        control_input = control_input.bfloat16() / 255 * 2 - 1
        control_weights = load_spatial_temporal_weights(
            control_weights, B=1, T=num_video_frames, H=target_h, W=target_w, patch_h=H, patch_w=W
        )
        data_batch["control_weight"] = control_weights

        if len(control_inputs) > 1:  # Multicontrol enabled
            data_batch["hint_key"] = "control_input_multi"
            data_batch["control_input_multi"] = control_input
        else:  # Single-control case
            data_batch["hint_key"] = hint_key
            data_batch[hint_key] = control_input

    data_batch["target_h"], data_batch["target_w"] = target_h // 8, target_w // 8
    data_batch["video"] = torch.zeros((1, 3, 121, H, W), dtype=torch.uint8).cuda()
    data_batch["image_size"] = torch.tensor([[H, W, H, W]] * 1, dtype=torch.bfloat16).cuda()
    data_batch["padding_mask"] = torch.zeros((1, 1, H, W), dtype=torch.bfloat16).cuda()

    return data_batch


def generate_control_input(input_file_path, save_folder, hint_key, blur_strength, canny_threshold, num_total_frames=10):
    log.info(
        f"Generating control input for {hint_key} with blur strength {blur_strength} and canny threshold {canny_threshold}"
    )
    video_input = read_video_or_image_into_frames_BCTHW(input_file_path, normalize=False)[0, :, :num_total_frames]
    control_input = get_augmentor_for_eval(
        input_key="video",
        output_key=hint_key,
        preset_blur_strength=blur_strength,
        preset_canny_threshold=canny_threshold,
        blur_config=BilateralOnlyBlurAugmentorConfig[blur_strength],
    )
    control_input = control_input({"video": video_input})[hint_key]
    control_input = control_input.numpy().transpose((1, 2, 3, 0))

    output_file_path = f"{save_folder}/{hint_key}_upsampler.mp4"
    log.info(f"Saving control input to {output_file_path}")
    save_video(frames=control_input, fps=24, filepath=output_file_path)
    return output_file_path


def generate_world_from_control(
    model: DiffusionV2WModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    guidance: float,
    num_steps: int,
    seed: int,
    condition_latent: torch.Tensor,
    num_input_frames: int,
    sigma_max: float,
    x_sigma_max=None,
) -> Tuple[np.array, list, list]:
    """Generate video using a conditioning video/image input.

    Args:
        model (DiffusionV2WModel): The diffusion model instance
        state_shape (list[int]): Shape of the latent state [C,T,H,W]
        is_negative_prompt (bool): Whether negative prompt is provided
        data_batch (dict): Batch containing model inputs including text embeddings
        guidance (float): Classifier-free guidance scale for sampling
        num_steps (int): Number of diffusion sampling steps
        seed (int): Random seed for generation
        condition_latent (torch.Tensor): Latent tensor from conditioning video/image file
        num_input_frames (int): Number of input frames

    Returns:
        np.array: Generated video frames in shape [T,H,W,C], range [0,255]
    """
    assert not model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i, "not supported"
    augment_sigma = DEFAULT_AUGMENT_SIGMA

    b, c, t, h, w = condition_latent.shape
    if condition_latent.shape[2] < state_shape[1]:
        # Padding condition latent to state shape
        condition_latent = torch.cat(
            [
                condition_latent,
                condition_latent.new_zeros(b, c, state_shape[1] - t, h, w),
            ],
            dim=2,
        ).contiguous()
    num_of_latent_condition = compute_num_latent_frames(model, num_input_frames)

    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        state_shape=[c, t, h, w],
        num_steps=num_steps,
        is_negative_prompt=is_negative_prompt,
        seed=seed,
        condition_latent=condition_latent,
        num_condition_t=num_of_latent_condition,
        condition_video_augment_sigma_in_inference=augment_sigma,
        x_sigma_max=x_sigma_max,
        sigma_max=sigma_max,
        target_h=data_batch["target_h"],
        target_w=data_batch["target_w"],
        patch_h=h,
        patch_w=w,
    )
    return sample


def read_video_or_image_into_frames_BCTHW(
    input_path: str,
    input_path_format: str = "mp4",
    H: int = None,
    W: int = None,
    normalize: bool = True,
    max_frames: int = -1,
    also_return_fps: bool = False,
) -> torch.Tensor:
    """Read video or image file and convert to tensor format.

    Args:
        input_path (str): Path to input video/image file
        input_path_format (str): Format of input file (default: "mp4")
        H (int, optional): Height to resize frames to
        W (int, optional): Width to resize frames to
        normalize (bool): Whether to normalize pixel values to [-1,1] (default: True)
        max_frames (int): Maximum number of frames to read (-1 for all frames)
        also_return_fps (bool): Whether to return fps along with frames

    Returns:
        torch.Tensor | tuple: Video tensor in shape [B,C,T,H,W], optionally with fps if requested
    """
    log.debug(f"Reading video from {input_path}")

    loaded_data = load_from_fileobj(input_path, format=input_path_format)
    frames, meta_data = loaded_data
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(frames[0])  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        fps = int(meta_data.get("fps"))
    if max_frames != -1:
        frames = frames[:max_frames]
    input_tensor = np.stack(frames, axis=0)
    input_tensor = einops.rearrange(input_tensor, "t h w c -> t c h w")
    if normalize:
        input_tensor = input_tensor / 128.0 - 1.0
        input_tensor = torch.from_numpy(input_tensor).bfloat16()  # TCHW
        log.debug(f"Raw data shape: {input_tensor.shape}")
        if H is not None and W is not None:
            input_tensor = transforms_F.resize(
                input_tensor,
                size=(H, W),  # type: ignore
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
    input_tensor = einops.rearrange(input_tensor, "(b t) c h w -> b c t h w", b=1)
    if normalize:
        input_tensor = input_tensor.to("cuda")
    log.debug(f"Load shape {input_tensor.shape} value {input_tensor.min()}, {input_tensor.max()}")
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def compute_num_latent_frames(model: DiffusionV2WModel, num_input_frames: int, downsample_factor=8) -> int:
    """This function computes the number of latent frames given the number of input frames.
    Args:
        model (DiffusionV2WModel): video generation model
        num_input_frames (int): number of input frames
        downsample_factor (int): downsample factor for temporal reduce
    Returns:
        int: number of latent frames
    """
    num_latent_frames = (
        num_input_frames
        // model.tokenizer.video_vae.pixel_chunk_duration
        * model.tokenizer.video_vae.latent_chunk_duration
    )
    if num_input_frames % model.tokenizer.video_vae.latent_chunk_duration == 1:
        num_latent_frames += 1
    elif num_input_frames % model.tokenizer.video_vae.latent_chunk_duration > 1:
        assert (
            num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1
        ) % downsample_factor == 0, f"num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1 must be divisible by {downsample_factor}"
        num_latent_frames += (
            1 + (num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration - 1) // downsample_factor
        )

    return num_latent_frames


def create_condition_latent_from_input_frames(
    model: DiffusionV2WModel,
    input_frames: torch.Tensor,
    num_frames_condition: int = 25,
):
    """Create condition latent for video generation from input frames.

    Takes the last num_frames_condition frames from input as conditioning.

    Args:
        model (DiffusionV2WModel): Video generation model
        input_frames (torch.Tensor): Input video tensor [B,C,T,H,W], range [-1,1]
        num_frames_condition (int): Number of frames to use for conditioning

    Returns:
        tuple: (condition_latent, encode_input_frames) where:
            - condition_latent (torch.Tensor): Encoded latent condition [B,C,T,H,W]
            - encode_input_frames (torch.Tensor): Padded input frames used for encoding
    """
    B, C, T, H, W = input_frames.shape
    num_frames_encode = (
        model.tokenizer.pixel_chunk_duration
    )  # (model.state_shape[1] - 1) / model.vae.pixel_chunk_duration + 1
    log.debug(
        f"num_frames_encode not set, set it based on pixel chunk duration and model state shape: {num_frames_encode}"
    )

    log.debug(
        f"Create condition latent from input frames {input_frames.shape}, value {input_frames.min()}, {input_frames.max()}, dtype {input_frames.dtype}"
    )

    assert (
        input_frames.shape[2] >= num_frames_condition
    ), f"input_frames not enough for condition, require at least {num_frames_condition}, get {input_frames.shape[2]}, {input_frames.shape}"
    assert (
        num_frames_encode >= num_frames_condition
    ), f"num_frames_encode should be larger than num_frames_condition, get {num_frames_encode}, {num_frames_condition}"

    # Put the conditioal frames to the begining of the video, and pad the end with zero
    condition_frames = input_frames[:, :, -num_frames_condition:]
    padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
    encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2)

    log.debug(
        f"create latent with input shape {encode_input_frames.shape} including padding {num_frames_encode - num_frames_condition} at the end"
    )
    latent = model.encode(encode_input_frames)
    return latent, encode_input_frames


def get_condition_latent(
    model: DiffusionV2WModel,
    input_video_path: str,
    num_input_frames: int = 1,
    state_shape: list[int] = None,
):
    """Get condition latent from input image/video file.

    Args:
        model (DiffusionV2WModel): Video generation model
        input_video_path (str): Path to conditioning image/video
        num_input_frames (int): Number of input frames for video2world prediction

    Returns:
        tuple: (condition_latent, input_frames) where:
            - condition_latent (torch.Tensor): Encoded latent condition [B,C,T,H,W]
            - input_frames (torch.Tensor): Input frames tensor [B,C,T,H,W]
    """
    if state_shape is None:
        state_shape = model.state_shape
    assert num_input_frames > 0, "num_input_frames must be greater than 0"

    H, W = (
        state_shape[-2] * model.tokenizer.spatial_compression_factor,
        state_shape[-1] * model.tokenizer.spatial_compression_factor,
    )

    input_path_format = input_video_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )

    condition_latent, _ = create_condition_latent_from_input_frames(model, input_frames, num_input_frames)
    condition_latent = condition_latent.to(torch.bfloat16)

    return condition_latent


def check_input_frames(input_path: str, required_frames: int) -> bool:
    """Check if input video/image has sufficient frames.

    Args:
        input_path: Path to input video or image
        required_frames: Number of required frames

    Returns:
        np.ndarray of frames if valid, None if invalid
    """
    if input_path.endswith((".jpg", ".jpeg", ".png")):
        if required_frames > 1:
            log.error(f"Input ({input_path}) is an image but {required_frames} frames are required")
            return False
        return True  # Let the pipeline handle image loading
    # For video input
    try:
        vid = imageio.get_reader(input_path, "ffmpeg")
        frame_count = vid.count_frames()

        if frame_count < required_frames:
            log.error(f"Input video has {frame_count} frames but {required_frames} frames are required")
            return False
        else:
            return True
    except Exception as e:
        log.error(f"Error reading video file {input_path}: {e}")
        return False


def load_spatial_temporal_weights(weight_paths, B, T, H, W, patch_h, patch_w):
    """
    Load and process spatial-temporal weight maps from .pt files
    Args:
        weight_paths: List of weights that can be scalars, paths to .pt files, or empty strings
        B, T, H, W: Desired tensor dimensions
        patch_h, patch_w: Patch dimensions for splitting
    Returns:
        For all scalar weights: tensor of shape [num_controls]
        For any spatial maps: tensor of shape [num_controls, B, 1, T, H, W]
    """
    # Process each weight path
    weights = []
    has_spatial_weights = False
    for path in weight_paths:
        if not path or (isinstance(path, str) and path.lower() == "none"):
            # Use default weight of 1.0
            w = torch.ones((T, H, W), dtype=torch.bfloat16)
        else:
            try:
                # Try to parse as scalar
                scalar_value = float(path)
                w = torch.full((T, H, W), scalar_value, dtype=torch.bfloat16)
            except ValueError:
                # Not a scalar, must be a path to a weight map
                has_spatial_weights = True
                w = torch.load(path, weights_only=False).to(dtype=torch.bfloat16)  # [T, H, W]
                if w.ndim == 2:  # Spatial only
                    w = w.unsqueeze(0).repeat(T, 1, 1)
                elif w.ndim != 3:
                    raise ValueError(f"Weight map must be 2D or 3D, got shape {w.shape}")

                if w.shape != (T, H, W):
                    w = (
                        torch.nn.functional.interpolate(
                            w.unsqueeze(0).unsqueeze(0),
                            size=(T, H, W),
                            mode="trilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
        w = torch.clamp(w, min=0)
        w = w.unsqueeze(0).unsqueeze(1)
        w = w.expand(B, 1, -1, -1, -1)
        weights.append(w)

    if not has_spatial_weights:
        scalar_weights = [float(w) for w in weight_paths]
        weights_tensor = torch.tensor(scalar_weights, dtype=torch.bfloat16)
        weights_tensor = weights_tensor / (weights_tensor.sum().clip(1))
        return weights_tensor.cuda()

    weights = torch.stack(weights, dim=0)
    weights = weights / (weights.sum(dim=0, keepdim=True).clip(1))

    # Split into patches if needed
    if patch_h != H or patch_w != W:
        num_controls = len(weights)
        weights = weights.reshape(num_controls * B, 1, T, H, W)
        weights = split_video_into_patches(weights, patch_h, patch_w)
        B_new = weights.shape[0] // num_controls
        weights = weights.reshape(num_controls, B_new, 1, T, H, W)

    return weights


def resize_control_weight_map(control_weight_map, size):
    assert control_weight_map.shape[2] == 1  # [num_control, B, 1, T, H, W]
    weight_map = control_weight_map.squeeze(2)  # [num_control, B, T, H, W]
    T, H, W = size
    if weight_map.shape[2:5] != (T, H, W):
        assert (weight_map.shape[2] == T) or (weight_map.shape[2] == 8 * (T - 1) + 1)
        weight_map_i = [
            torch.nn.functional.interpolate(
                weight_map[:, :, :1],
                size=(1, H, W),
                mode="trilinear",
                align_corners=False,
            )
        ]
        weight_map_i += [
            torch.nn.functional.interpolate(
                weight_map[:, :, 1:],
                size=(T - 1, H, W),
                mode="trilinear",
                align_corners=False,
            )
        ]
        weight_map = torch.cat(weight_map_i, dim=2)
    return weight_map.unsqueeze(2)


def split_video_into_patches(tensor, patch_h, patch_w):
    h, w = tensor.shape[-2:]
    n_img_w = (w - 1) // patch_w + 1
    n_img_h = (h - 1) // patch_h + 1
    overlap_size_h = overlap_size_w = 0
    if n_img_w > 1:
        overlap_size_w = (n_img_w * patch_w - w) // (n_img_w - 1)  # 512 for n=2, 320 for n=4
        assert n_img_w * patch_w - overlap_size_w * (n_img_w - 1) == w
    if n_img_h > 1:
        overlap_size_h = (n_img_h * patch_h - h) // (n_img_h - 1)
        assert n_img_h * patch_h - overlap_size_h * (n_img_h - 1) == h
    p_h = patch_h - overlap_size_h
    p_w = patch_w - overlap_size_w

    patches = []
    for i in range(n_img_h):
        for j in range(n_img_w):
            patches += [tensor[:, :, :, p_h * i : (p_h * i + patch_h), p_w * j : (p_w * j + patch_w)]]
    return torch.cat(patches)


def merge_patches_into_video(imgs, overlap_size_h, overlap_size_w, n_img_h, n_img_w):
    b, c, t, h, w = imgs.shape
    imgs = rearrange(imgs, "(b m n) c t h w -> m n b c t h w", m=n_img_h, n=n_img_w)
    H = n_img_h * h - (n_img_h - 1) * overlap_size_h
    W = n_img_w * w - (n_img_w - 1) * overlap_size_w
    img_sum = torch.zeros((b // (n_img_h * n_img_w), c, t, H, W)).to(imgs)
    mask_sum = torch.zeros((H, W)).to(imgs)

    # Create a linear mask for blending.
    def create_linear_gradient_tensor(H, W, overlap_size_h, overlap_size_w):
        y, x = torch.meshgrid(
            torch.minimum(torch.arange(H), H - torch.arange(H)) / (overlap_size_h + 1e-6),
            torch.minimum(torch.arange(W), W - torch.arange(W)) / (overlap_size_w + 1e-6),
        )
        return torch.clamp(y, 0.01, 1) * torch.clamp(x, 0.01, 1)

    mask_ij = create_linear_gradient_tensor(h, w, overlap_size_h, overlap_size_w).to(imgs)

    for i in range(n_img_h):
        for j in range(n_img_w):
            h_start = i * (h - overlap_size_h)
            w_start = j * (w - overlap_size_w)
            img_sum[:, :, :, h_start : h_start + h, w_start : w_start + w] += (
                imgs[i, j] * mask_ij[None, None, None, :, :]
            )
            mask_sum[h_start : h_start + h, w_start : w_start + w] += mask_ij
    return img_sum / (mask_sum[None, None, None, :, :] + 1e-6)


valid_hint_keys = {"vis", "seg", "edge", "depth", "keypoint", "upscale", "hdmap", "lidar"}


def load_controlnet_specs(cfg) -> Dict[str, Any]:
    with open(cfg.controlnet_specs, "r") as f:
        controlnet_specs_in = json.load(f)

    controlnet_specs = {}
    args = {}

    for hint_key, config in controlnet_specs_in.items():
        if hint_key in valid_hint_keys:
            controlnet_specs[hint_key] = config
        else:
            if type(config) == dict:
                raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")
            else:
                args[hint_key] = config
                continue
    return controlnet_specs, args


def validate_controlnet_specs(cfg, controlnet_specs) -> Dict[str, Any]:
    """
    Load and validate controlnet specifications from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing controlnet specs.
        checkpoint_dir (str): Base directory for checkpoint files.

    Returns:
        Dict[str, Any]: Validated and processed controlnet specifications.
    """
    checkpoint_dir = cfg.checkpoint_dir
    sigma_max = cfg.sigma_max
    input_video_path = cfg.input_video_path

    for hint_key, config in controlnet_specs.items():
        if hint_key not in valid_hint_keys:
            raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")

        if not input_video_path and sigma_max < 80:
            raise ValueError("Must have 'input_video' specified if sigma_max < 80")

        if not input_video_path and "input_control" not in config:
            raise ValueError(
                f"{hint_key} controlnet must have 'input_control' video specified if no 'input_video' specified."
            )

        if "ckpt_path" not in config:
            log.info(f"No checkpoint path specified for {hint_key}. Using default.")
            config["ckpt_path"] = os.path.join(checkpoint_dir, default_model_names[hint_key])
            log.info(f"Using default checkpoint path: {config['ckpt_path']}")

        # Regardless whether "control_weight_prompt" is provided (i.e. whether we automatically
        # generate spatiotemporal control weight binary masks), control_weight is needed to.
        if "control_weight" not in config:
            log.warning(f"No control weight specified for {hint_key}. Setting to 0.5.")
            config["control_weight"] = "0.5"
        else:
            # Check if control weight is a path or a scalar
            weight = config["control_weight"]
            if not isinstance(weight, str) or not weight.endswith(".pt"):
                try:
                    # Try converting to float
                    scalar_value = float(weight)
                    if scalar_value < 0:
                        raise ValueError(f"Control weight for {hint_key} must be non-negative.")
                except ValueError:
                    raise ValueError(
                        f"Control weight for {hint_key} must be a valid non-negative float or a path to a .pt file."
                    )

    return controlnet_specs
