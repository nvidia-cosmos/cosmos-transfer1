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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import einops
import imageio
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as transforms_F
from einops import rearrange
from matplotlib import pyplot as plt

from cosmos_transfer1.auxiliary.guardrail.common.io_utils import save_video
from cosmos_transfer1.checkpoints import (
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
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
from cosmos_transfer1.diffusion.training.models.extend_model import ExtendDiffusionModel
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

default_distilled_model_names = {
    "edge": EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
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
    if ckpt_path:
        with skip_init_linear():
            model.set_up_model()
        net_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # , weights_only=True)
        non_strict_load_model(model.model, net_state_dict)
    else:
        model.set_up_model()
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


def get_video_batch_for_multiview_model(
    model, prompt_embedding, height, width, fps, num_video_frames, frame_repeat_negative_condition
):
    """Prepare complete input batch for video generation including latent dimensions.

    Args:
        model: Diffusion model instance
        prompt_embedding list(torch.Tensor): Text prompt embeddings
        height (int): Output video height
        width (int): Output video width
        fps (int): Output video frame rate
        num_video_frames (int): Number of frames to generate
        frame_repeat_negative_condition (int): Number of frames to generate

    Returns:
        tuple:
            - data_batch (dict): Complete model input batch
            - state_shape (list): Shape of latent state [C,T,H,W] accounting for VAE compression
    """
    n_views = len(prompt_embedding)
    prompt_embedding = einops.rearrange(prompt_embedding, "n t d -> (n t) d").unsqueeze(0)
    raw_video_batch = prepare_data_batch(
        height=height,
        width=width,
        num_frames=num_video_frames,
        fps=fps,
        prompt_embedding=prompt_embedding,
    )
    if frame_repeat_negative_condition != -1:
        frame_repeat = torch.zeros(n_views)
        frame_repeat[-1] = frame_repeat_negative_condition
        frame_repeat[-2] = frame_repeat_negative_condition
        raw_video_batch["frame_repeat"] = frame_repeat.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
    state_shape = [
        model.tokenizer.channel,
        model.tokenizer.get_latent_num_frames(int(num_video_frames / n_views)) * n_views,
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    ]
    return raw_video_batch, state_shape


def get_ctrl_batch_mv(H, W, data_batch, num_total_frames, control_inputs, num_views, num_video_frames):
    # Initialize control input dictionary
    control_input_dict = {k: v for k, v in data_batch.items()}
    control_weights = []
    hint_keys = []
    for hint_key, control_info in control_inputs.items():
        if hint_key not in valid_hint_keys:
            continue
        if "input_control" in control_info:
            cond_videos = []
            for in_file in control_info["input_control"]:
                log.info(f"reading control input {in_file} for hint {hint_key}")
                cond_vid, fps = read_video_or_image_into_frames_BCTHW(
                    in_file,
                    normalize=False,  # s.t. output range is [0, 255]
                    max_frames=num_total_frames,
                    also_return_fps=True,
                )
                cond_vid = resize_video(cond_vid, H, W, interpolation=cv2.INTER_LINEAR)
                cond_vid = torch.from_numpy(cond_vid[0])

                cond_videos.append(cond_vid)

            input_frames = torch.cat(cond_videos, dim=1)
            control_input_dict[f"control_input_{hint_key}"] = input_frames
            hint_keys.append(hint_key)
        control_weights.append(control_info["control_weight"])

    target_w, target_h = W, H
    hint_key = "control_input_" + "_".join(hint_keys)
    add_control_input = get_augmentor_for_eval(input_key="video", output_key=hint_key)

    if len(control_input_dict):
        control_input = add_control_input(control_input_dict)[hint_key]
        if control_input.ndim == 4:
            control_input = control_input[None]
        control_input = control_input.bfloat16() / 255 * 2 - 1
        control_weights = load_spatial_temporal_weights(
            control_weights, B=1, T=num_total_frames, H=target_h, W=target_w, patch_h=H, patch_w=W
        )
        data_batch["control_weight"] = control_weights

        if len(control_inputs) > 1:  # Multicontrol enabled
            data_batch["hint_key"] = "control_input_multi"
            data_batch["control_input_multi"] = control_input
        else:  # Single-control case
            data_batch["hint_key"] = hint_key
            data_batch[hint_key] = control_input

    data_batch["target_h"], data_batch["target_w"] = target_h // 8, target_w // 8
    data_batch["video"] = torch.zeros((1, 3, 57, H, W), dtype=torch.uint8).cuda()  # ?????
    data_batch["image_size"] = torch.tensor([[H, W, H, W]] * 1, dtype=torch.bfloat16).cuda()
    data_batch["padding_mask"] = torch.zeros((1, 1, H, W), dtype=torch.bfloat16).cuda()

    # add view indices for post-train model
    if num_views == 5:
        mapped_indices = [0, 1, 2, 4, 5]
        view_indices_conditioning = []
        for v_index in mapped_indices:
            view_indices_conditioning.append(torch.ones(num_video_frames, device="cuda") * v_index)
        view_indices_conditioning = torch.cat(view_indices_conditioning, dim=0)
        data_batch["view_indices"] = view_indices_conditioning.unsqueeze(0).contiguous()

    return data_batch


def get_batched_ctrl_batch(
    model,
    prompt_embeddings,  # [B, ...]
    negative_prompt_embeddings,  # [B, ...] or None
    height,
    width,
    fps,
    num_video_frames,
    input_video_paths,  # List[str], length B
    control_inputs_list,  # List[dict], length B
    blur_strength,
    canny_threshold,
):
    """
    Create a fully batched data_batch for video generation, including all control and video inputs.

    Args:
        model: The diffusion model instance.
        prompt_embeddings: [B, ...] tensor of prompt embeddings.
        negative_prompt_embeddings: [B, ...] tensor of negative prompt embeddings or None.
        height, width, fps, num_video_frames: Video parameters.
        input_video_paths: List of input video paths, length B.
        control_inputs_list: List of control input dicts, length B.
        blur_strength, canny_threshold: ControlNet augmentation parameters.

    Returns:
        data_batch: Dict with all fields batched along dim 0 (batch dimension).
        state_shape: List describing the latent state shape.
    """
    B = len(input_video_paths)

    def prepare_single_data_batch(b):
        data_batch = {
            "video": torch.zeros((1, 3, num_video_frames, height, width), dtype=torch.uint8).cuda(),
            "t5_text_mask": torch.ones(1, 512, dtype=torch.bfloat16).cuda(),
            "image_size": torch.tensor([[height, width, height, width]], dtype=torch.bfloat16).cuda(),
            "fps": torch.tensor([fps], dtype=torch.bfloat16).cuda(),
            "num_frames": torch.tensor([num_video_frames], dtype=torch.bfloat16).cuda(),
            "padding_mask": torch.zeros((1, 1, height, width), dtype=torch.bfloat16).cuda(),
            "t5_text_embeddings": prompt_embeddings[b : b + 1].to(dtype=torch.bfloat16).cuda(),
        }
        if negative_prompt_embeddings is not None:
            data_batch["neg_t5_text_embeddings"] = negative_prompt_embeddings[b : b + 1].to(dtype=torch.bfloat16).cuda()
            data_batch["neg_t5_text_mask"] = torch.ones(1, 512, dtype=torch.bfloat16).cuda()
        return data_batch

    # Prepare and process each sample
    single_batches = []
    for b in range(B):
        single_data_batch = prepare_single_data_batch(b)
        processed = get_ctrl_batch(
            model,
            single_data_batch,
            num_video_frames,
            input_video_paths[b],
            control_inputs_list[b],
            blur_strength,
            canny_threshold,
        )
        single_batches.append(processed)

    # Merge all single-sample batches into a batched data_batch
    batched_data_batch = {}
    for k in single_batches[0]:
        if isinstance(single_batches[0][k], torch.Tensor):
            if k == "control_weight" and single_batches[0][k].ndim == 6:
                # [num_controls, 1, 1, T, H, W] per sample
                # Stack along dim=1 to get [num_controls, B, 1, T, H, W]
                batched_data_batch[k] = torch.cat([d[k] for d in single_batches], dim=1)
            else:
                # Concatenate along batch dimension (dim=0) for other tensors
                batched_data_batch[k] = torch.cat([d[k] for d in single_batches], dim=0)
        else:
            batched_data_batch[k] = single_batches[0][k]  # assume they're the same for now

    state_shape = [
        model.tokenizer.channel,
        model.tokenizer.get_latent_num_frames(num_video_frames),
        height // model.tokenizer.spatial_compression_factor,
        width // model.tokenizer.spatial_compression_factor,
    ]

    return batched_data_batch, state_shape


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
    sigma_max: float | None,
    x_sigma_max=None,
    augment_sigma=None,
    use_batch_processing: bool = True,
    save_input_noise: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
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
        save_input_noise (bool): Whether to save input noise

    Returns:
        Generated video frames tensor or tuple of (video frames tensor, input noise tensor)
    """
    assert not model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i, "not supported"

    if augment_sigma is None:
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
        use_batch_processing=use_batch_processing,
        save_input_noise=save_input_noise,
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
    # First find how many vae chunks are contained with in num_input_frames
    num_latent_frames = (
        num_input_frames
        // model.tokenizer.video_vae.pixel_chunk_duration
        * model.tokenizer.video_vae.latent_chunk_duration
    )
    # Then handle the remainder
    if num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration == 1:
        num_latent_frames += 1
    elif num_input_frames % model.tokenizer.video_vae.pixel_chunk_duration > 1:
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
    from_back: bool = True,
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
    num_frames_encode = model.tokenizer.pixel_chunk_duration

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
    if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        condition_frames_first = input_frames[:, :, :num_frames_condition]
        condition_frames_last = input_frames[:, :, -num_frames_condition:]
        padding_frames = condition_frames_first.new_zeros(B, C, num_frames_encode + 1 - 2 * num_frames_condition, H, W)
        encode_input_frames = torch.cat([condition_frames_first, padding_frames, condition_frames_last], dim=2)
    elif not from_back:
        condition_frames = input_frames[:, :, :num_frames_condition]
        padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
        encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2)
    else:
        condition_frames = input_frames[:, :, -num_frames_condition:]
        padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
        encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2)

    log.info(
        f"create latent with input shape {encode_input_frames.shape} including padding {num_frames_encode - num_frames_condition} at the end"
    )
    if hasattr(model, "n_views") and encode_input_frames.shape[0] == model.n_views:
        encode_input_frames = einops.rearrange(encode_input_frames, "(B V) C T H W -> B C (V T) H W", V=model.n_views)
        latent = model.encode(encode_input_frames)
    elif model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        latent1 = model.encode(encode_input_frames[:, :, :num_frames_encode])  # BCTHW
        latent2 = model.encode(encode_input_frames[:, :, num_frames_encode:])
        latent = torch.cat([latent1, latent2], dim=2)  # BCTHW
    elif encode_input_frames.shape[0] == 1:
        # treat as single view video
        latent = model.tokenizer.encode(encode_input_frames) * model.sigma_data
    else:
        raise ValueError(
            f"First dimension of encode_input_frames {encode_input_frames.shape[0]} does not match "
            f"model.n_views or model.n_views is not defined and first dimension is not 1"
        )
    return latent, encode_input_frames


def compute_num_frames_condition(model: DiffusionV2WModel, num_of_latent_overlap: int, downsample_factor=8) -> int:
    """This function computes the number of condition pixel frames given the number of latent frames to overlap.
    Args:
        model (ExtendDiffusionModel): video generation model
        num_of_latent_overlap (int): number of latent frames to overlap
        downsample_factor (int): downsample factor for temporal reduce
    Returns:
        int: number of condition frames in output space
    """
    # For causal tokenizer
    num_frames_condition = (
        num_of_latent_overlap // model.tokenizer.latent_chunk_duration * model.tokenizer.pixel_chunk_duration
    )
    if num_of_latent_overlap % model.tokenizer.latent_chunk_duration == 1:
        num_frames_condition += 1
    elif num_of_latent_overlap % model.tokenizer.latent_chunk_duration > 1:
        num_frames_condition += (
            1 + (num_of_latent_overlap % model.tokenizer.latent_chunk_duration - 1) * downsample_factor
        )
    return num_frames_condition


def get_condition_latent(
    model: DiffusionV2WModel,
    input_image_or_video_path: str,
    num_input_frames: int = 1,
    state_shape: list[int] = None,
    frame_index: int = 0,
    frame_stride: int = 1,
    from_back: bool = True,
    start_frame: int = 0,
) -> torch.Tensor:
    """Get condition latent from input image/video file.

    Args:
        model (DiffusionV2WModel): Video generation model
        input_image_or_video_path (str): Path to conditioning image/video
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

    input_path_format = input_image_or_video_path.split(".")[-1]
    input_frames = read_video_or_image_into_frames_BCTHW(
        input_image_or_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )
    if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        start_frame = frame_index * frame_stride
        end_frame = (frame_index + 1) * frame_stride
        curr_input_frames = torch.cat(
            [input_frames[:, :, start_frame : start_frame + 1], input_frames[:, :, end_frame : end_frame + 1]], dim=2
        ).contiguous()  # BCTHW
        num_of_latent_condition = 1
        num_frames_condition = compute_num_frames_condition(
            model, num_of_latent_condition, downsample_factor=model.tokenizer.temporal_compression_factor
        )

        condition_latent, _ = create_condition_latent_from_input_frames(model, curr_input_frames, num_frames_condition)
        condition_latent = condition_latent.to(torch.bfloat16)
        return condition_latent
    input_frames = input_frames[:, :, start_frame:, :, :]
    condition_latent, _ = create_condition_latent_from_input_frames(
        model, input_frames, num_input_frames, from_back=from_back
    )
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

    weights = torch.stack(weights, dim=0).cuda()
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
    use_distilled = cfg.use_distilled

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
            ckpt_path = os.path.join(checkpoint_dir, default_model_names[hint_key])
            if use_distilled:
                if hint_key in default_distilled_model_names:
                    ckpt_path = os.path.join(checkpoint_dir, default_distilled_model_names[hint_key])
                else:
                    raise ValueError(
                        f"No default distilled checkpoint for {hint_key}. Users must specify ckpt_path in config."
                    )

            config["ckpt_path"] = ckpt_path
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


@contextmanager
def switch_config_for_inference(model: ExtendDiffusionModel):
    """For extend model inference, we need to make sure the condition_location is set to "first_n" and apply_corruption_to_condition_region is False.
    In the interpolator case, condition_location should be "first_and_last_1" for both training and inference.
    This context manager changes the model configuration to the correct settings for inference, and then restores the original settings when exiting the context.
    Args:
        model (ExtendDiffusionModel): video generation model
    """
    # Store the current condition_location
    current_condition_location = model.config.conditioner.video_cond_bool.condition_location
    current_apply_corruption_to_condition_region = (
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region
    )
    try:
        # Change the condition_location to "first_n" for inference, unless it is "first_and_last_1" for interpolator
        if current_condition_location != "first_and_last_1":
            log.info("Change the condition_location to 'first_n' for inference")
            model.config.conditioner.video_cond_bool.condition_location = "first_n"
        if current_apply_corruption_to_condition_region == "gaussian_blur":
            model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = "clean"
            log.info("Change apply_corruption_to_condition_region to clean")
        elif current_apply_corruption_to_condition_region == "noise_with_sigma":
            model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = "noise_with_sigma_fixed"
            log.info("Change apply_corruption_to_condition_region to noise_with_sigma_fixed")
        # Yield control back to the calling context
        yield
    finally:
        # Restore the original condition_location after exiting the context
        log.info(
            f"Restore the original condition_location {current_condition_location}, apply_corruption_to_condition_region {current_apply_corruption_to_condition_region}"
        )
        model.config.conditioner.video_cond_bool.condition_location = current_condition_location
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = (
            current_apply_corruption_to_condition_region
        )


def visualize_latent_tensor_bcthw(tensor, nrow=1, show_norm=False, save_fig_path=None):
    """Debug function to display a latent tensor as a grid of images.
    Args:
        tensor (torch.Tensor): tensor in shape BCTHW
        nrow (int): number of images per row
        show_norm (bool): whether to display the norm of the tensor
        save_fig_path (str): path to save the visualization

    """
    log.info(
        f"display latent tensor shape {tensor.shape}, max={tensor.max()}, min={tensor.min()}, mean={tensor.mean()}, std={tensor.std()}"
    )
    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c (t n) h w -> (b t h) (n w) c", n=nrow)  # .numpy()
    # display the grid
    tensor_mean = tensor.mean(-1)
    tensor_norm = tensor.norm(dim=-1)
    log.info(f"tensor_norm, tensor_mean {tensor_norm.shape}, {tensor_mean.shape}")
    plt.figure(figsize=(20, 20))
    plt.imshow(tensor_mean)
    plt.title(f"mean {tensor_mean.mean()}, std {tensor_mean.std()}")
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        log.info(f"save to {os.path.abspath(save_fig_path)}")
        plt.savefig(save_fig_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    if show_norm:
        plt.figure(figsize=(20, 20))
        plt.imshow(tensor_norm)
        plt.show()


def visualize_tensor_bcthw(tensor: torch.Tensor, nrow=4, save_fig_path=None):
    """Debug function to display a tensor as a grid of images.
    Args:
        tensor (torch.Tensor): tensor in shape BCTHW
        nrow (int): number of images per row
        save_fig_path (str): path to save the visualization
    """
    log.info(f"display {tensor.shape}, {tensor.max()}, {tensor.min()}")
    assert tensor.max() < 200, f"tensor max {tensor.max()} > 200, the data range is likely wrong"
    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c t h w -> (b t) c h w")
    # use torchvision to save the tensor as a grid of images
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)
    if save_fig_path is not None:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        log.info(f"save to {os.path.abspath(save_fig_path)}")
        torchvision.utils.save_image(tensor, save_fig_path)
    # display the grid
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def generate_video_from_batch_with_loop(
    model: ExtendDiffusionModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    condition_latent: torch.Tensor,
    # hyper-parameters for inference
    num_of_loops: int,
    num_of_latent_overlap_list: list[int],
    guidance: float,
    num_steps: int,
    seed: int,
    add_input_frames_guidance: bool = False,
    augment_sigma_list: list[float] = None,
    data_batch_list: Union[None, list[dict]] = None,
    visualize: bool = False,
    save_fig_path: str = None,
    skip_reencode: int = 0,
    return_noise: bool = False,
    **extra_generate_kwargs,
) -> Tuple[np.array, list, list, torch.Tensor] | Tuple[np.array, list, list, torch.Tensor, torch.Tensor]:
    """Generate video with loop, given data batch. The condition latent will be updated at each loop.
    Args:
        model (ExtendDiffusionModel)
        state_shape (list): shape of the state tensor
        is_negative_prompt (bool): whether to use negative prompt

        data_batch (dict): data batch for video generation
        condition_latent (torch.Tensor): condition latent in shape BCTHW

        num_of_loops (int): number of loops to generate video
        num_of_latent_overlap_list (list[int]): list number of latent frames to overlap between clips, different clips can have different overlap
        guidance (float): The guidance scale to use during sample generation; defaults to 5.0.
        num_steps (int): number of steps for diffusion sampling
        seed (int): random seed for sampling
        add_input_frames_guidance (bool): whether to add image guidance, default is False
        augment_sigma_list (list): list of sigma value for the condition corruption at different clip, used when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed". default is None

        data_batch_list (list): list of data batch for video generation, used when num_of_loops >= 1, to support multiple prompts in auto-regressive generation. default is None
        visualize (bool): whether to visualize the latent and grid, default is False
        save_fig_path (str): path to save the visualization, default is None

        skip_reencode (int): whether to skip re-encode the input frames, default is 0
        return_noise (bool): whether to return the initial noise used for sampling, used for ODE pairs generation. Default is False
    Returns:
        np.array: generated video in shape THWC, range [0, 255]
        list: list of condition latent, each in shape BCTHW
        list: list of sample latent, each in shape BCTHW
        torch.Tensor: initial noise used for sampling, shape BCTHW (if return_noise is True)
    """
    if data_batch_list is None:
        data_batch_list = [data_batch for _ in range(num_of_loops)]
    if visualize:
        assert save_fig_path is not None, "save_fig_path should be set when visualize is True"

    # Generate video with loop
    condition_latent_list = []
    decode_latent_list = []  # list collect the latent token to be decoded at the end
    sample_latent = []
    grid_list = []

    augment_sigma_list = (
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region_sigma_value
        if augment_sigma_list is None
        else augment_sigma_list
    )

    for i in range(num_of_loops):
        num_of_latent_overlap_i = num_of_latent_overlap_list[i]
        num_of_latent_overlap_i_plus_1 = (
            num_of_latent_overlap_list[i + 1]
            if i + 1 < len(num_of_latent_overlap_list)
            else num_of_latent_overlap_list[-1]
        )
        if condition_latent.shape[2] < state_shape[1]:
            # Padding condition latent to state shape
            log.info(f"Padding condition latent {condition_latent.shape} to state shape {state_shape}")
            b, c, t, h, w = condition_latent.shape
            condition_latent = torch.cat(
                [
                    condition_latent[:, :, :1],
                    condition_latent.new_zeros(b, c, state_shape[1] - t, h, w),
                    condition_latent[:, :, -1:],
                ],
                dim=2,
            ).contiguous()
            log.info(f"after padding, condition latent shape {condition_latent.shape}")
        log.info(f"Generate video loop {i} / {num_of_loops}")
        if visualize:
            log.info(f"Visualize condition latent {i}")
            visualize_latent_tensor_bcthw(
                condition_latent[:, :, :4].float(),
                nrow=4,
                save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_condition_latent_first_4.png"),
            )  # BCTHW

        condition_latent_list.append(condition_latent)

        if i < len(augment_sigma_list):
            condition_video_augment_sigma_in_inference = augment_sigma_list[i]
            log.info(f"condition_video_augment_sigma_in_inference {condition_video_augment_sigma_in_inference}")
        else:
            condition_video_augment_sigma_in_inference = augment_sigma_list[-1]
        assert not add_input_frames_guidance, "add_input_frames_guidance should be False, not supported"
        sample = model.generate_samples_from_batch(
            data_batch_list[i],
            guidance=guidance,
            state_shape=state_shape,
            num_steps=num_steps,
            is_negative_prompt=is_negative_prompt,
            seed=seed + i,
            condition_latent=condition_latent,
            num_condition_t=num_of_latent_overlap_i,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            return_noise=return_noise,
            **extra_generate_kwargs,
        )

        if return_noise:
            sample, noise = sample

        if visualize:
            log.info(f"Visualize sampled latent {i} 4-8 frames")
            visualize_latent_tensor_bcthw(
                sample[:, :, 4:8].float(),
                nrow=4,
                save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_sample_latent_last_4.png"),
            )  # BCTHW

            diff_between_sample_and_condition = (sample - condition_latent)[:, :, :num_of_latent_overlap_i]
            log.info(
                f"Visualize diff between sample and condition latent {i} first 4 frames {diff_between_sample_and_condition.mean()}"
            )

        sample_latent.append(sample)
        T = condition_latent.shape[2]
        assert num_of_latent_overlap_i <= T, f"num_of_latent_overlap should be < T, get {num_of_latent_overlap_i}, {T}"

        if model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
            assert skip_reencode, "skip_reencode should be turned on when sample_tokens_start_from_p_or_i is True"
            if i == 0:
                decode_latent_list.append(sample)
            else:
                decode_latent_list.append(sample[:, :, num_of_latent_overlap_i:])
        else:
            # Interpolator mode. Decode the first and last as an image.
            # each decode should operate on pixe_chunk_duration, otherwise the output will be incorrect.
            # Interpolator works on pixel_chunk_duration==1.
            # the root cause is the mean,std will be incorrect if decode a size different from pixel_chunk_duration.
            if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
                grid_BCTHW_list = []
                chunk_size = model.tokenizer.video_vae.pixel_chunk_duration
                for idx in range(sample.shape[2], chunk_size):
                    grid_BCTHW = (1.0 + model.decode(sample[:, :, idx : idx + chunk_size, ...])).clamp(
                        0, 2
                    ) / 2  # [B, 3, 1, H, W], [0, 1]
                    grid_BCTHW_list.append(grid_BCTHW)
                grid_BCTHW = torch.cat(grid_BCTHW_list, dim=2)  # [B, 3, T, H, W], [0, 1]
            else:
                grid_BCTHW = (1.0 + model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W], [0, 1]

            if visualize:
                log.info(f"Visualize grid {i}")
                visualize_tensor_bcthw(
                    grid_BCTHW.float(), nrow=5, save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_grid.png")
                )
            grid_np_THWC = (
                (grid_BCTHW[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
            )  # THW3, range [0, 255]

            # Post-process the output: cut the conditional frames from the output if it's not the first loop
            num_cond_frames = compute_num_frames_condition(
                model, num_of_latent_overlap_i_plus_1, downsample_factor=model.tokenizer.temporal_compression_factor
            )
            if i == 0:
                new_grid_np_THWC = grid_np_THWC  # First output, dont cut the conditional frames
            else:
                new_grid_np_THWC = grid_np_THWC[
                    num_cond_frames:
                ]  # Remove the conditional frames from the output, since it's overlapped with previous loop
            grid_list.append(new_grid_np_THWC)

            # Prepare the next loop: re-compute the condition latent
            if hasattr(model, "n_cameras"):
                grid_BCTHW = einops.rearrange(grid_BCTHW, "B C (V T) H W -> (B V) C T H W", V=model.n_cameras)
            condition_frame_input = grid_BCTHW[:, :, -num_cond_frames:] * 2 - 1  # BCTHW, range [0, 1] to [-1, 1]
        if skip_reencode:
            # Use the last num_of_latent_overlap latent token as condition latent
            log.info(f"Skip re-encode the condition frames, use the last {num_of_latent_overlap_i_plus_1} latent token")
            condition_latent = sample[:, :, -num_of_latent_overlap_i_plus_1:]
        else:
            # Re-encode the condition frames to get the new condition latent
            condition_latent, _ = create_condition_latent_from_input_frames(
                model, condition_frame_input, num_frames_condition=num_cond_frames
            )  # BCTHW
        condition_latent = condition_latent.to(torch.bfloat16)

    # save videos
    if model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
        # decode all video together
        decode_latent_list = torch.cat(decode_latent_list, dim=2)
        grid_BCTHW = (1.0 + model.decode(decode_latent_list)).clamp(0, 2) / 2  # [B, 3, T, H, W], [0, 1]
        video_THWC = (
            (grid_BCTHW[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
        )  # THW3, range [0, 255]
    else:
        video_THWC = np.concatenate(grid_list, axis=0)  # THW3, range [0, 255]

    if return_noise:
        return video_THWC, condition_latent_list, sample_latent, noise
    return video_THWC, condition_latent_list, sample_latent
