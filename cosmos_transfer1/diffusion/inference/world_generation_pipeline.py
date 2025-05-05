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

import os
import cv2
from typing import Optional
import einops

import numpy as np
import torch
from tqdm import tqdm

#from cosmos_transfer1.auxiliary.upsampler.model.upsampler import PixtralPromptUpsampler
from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_7B_CHECKPOINT_PATH,
    COSMOS_TOKENIZER_CHECKPOINT,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_t2v_BASE_CHECKPOINT_AV_SAMPLE_PATH_dbg,
    SV2MV_t2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg,
    SV2MV_t2v_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_i2v_BASE_CHECKPOINT_AV_SAMPLE_PATH_dbg,
    SV2MV_i2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg
)
from cosmos_transfer1.diffusion.inference.inference_utils import (
    detect_aspect_ratio,
    generate_control_input,
    generate_world_from_control,
    get_ctrl_batch,
    get_ctrl_batch_mv,
    get_upscale_size,
    get_video_batch,
    get_video_batch_for_multiview_model,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
    merge_patches_into_video,
    non_strict_load_model,
    resize_control_weight_map,
    split_video_into_patches,
    read_and_resize_input,
    resize_video,
    read_video_or_image_into_frames_BCTHW,
    valid_hint_keys
)
from cosmos_transfer1.diffusion.model.model_ctrl import VideoDiffusionModelWithCtrl, VideoDiffusionT2VModelWithCtrl
from cosmos_transfer1.diffusion.model.model_multi_camera_ctrl import MultiVideoDiffusionModelWithCtrl
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.base_world_generation_pipeline import BaseWorldGenerationPipeline

MODEL_NAME_DICT = {
    BASE_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_vis_block3",
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3",
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_keypoint_block3",
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3",
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_upscale_block3",
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_lidar_block3",
    SV2MV_t2v_BASE_CHECKPOINT_AV_SAMPLE_PATH_dbg: "CTRL_7Bv1pt3_mv_t2v_57frames_control_input_hdmap_block3",
    SV2MV_t2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg: "CTRL_7Bv1pt3_mv_t2v_57frames_control_input_hdmap_block3",
    SV2MV_i2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg: "CTRL_7Bv1pt3_mv_t2v_57frames_control_input_hdmap_block3",
    "": "CTRL_7Bv1pt3_mv_t2v_57frames_control_input_hdmap_block3_SMALL",
}
MODEL_CLASS_DICT = {
    BASE_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: VideoDiffusionT2VModelWithCtrl,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    SV2MV_t2v_BASE_CHECKPOINT_AV_SAMPLE_PATH_dbg: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg: MultiVideoDiffusionModelWithCtrl,
    SV2MV_i2v_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH_dbg: MultiVideoDiffusionModelWithCtrl,
    "": MultiVideoDiffusionModelWithCtrl
}


class DiffusionControl2WorldGenerationPipeline(BaseWorldGenerationPipeline):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
        control_inputs: dict = None,
        sigma_max: float = 70.0,
        blur_strength: str = "medium",
        canny_threshold: str = "medium",
        upsample_prompt: bool = False,
        offload_prompt_upsampler: bool = False,
        process_group: torch.distributed.ProcessGroup | None = None,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
            control_inputs: Dictionary of control inputs for guided generation
            sigma_max: Sigma max for partial denoising
            blur_strength: Strength of blur applied to input
            canny_threshold: Threshold for edge detection
            upsample_prompt: Whether to upsample prompts using prompt upsampler model
            offload_prompt_upsampler: Whether to offload prompt upsampler after use
            process_group: Process group for distributed training
        """
        self.num_input_frames = num_input_frames
        self.control_inputs = control_inputs
        self.sigma_max = sigma_max
        self.blur_strength = blur_strength
        self.canny_threshold = canny_threshold
        self.upsample_prompt = upsample_prompt
        self.offload_prompt_upsampler = offload_prompt_upsampler
        self.prompt_upsampler = None
        self.upsampler_hint_key = None
        self.hint_details = None
        self.process_group = process_group

        self.model_name = MODEL_NAME_DICT[checkpoint_name]
        self.model_class = MODEL_CLASS_DICT[checkpoint_name]
        self.guidance = guidance
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.seed = seed

        super().__init__(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_guardrail_models=offload_guardrail_models,
        )

        # Initialize prompt upsampler if needed
        if self.upsample_prompt:
            if int(os.environ["RANK"]) == 0:
                self._push_torchrun_environ_variables()
                self._init_prompt_upsampler()
                self._pop_torchrun_environ_variables()

    def _push_torchrun_environ_variables(self):
        dist_keys = [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_NAME",
            "OMP_NUM_THREADS",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCHELASTIC_ERROR_FILE",
        ]

        self.torchrun_environ_variables = {}
        for dist_key in dist_keys:
            if dist_key in os.environ:
                self.torchrun_environ_variables[dist_key] = os.environ[dist_key]
                del os.environ[dist_key]

    def _pop_torchrun_environ_variables(self):
        for dist_key in self.torchrun_environ_variables.keys():
            os.environ[dist_key] = self.torchrun_environ_variables[dist_key]

    def _init_prompt_upsampler(self):
        """
        Initializes the prompt upsampler based on the provided control inputs.

        Returns:
            None: Sets instance variables for prompt upsampler, hint key, and hint details
        """
        vis_hint_keys = ["vis", "edge"]
        other_hint_keys = ["seg", "depth"]
        self.hint_details = None

        log.info("Initializing prompt upsampler...")

        if any(key in vis_hint_keys for key in self.control_inputs):
            self.upsampler_hint_key = "vis"
            self.hint_details = "vis" if "vis" in self.control_inputs else "edge"
        elif any(key in other_hint_keys for key in self.control_inputs):
            selected_hint_keys = [key for key in self.control_inputs if key in other_hint_keys]
            self.upsampler_hint_key = selected_hint_keys[0]
        else:
            self.upsampler_hint_key = None

        if self.upsampler_hint_key:
            self.prompt_upsampler = PixtralPromptUpsampler(
                checkpoint_dir=self.checkpoint_dir,
                offload_prompt_upsampler=self.offload_prompt_upsampler,
            )

        log.info(
            f"Prompt upsampler initialized with hint key: {self.upsampler_hint_key} and hint details: {self.hint_details}"
        )

    def _process_prompt_upsampler(self, prompt, video_path, save_folder):
        """
        Processes and upscales a given prompt using the prompt upsampler.

        Args:
            prompt: The text prompt to upsample
            video_path: Path to the input video
            save_folder: Folder to save intermediate files

        Returns:
            str: The upsampled prompt
        """
        if not self.prompt_upsampler:
            return prompt

        log.info(f"Upsampling prompt with controlnet: {self.upsampler_hint_key}")

        if self.upsampler_hint_key in ["vis"]:  # input video or control input, one of them is required
            # prompt upsampler for viscontrol(vis, edge)
            if self.control_inputs[self.hint_details].get("input_control", None) is not None:
                input_control_path = self.control_inputs[self.hint_details].get("input_control", None)
            else:
                hint_key = f"control_input_{self.hint_details}"
                input_control_path = generate_control_input(
                    input_file_path=video_path,
                    save_folder=save_folder,
                    hint_key=hint_key,
                    blur_strength=self.blur_strength,
                    canny_threshold=self.canny_threshold,
                )
        else:
            # prompt upsampler for seg, depth
            input_control_path = self.control_inputs[self.upsampler_hint_key].get("input_control", None)

        prompt = self.prompt_upsampler._prompt_upsample_with_offload(prompt=prompt, video_path=input_control_path)
        return prompt

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_transfer1/diffusion/config/transfer/config.py",
            model_class=self.model_class,
            base_checkpoint_dir=self.checkpoint_dir,
        )

    # load the hint encoders. these encoders are run along with the main model to provide additional context
    def _load_network(self):
        # This load seems to be non-functional for av-sample checkpoints. The base_model loading in build_model is required
        if self.checkpoint_name == "":
            load_network_model(self.model, "")
        else:
            load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}")
        if len(self.control_inputs) > 1:
            hint_encoders = torch.nn.ModuleList([])
            for key, spec in self.control_inputs.items():
                if key in valid_hint_keys:
                    model = load_model_by_config(
                        config_job_name=self.model_name,
                        config_file="cosmos_transfer1/diffusion/config/transfer/config.py",
                        model_class=self.model_class,
                        base_checkpoint_dir=self.checkpoint_dir,
                    )
                    load_network_model(model, spec["ckpt_path"])
                    hint_encoders.append(model.model.net)
                    del model
                    torch.cuda.empty_cache()
            self.model.hint_encoders = hint_encoders
        else:
            for _, spec in self.control_inputs.items():
                net_state_dict = torch.load(
                    spec["ckpt_path"], map_location="cpu", weights_only=False
                )  # , weights_only=True)
                non_strict_load_model(self.model.model, net_state_dict)

        if self.process_group is not None:
            self.model.model.net.enable_context_parallel(self.process_group)
            self.model.model.base_model.net.enable_context_parallel(self.process_group)
            if hasattr(self.model.model, "hint_encoders"):
                self.model.model.hint_encoders.net.enable_context_parallel(self.process_group)

    def _load_tokenizer(self):
        load_tokenizer_model(self.model, f"{self.checkpoint_dir}/{COSMOS_TOKENIZER_CHECKPOINT}")

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        if sample.shape[0] == 1:
            video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        else:
            # Do decoding for each batch sequentially to prevent OOM.
            samples = []
            for sample_i in sample:
                samples += [self.model.decode(sample_i.unsqueeze(0)).cpu()]
            samples = (torch.cat(samples) + 1).clamp(0, 2) / 2

            # Stitch the patches together to form the final video.
            patch_h, patch_w = samples.shape[-2:]
            orig_size = (patch_w, patch_h)
            aspect_ratio = detect_aspect_ratio(orig_size)
            stitch_w, stitch_h = get_upscale_size(orig_size, aspect_ratio, upscale_factor=3)
            n_img_w = (stitch_w - 1) // patch_w + 1
            n_img_h = (stitch_h - 1) // patch_h + 1
            overlap_size_w = overlap_size_h = 0
            if n_img_w > 1:
                overlap_size_w = (n_img_w * patch_w - stitch_w) // (n_img_w - 1)
            if n_img_h > 1:
                overlap_size_h = (n_img_h * patch_h - stitch_h) // (n_img_h - 1)
            video = merge_patches_into_video(samples, overlap_size_h, overlap_size_w, n_img_h, n_img_w)
            video = torch.nn.functional.interpolate(video[0], size=(patch_h * 3, patch_w * 3), mode="bicubic")[None]
            video = video.clamp(0, 1)

        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return video

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
        control_inputs: dict = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            video_path: Path to input video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding, negative_prompt_embedding, video_path, control_inputs)

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
        video_path="",
        control_inputs: dict = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        data_batch, _ = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )
        data_batch = get_ctrl_batch(
            self.model,
            data_batch,
            self.num_video_frames,
            video_path,
            control_inputs,
            self.blur_strength,
            self.canny_threshold,
        )

        hint_key = data_batch["hint_key"]
        input_video = data_batch["input_video"]
        control_input = data_batch[hint_key]
        control_weight = data_batch["control_weight"]
        num_new_generated_frames = self.num_video_frames - self.num_input_frames
        B, C, T, H, W = control_input.shape
        if (T - self.num_input_frames) % num_new_generated_frames != 0:  # pad duplicate frames at the end
            pad_t = num_new_generated_frames - ((T - self.num_input_frames) % num_new_generated_frames)
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            if input_video is not None:
                pad_video = input_video[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                input_video = torch.cat([input_video, pad_video], dim=2)
            num_total_frames_with_padding = control_input.shape[2]
            if (
                isinstance(control_weight, torch.Tensor)
                and control_weight.ndim > 5
                and num_total_frames_with_padding > control_weight.shape[3]
            ):
                pad_t = num_total_frames_with_padding - control_weight.shape[3]
                pad_weight = control_weight[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)
                control_weight = torch.cat([control_weight, pad_weight], dim=3)
        else:
            num_total_frames_with_padding = T
        N_clip = (num_total_frames_with_padding - self.num_input_frames) // num_new_generated_frames

        video = []
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            if input_video is not None:
                x_sigma_max = []
                for b in range(B):
                    input_frames = input_video[b : b + 1, :, start_frame:end_frame].cuda()
                    x0 = self.model.encode(input_frames).contiguous()
                    x_sigma_max.append(self.model.get_x_from_clean(x0, self.sigma_max, seed=(self.seed + i_clip)))
                x_sigma_max = torch.cat(x_sigma_max)
            else:
                x_sigma_max = None

            data_batch_i[hint_key] = control_input[:, :, start_frame:end_frame].cuda()
            latent_hint = []
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)

            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :].cuda()
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(control_weight_t, (t, h // 2, w // 2))

            if i_clip == 0:
                num_input_frames = 0
                latent_tmp = latent_hint if latent_hint.ndim == 5 else latent_hint[:, 0]
                condition_latent = torch.zeros_like(latent_tmp)
            else:
                num_input_frames = self.num_input_frames
                prev_frames = split_video_into_patches(prev_frames, control_input.shape[-2], control_input.shape[-1])
                condition_latent = []
                for b in range(B):
                    input_frames = prev_frames[b : b + 1].cuda().bfloat16() / 255.0 * 2 - 1
                    condition_latent += [self.model.encode(input_frames).contiguous()]
                condition_latent = torch.cat(condition_latent)

            # Generate video frames
            latents = generate_world_from_control(
                model=self.model,
                state_shape=self.model.state_shape,
                is_negative_prompt=True,
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                seed=(self.seed + i_clip),
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=self.sigma_max if x_sigma_max is not None else None,
                x_sigma_max=x_sigma_max,
            )
            frames = self._run_tokenizer_decoding(latents)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2)[None]

            if i_clip == 0:
                video.append(frames)
            else:
                video.append(frames[:, :, self.num_input_frames :])
            prev_frames = torch.zeros_like(frames)
            prev_frames[:, :, : self.num_input_frames] = frames[:, :, -self.num_input_frames :]

        video = torch.cat(video, dim=2)[:, :, :T]
        video = video[0].permute(1, 2, 3, 0).numpy()
        return video

    def generate(
        self,
        prompt: str,
        video_path: str,
        negative_prompt: Optional[str] = None,
        control_inputs: dict = None,
        save_folder: str = "outputs/",
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and control video.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Convert prompt to embeddings
        3. Generate video frames using diffusion
        4. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            video_path: Path to input video
            negative_prompt: Optional text to guide what not to generate
            control_inputs: Control inputs for guided generation
            save_folder: Folder to save intermediate files

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with video path: {video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")

        # Upsample prompt if enabled
        if self.prompt_upsampler:
            if int(os.environ["RANK"]) == 0:
                self._push_torchrun_environ_variables()
                prompt = self._process_prompt_upsampler(prompt, video_path, save_folder)
                self._pop_torchrun_environ_variables()
                log.info(f"Upsampled prompt: {prompt}")

        log.info("Run guardrail on prompt")
        is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
        if not is_safe:
            log.critical("Input text prompt is not safe")
            return None
        log.info("Pass guardrail on prompt")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            video_path=video_path,
            control_inputs=control_inputs,
        )
        log.info("Finish generation")

        log.info("Run guardrail on generated video")
        video = self._run_guardrail_on_video_with_offload(video)
        if video is None:
            log.critical("Generated video is not safe")
            raise ValueError("Guardrail check failed: Generated video is unsafe")

        log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionControl2WorldMultiviewGenerationPipeline(DiffusionControl2WorldGenerationPipeline):

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video_segments = einops.rearrange(video, "b c (v t) h w -> b c v t h w", v=self.model.n_views)
        grid_video = torch.stack(
            [video_segments[:, :, i] for i in [1, 0, 2, 4, 3, 5]],
            dim=2,
        )
        grid_video = einops.rearrange(grid_video, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=3)
        grid_video = (grid_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return [grid_video, video]

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        video_path="",
        control_inputs: dict = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            video_path: Path to input video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding,
                                 video_path=video_path,
                                 control_inputs=control_inputs
                                 )

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def _run_model(
        self,
        embedding: torch.Tensor,
        video_path: str,
        control_inputs: dict = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        #model, prompt_embedding, height, width, fps, num_video_frames, frame_repeat_negative_condition
        assert len(embedding) == self.model.n_views
        data_batch, state_shape = get_video_batch_for_multiview_model(
            model=self.model,
            prompt_embedding=embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(embedding),
            frame_repeat_negative_condition=0,
        )
        requisite_input_views = [0]
        self.model.condition_location = "first_cam"

        # view_condition_video, fps, aspect_ratio = read_and_resize_input(
        #     video_path, num_total_frames=self.num_video_frames, interpolation=cv2.INTER_LINEAR
        # )
        view_condition_video, fps = read_video_or_image_into_frames_BCTHW(
            video_path,
            normalize=False,  # s.t. output range is [0, 255]
            max_frames=self.num_video_frames,
            also_return_fps=True,
        )
        view_condition_video = resize_video(view_condition_video, self.height, self.width, interpolation=cv2.INTER_LINEAR)
        view_condition_video = torch.from_numpy(view_condition_video)
        total_T = view_condition_video.shape[2]

        data_batch = get_ctrl_batch_mv( #model, data_batch, num_total_frames, control_inputs
            self.height,
            self.width,
            data_batch,
            total_T, #self.num_video_frames,
            control_inputs
        )
        """
        data_batch["control_weight"] = control_weights

        if len(control_inputs) > 1:  # Multicontrol enabled
            data_batch["hint_key"] = "control_input_multi"
            data_batch["control_input_multi"] = control_input
        else:  # Single-control case
            data_batch["hint_key"] = hint_key
            data_batch[hint_key] = control_input

        data_batch["target_h"], data_batch["target_w"] = target_h // 8, target_w // 8
        data_batch["video"] = torch.zeros((1, 3, 121, H, W), dtype=torch.uint8).cuda()  #?????
        data_batch["image_size"] = torch.tensor([[H, W, H, W]] * 1, dtype=torch.bfloat16).cuda()
        data_batch["padding_mask"] = torch.zeros((1, 1, H, W), dtype=torch.bfloat16).cuda()
        """
        hint_key = data_batch["hint_key"]
        input_video = None
        control_input = data_batch[hint_key]
        control_weight = data_batch["control_weight"]

        num_new_generated_frames = self.num_video_frames - self.num_input_frames
        B, C, T, H, W = control_input.shape
        T = T // self.model.n_views
        if (T - self.num_input_frames) % num_new_generated_frames != 0:  # pad duplicate frames at the end
            pad_t = num_new_generated_frames - ((T - self.num_input_frames) % num_new_generated_frames)
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            if input_video is not None:
                pad_video = input_video[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                input_video = torch.cat([input_video, pad_video], dim=2)
            num_total_frames_with_padding = control_input.shape[2] // self.model.n_views
            if (
                isinstance(control_weight, torch.Tensor)
                and control_weight.ndim > 5
                and num_total_frames_with_padding > control_weight.shape[3]
            ):
                pad_t = num_total_frames_with_padding - control_weight.shape[3]
                pad_weight = control_weight[:, :, :, -1:].repeat(1, 1, 1, pad_t, 1, 1)
                control_weight = torch.cat([control_weight, pad_weight], dim=3)
        else:
            num_total_frames_with_padding = T
        N_clip = (num_total_frames_with_padding - self.num_input_frames) // num_new_generated_frames

        video = []
        for i_clip in tqdm(range(N_clip)):
            data_batch_i = {k: v for k, v in data_batch.items()}
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + self.num_input_frames

            if input_video is not None:
                x_sigma_max = []
                for b in range(B):
                    input_frames = input_video[b : b + 1, :, start_frame:end_frame].cuda()
                    x0 = self.model.encode(input_frames).contiguous()
                    x_sigma_max.append(self.model.get_x_from_clean(x0, self.sigma_max, seed=(self.seed + i_clip)))
                x_sigma_max = torch.cat(x_sigma_max)
            else:
                x_sigma_max = None

            control_input_BVCT = einops.rearrange(control_input, "B C (V T) H W -> (B V) C T H W", V=self.model.n_views)
            control_input_i = control_input_BVCT[:, :, start_frame: end_frame].cuda()

            data_batch_i[hint_key] = einops.rearrange(
                control_input_i, "(B V) C T H W -> B C (V T) H W", V=self.model.n_views
            )

            condition_input_i = view_condition_video[:, :, start_frame: end_frame].cuda()

            latent_hint = []
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[hint_key] = data_batch_i[hint_key][b : b + 1]
                if len(control_inputs) > 1:
                    latent_hint_i = []
                    for idx in range(0, data_batch_p[hint_key].size(1), 3):
                        x_rgb = data_batch_p[hint_key][:, idx : idx + 3]
                        latent_hint_i.append(self.model.encode(x_rgb))
                    latent_hint.append(torch.cat(latent_hint_i).unsqueeze(0))
                else:
                    latent_hint.append(self.model.encode_latent(data_batch_p))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint)

            if isinstance(control_weight, torch.Tensor) and control_weight.ndim > 4:
                control_weight_t = control_weight[..., start_frame:end_frame, :, :].cuda()
                t, h, w = latent_hint.shape[-3:]
                data_batch_i["control_weight"] = resize_control_weight_map(control_weight_t, (t, h // 2, w // 2))

            if i_clip == 0:
                #num_condition_t = 0
                num_input_frames = 0
                #condition_video_augment_sigma_in_inference = 0.0
                prev_frames = None
            else:
                #num_condition_t = (self.num_input_frames - 1) // self.model.tokenizer.temporal_compression_factor + 1
                num_input_frames = self.num_input_frames
                #condition_video_augment_sigma_in_inference = 0.0
            condition_latent = self.get_condition_latent(state_shape, data_batch_i,
                                                           cond_video=condition_input_i,
                                                           prev_frames=prev_frames,
                                                           patch_h=H,
                                                           patch_w=W,
                                                           skip_reencode=False,
                                                           ).bfloat16()
            # Generate video frames
            latents = generate_world_from_control(
                model=self.model,
                state_shape=self.model.state_shape,
                is_negative_prompt=True,
                data_batch=data_batch_i,
                guidance=self.guidance,
                num_steps=self.num_steps,
                seed=(self.seed + i_clip),
                condition_latent=condition_latent,
                num_input_frames=num_input_frames,
                sigma_max=self.sigma_max if x_sigma_max is not None else None,
                x_sigma_max=x_sigma_max,
            )
            torch.cuda.empty_cache()
            _, frames = self._run_tokenizer_decoding(latents)   # T H W C
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2)[None] # 1 C T H W

            if i_clip == 0:
                video.append(frames)
            else:
                video.append(frames[:, :, self.num_input_frames :])
            prev_frames = torch.zeros_like(frames)
            prev_frames[:, :, : self.num_input_frames] = frames[:, :, -self.num_input_frames :]

        video = torch.cat(video, dim=2)[:, :, :(T * self.model.n_views)]
        video = video[0].permute(1, 2, 3, 0).numpy()    # T H W C
        return video

    def get_condition_latent(
        self,
        state_shape,
        data_batch_i,
        cond_video=None,
        prev_frames=None,
        patch_h=1024,
        patch_w=1024,
        skip_reencode=False,
        prev_latents=None,
    ):
        """
        Create the condition latent used in this loop for generation from RGB frames
        Args:
            model:
            state_shape: tuple (C T H W), shape of latent to be generated
            data_batch_i: (dict) this is only used to get batch size
            multi_cam: (bool) whether to use multicam processing or revert to original behavior from tpsp_demo
            cond_video: (tensor) the front view video for conditioning sv2mv
            prev_frames: (tensor) frames generated in previous loop
            patch_h: (int)
            patch_w: (int)
            skip_reencode: (bool) whether to use the tokenizer to encode prev_frames, or read from prev_latents directly
            prev_latents: (tensor) latent generated in previous loop, must not be None if skip_reencode

        Returns:

        """
        # this might be not 1 when patching is used
        B = data_batch_i["video"].shape[0]

        latent_sample = torch.zeros(state_shape).unsqueeze(0).repeat(B, 1, 1, 1, 1).cuda()  # B, C, (V T), H, W
        latent_sample = einops.rearrange(latent_sample, "B C (V T) H W -> B V C T H W", V=self.model.n_views)
        log.info(f"model.sigma_data {self.model.sigma_data}")
        if self.model.config.conditioner.video_cond_bool.condition_location.endswith("first_n"):
            if skip_reencode:
                assert prev_latents is not None
                prev_latents = einops.rearrange(prev_latents, "B C (V T) H W -> B V C T H W", V=self.model.n_views)
                latent_sample = prev_latents.clone()
            else:
                prev_frames = split_video_into_patches(prev_frames, patch_h, patch_w)
                for b in range(prev_frames.shape[0]):
                    input_frames = prev_frames[b : b + 1].cuda() / 255.0 * 2 - 1
                    input_frames = einops.rearrange(input_frames, "1 C (V T) H W -> V C T H W", V=self.model.n_views)
                    encoded_frames = self.model.tokenizer.encode(input_frames).contiguous() * self.model.sigma_data
                    latent_sample[b : b + 1, :] = encoded_frames

        if self.model.config.conditioner.video_cond_bool.condition_location.startswith("first_cam"):
            assert cond_video is not None
            cond_video = split_video_into_patches(cond_video, patch_h, patch_w)
            for b in range(cond_video.shape[0]):
                input_frames = cond_video[b : b + 1].cuda() / 255.0 * 2 - 1
                #input_frames = einops.rearrange(input_frames, "1 C (V T) H W -> V C T H W", V=self.model.n_views)[:1]
                latent_sample[
                    b : b + 1,
                    0,
                ] = (
                    self.model.tokenizer.encode(input_frames).contiguous() * self.model.sigma_data
                )

        latent_sample = einops.rearrange(latent_sample, " B V C T H W -> B C (V T) H W")
        log.info(f"latent_sample, {latent_sample[:,0,:,0,0]}")

        return latent_sample

    def build_mv_prompt(self, mv_prompts, n_views):
        """
        Apply multiview prompt formatting to the input prompt such that hte text conditioning matches that used during
        training.
        Args:
            prompt: caption of one scene, with prompt of each view separated by ";"
            n_views: number of cameras to format the caption to

        Returns:

        """
        base_prompts = [
            "The video is captured from a camera mounted on a car. The camera is facing forward. ",
            "The video is captured from a camera mounted on a car. The camera is facing to the left. ",
            "The video is captured from a camera mounted on a car. The camera is facing to the right. ",
            "The video is captured from a camera mounted on a car. The camera is facing backwards. ",
            "The video is captured from a camera mounted on a car. The camera is facing the rear left side. ",
            "The video is captured from a camera mounted on a car. The camera is facing the rear right side. ",
        ]

        log.info(f"Reading multiview prompts, found {len(mv_prompts)} splits")
        n = len(mv_prompts)
        if n < n_views:
            mv_prompts += base_prompts[n:]
        else:
            mv_prompts = mv_prompts[:n_views]

        for vid, p in enumerate(mv_prompts):
            if not p.startswith(base_prompts[vid]):
                mv_prompts[vid] = base_prompts[vid] + " " + p
                log.info(f"Adding missing camera caption to view {vid}, {p[:30]}")

        log.info(f"Procced multiview prompts, {len(mv_prompts)} splits")
        return mv_prompts

    def generate(
        self,
        prompts: list,
        video_path: str,
        control_inputs: dict = None,
        save_folder: str = "outputs/",
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and control video.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Convert prompt to embeddings
        3. Generate video frames using diffusion
        4. Run safety checks and apply face blur on generated video frames

        Args:
            control_inputs: Control inputs for guided generation
            save_folder: Folder to save intermediate files

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """


        log.info(f"Run with video path: {video_path}")

        # Process prompts into multiview format
        if False:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")

        mv_prompts = self.build_mv_prompt(prompts, self.model.n_views)
        log.info(f"Run with prompt: {mv_prompts}")

        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(mv_prompts)
        prompt_embedding = torch.concat(prompt_embeddings, dim=0).cuda()

        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")

        video = self._run_model_with_offload(
            prompt_embedding,
            video_path,
            control_inputs=control_inputs,
        )
        log.info("Finish generation")
        if False:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                raise ValueError("Guardrail check failed: Generated video is unsafe")

            log.info("Pass guardrail on generated video")

        return video, mv_prompts