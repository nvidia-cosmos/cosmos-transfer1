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

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_transfer1.diffusion.conditioner import VideoExtendCondition
from cosmos_transfer1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_transfer1.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.model.model_t2w import broadcast_condition
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_transfer1.diffusion.module.parallel import broadcast, cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.utils import log, misc


def deepcopy_no_copy_model(obj):
    """
    We need to create a copy of the condition construct such that condition masks can be adjusted dynamically, but
    the controlnet encoder plug-in also uses the condition construct to pass along the base_model object which cannot be
    deep-copied, hence this funciton
    """
    if hasattr(obj, "base_model") and obj.base_model is not None:
        my_base_model = obj.base_model
        obj.base_model = None
        copied_obj = copy.deepcopy(obj)
        copied_obj.base_model = my_base_model
        obj.base_model = my_base_model
    else:
        copied_obj = copy.deepcopy(obj)
    return copied_obj


@dataclass
class VideoDenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction, can be used a confidence / uncertainty
    xt: Optional[torch.Tensor] = None  # input to the network, before muliply with c_in
    x0_pred_replaced: Optional[torch.Tensor] = None  # x0 prediction with condition region replaced by gt_latent


class DiffusionV2WMultiviewModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_views = config.n_views

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        encoded_state = self.tokenizer.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        decoded_state = self.tokenizer.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return decoded_state

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed: int = 1,
    ) -> VideoDenoisePrediction:
        """Denoises input tensor using conditional video generation.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            seed (int): Random seed for reproducibility
        Returns:
            VideoDenoisePrediction containing:
            - x0: Denoised prediction
            - eps: Noise prediction
            - logvar: Log variance of noise prediction
            - xt: Input before c_in multiplication
            - x0_pred_replaced: x0 prediction with condition regions replaced by ground truth
        """

        assert (
            condition.gt_latent is not None
        ), f"find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
        condition = deepcopy_no_copy_model(condition)
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        condition, augment_latent = self.augment_conditional_latent_frames(
            condition, cfg_video_cond_bool, condition_latent, condition_video_augment_sigma_in_inference, sigma, seed
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]

        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition_video_indicator = rearrange(
                condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
            )
            augment_latent = rearrange(augment_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            gt_latent = rearrange(gt_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            if getattr(condition, "view_indices_B_T", None) is not None:
                view_indices_B_V_T = rearrange(condition.view_indices_B_T, "B (V T) -> (B V) T", V=self.n_views)
                view_indices_B_V_T = split_inputs_cp(view_indices_B_V_T, seq_dim=1, cp_group=cp_group)
                condition.view_indices_B_T = rearrange(view_indices_B_V_T, "(B V) T -> B (V T)", V=self.n_views)
            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

            condition_video_indicator = rearrange(
                condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
            )
            augment_latent = rearrange(augment_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
            gt_latent = rearrange(gt_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        # Call the abse model
        denoise_pred = super(DiffusionV2WModel, self).denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0

        x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """Generates video samples conditioned on input frames.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of output tensor (defaults to model's state shape)
            n_sample: Number of samples to generate (defaults to batch size)
            is_negative_prompt: Whether to use negative prompting
            num_steps: Number of denoising steps
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_video_augment_sigma_in_inference: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames
            x_sigma_max: Maximum noise level tensor

        Returns:
            Generated video samples tensor
        """

        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape

        assert condition_latent is not None, "condition_latent should be provided"

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            seed=seed,
        )
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * sigma_max
            )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max)

        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed: int = 1,
    ) -> Callable:
        """Creates denoising function for conditional video generation.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_video_augment_sigma_in_inference: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames
            seed: Random seed for reproducibility

        Returns:
            Function that takes noisy input and noise level and returns denoised prediction
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        if "view_indices" in data_batch:
            comp_factor = self.tokenizer.temporal_compression_factor
            view_indices = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=self.n_views)
            view_indices_B_V_0 = view_indices[:, :, :1]
            view_indices_B_V_1T = view_indices[:, :, 1:-1:comp_factor]
            view_indices_B_V_T = torch.cat([view_indices_B_V_0, view_indices_B_V_1T], dim=-1)
            condition.view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=self.n_views)
            uncondition.view_indices_B_T = condition.view_indices_B_T

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed=seed,
            ).x0_pred_replaced

            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Adds conditioning masks to VideoExtendCondition object.

        Creates binary indicators and input masks for conditional video generation.

        Args:
            latent_state: Input latent tensor (B,C,T,H,W)
            condition: VideoExtendCondition object to update
            num_condition_t: Number of frames to condition on

        Returns:
            Updated VideoExtendCondition with added masks:
            - condition_video_indicator: Binary tensor marking condition regions
            - condition_video_input_mask: Input mask for network
            - gt_latent: Ground truth latent tensor
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region
        condition_video_indicator = rearrange(condition_video_indicator, "B C (V T) H W -> B V C T H W", V=self.n_views)
        if self.config.conditioner.video_cond_bool.condition_location == "first_cam":
            # condition on first cam
            condition_video_indicator[:, 0, :, :, :, :] += 1.0

        elif self.config.conditioner.video_cond_bool.condition_location.startswith("fixed_cam_and_first_n"):
            # condition on a list of cameras specified through the string
            cond_vids = [int(c) for c in self.config.conditioner.video_cond_bool.condition_location.split("_")[5:]]

            for vidx in cond_vids:
                condition_video_indicator[:, vidx, :, :, :, :] += 1.0
            # also condition on first n_condition_t frames
            condition_video_indicator[:, :, :, :num_condition_t] += 1.0
            condition_video_indicator = condition_video_indicator.clamp(max=1.0)

        elif self.config.conditioner.video_cond_bool.condition_location.startswith("fixed_cam"):
            # condition on a list of cameras specified through the string
            cond_vids = [int(c) for c in self.config.conditioner.video_cond_bool.condition_location.split("_")[2:]]

            for vidx in cond_vids:
                condition_video_indicator[:, vidx, :, :, :, :] += 1.0
            condition_video_indicator = torch.clamp(condition_video_indicator, 0, 1)

        elif self.config.conditioner.video_cond_bool.condition_location == "first_cam_and_first_n":
            # condition on first cam
            condition_video_indicator[:, 0, :, :, :, :] += 1.0
            condition_video_indicator[:, :, :, :num_condition_t] += 1.0
            condition_video_indicator = condition_video_indicator.clamp(max=1.0)
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location } not implemented"
            )
        condition_video_indicator = rearrange(
            condition_video_indicator, "B V C T H W  -> B C (V T) H W", V=self.n_views
        )

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        return condition
