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

from abc import ABC
from typing import Any, Dict, Tuple

import torch
from einops import rearrange
from megatron.core import parallel_state

from cosmos_transfer1.diffusion.conditioner import BaseVideoCondition, DataType, VideoExtendCondition
from cosmos_transfer1.diffusion.inference.inference_utils import merge_patches_into_video, split_video_into_patches
from cosmos_transfer1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.diffusion.training.models.model import broadcast_condition
from cosmos_transfer1.diffusion.training.models.model_ctrl import VideoDiffusionModelWithCtrl
from cosmos_transfer1.distillation.models.v2w_model_distill import V2WDistillationMixin
from cosmos_transfer1.utils import log


class CtrlDistillationMixin(V2WDistillationMixin, ABC):
    """ControlNet distillation mixin class."""

    def _get_condition_uncondition(
        self,
        gt_latent: torch.Tensor,
        data_batch: dict[str, torch.Tensor],
        num_condition_t: int | None = None,
    ) -> Any:
        """
        Get the condition (without drop rate) and uncondition from data_batch

        Args:
            gt_latent: gt_latent
            data_batch: The data batch
        """
        if data_batch.get("is_negative_prompt", False):
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        if self.is_image_batch(data_batch):
            condition.data_type = DataType.IMAGE
            uncondition.data_type = DataType.IMAGE
        else:
            condition.video_cond_bool = True  # Not do cfg on condition frames
            condition = self.add_condition_video_indicator_and_video_input_mask(
                gt_latent,
                condition,
                num_condition_t,
            )
            if self.config.conditioner.video_cond_bool.add_pose_condition:
                condition = self.add_condition_pose(data_batch, condition)

            uncondition.video_cond_bool = True  # Not do cfg on condition frames
            uncondition = self.add_condition_video_indicator_and_video_input_mask(
                gt_latent,
                uncondition,
                num_condition_t,
            )
            # make sure condition and uncondition have the same indicator and mask
            uncondition.condition_video_indicator = condition.condition_video_indicator.clone()
            uncondition.condition_video_input_mask = condition.condition_video_input_mask.clone()

            if self.config.conditioner.video_cond_bool.add_pose_condition:
                uncondition = self.add_condition_pose(data_batch, uncondition)

        latent_hint = None if data_batch.get("use_none_hint") else data_batch["latent_hint"]
        setattr(condition, data_batch["hint_key"], latent_hint)
        setattr(uncondition, data_batch["hint_key"], latent_hint)

        # check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=True)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=True)
        else:
            raise RuntimeError("parallel_state is not initialized, context parallel should be turned off.")

        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        return condition, uncondition

    def generate_samples_from_patches(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        condition_latent: torch.Tensor | None = None,
        num_condition_t: int | None = None,
        condition_video_augment_sigma_in_inference: float = None,
        use_teacher: bool = False,
        target_h: int = 2112,
        target_w: int = 3840,
        patch_h: int = 704,
        patch_w: int = 1280,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch using patch-based inference. During each denoising step, it will denoise each
        patch separately then average the overlapping regions.

        Additional args to original function (generate_samples_from_batch):
            target_h (int): final stitched video height
            target_w (int): final stitched video width
            patch_h (int): video patch height for each network inference
            patch_w (int): video patch width for each network inference
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call model_distill generate_samples_from_batch")
            return super(VideoDiffusionModelWithCtrl, self).generate_samples_from_batch(
                data_batch, seed=seed, state_shape=state_shape, n_sample=n_sample, use_teacher=use_teacher
            )

        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        random_noise = torch.randn(n_sample, *state_shape, generator=generator, **self.tensor_kwargs)
        dummy_tensor = torch.randn(n_sample, *state_shape, generator=generator, **self.tensor_kwargs)

        if condition_latent is None:
            condition_latent = torch.zeros(state_shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        condition, uncondition = self._get_condition_uncondition(condition_latent, data_batch, num_condition_t)
        _, _, condition, uncondition, _ = self.setup_context_parallel(
            data_batch, dummy_tensor, condition, uncondition, dummy_tensor
        )

        cp_enabled = self.net.is_context_parallel_enabled
        if cp_enabled:
            random_noise = split_inputs_cp(x=random_noise, seq_dim=2, cp_group=self.net.cp_group)

        samples = self._forward_patch(
            epsilon=random_noise,
            condition=condition,
            uncondition=uncondition,
            hint_key=data_batch["hint_key"],
            use_teacher=use_teacher,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            inference_mode=True,
            target_h=target_h,
            target_w=target_w,
            patch_h=patch_h,
            patch_w=patch_w,
        )

        if cp_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def _forward_patch(
        self,
        epsilon: torch.Tensor,
        condition: VideoExtendCondition,
        uncondition: VideoExtendCondition,
        hint_key: str,
        use_teacher: bool = False,
        condition_video_augment_sigma_in_inference: float = 0.001,
        inference_mode: bool = False,
        target_h: int = 2112,
        target_w: int = 3840,
        patch_h: int = 704,
        patch_w: int = 1280,
    ) -> torch.Tensor:
        """
        Patch-wise denoising process.

        Data batch inputs are split into non-overlapping, contiguous patches to generate partial video frames
        """
        condition_latent = condition.gt_latent
        latent_hint = getattr(condition, hint_key)

        w, h = target_w, target_h
        n_img_w = (w - 1) // patch_w + 1
        n_img_h = (h - 1) // patch_h + 1

        overlap_size_w = overlap_size_h = 0
        if n_img_w > 1:
            overlap_size_w = (n_img_w * patch_w - w) // (n_img_w - 1)
            assert n_img_w * patch_w - overlap_size_w * (n_img_w - 1) == w
        if n_img_h > 1:
            overlap_size_h = (n_img_h * patch_h - h) // (n_img_h - 1)
            assert n_img_h * patch_h - overlap_size_h * (n_img_h - 1) == h

        output = []
        for idx, current_images in enumerate(epsilon):
            condition.gt_latent = condition_latent[idx : idx + 1]
            uncondition.gt_latent = condition_latent[idx : idx + 1]
            setattr(condition, hint_key, latent_hint[idx : idx + 1])
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint[idx : idx + 1])

            xT = current_images.unsqueeze(0) * self.sde.sigma_max
            sigma_max = torch.tensor(self.sde.sigma_max).repeat(xT.size(0))
            net = self.teacher if use_teacher else self.net
            video_pred = self.denoise_net(
                net=net,
                xt=xT,
                sigma=sigma_max,
                condition=condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            )
            if inference_mode:
                x0 = video_pred.x0_pred_replaced
            else:
                x0 = video_pred.x0
            output.append(x0)
        output = rearrange(torch.stack(output), "(n t) b ... -> (b n t) ...", n=n_img_h, t=n_img_w)  # 8x3xhxw
        final_output = merge_patches_into_video(output, overlap_size_h, overlap_size_w, n_img_h, n_img_w)
        final_output = split_video_into_patches(final_output, patch_h, patch_w)
        return final_output

    def setup_context_parallel(
        self,
        data_batch: dict[str, torch.Tensor],
        x0: torch.Tensor,
        condition: BaseVideoCondition,
        uncondition: BaseVideoCondition,
        epsilon: torch.Tensor,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor, BaseVideoCondition, BaseVideoCondition, torch.Tensor]:
        """Toggle context parallelism, splitting any and all tensors for which CP is relevant. Note, tensor broadcasting
        is performed at initialisation time, not here.

        This overload handles CP for ControlNets, that is, toggles CP in the base Diffusion network (`base_model.net`)
        and splits the latent-encoded hints contained in the condition (`condition.latent_hint`).

        Args:
            data_batch (dict[str, torch.Tensor]): The current training iteration.
            x0 (torch.Tensor): inputs in latent space
            condition (BaseVideoCondition): configuration for conditional generation. Note, tensors relevant for CP are
                                            held through reference in the `uncondition` object, hence only `condition`
                                            needs to be handled.
            unconditional (BaseVideoCondition): configuration for unconditional generation for DMD2.
            epsilon (torch.Tensor): initial noise state

        Returns:
            Tuple(data_batch, x0, condition, epsilon): returns input tensors possibly split over rank if CP is enabled.
        """
        if data_batch.get("context_parallel_setup", False):
            # Context Parallel already handled for current data batch
            return data_batch, x0, condition, uncondition, epsilon
        data_batch, x0, condition, uncondition, epsilon = super().setup_context_parallel(
            data_batch, x0, condition, uncondition, epsilon
        )

        # Extra handling for V2W ControlNet
        if self.is_image_batch(data_batch):
            # Turn off CP
            self.base_model.net.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.base_model.net.enable_context_parallel(cp_group)
                    log.debug("[CP] Split hint")
                    latent_hint = getattr(condition, data_batch["hint_key"])
                    latent_hint = split_inputs_cp(x=latent_hint, seq_dim=2, cp_group=cp_group)
                    setattr(condition, data_batch["hint_key"], latent_hint)

                    latent_hint_uncond = getattr(uncondition, data_batch["hint_key"])
                    if latent_hint_uncond.shape[2] > latent_hint.shape[2]:
                        latent_hint_uncond = split_inputs_cp(x=latent_hint_uncond, seq_dim=2, cp_group=cp_group)
                        setattr(uncondition, data_batch["hint_key"], latent_hint_uncond)

        data_batch["context_parallel_setup"] = True
        return data_batch, x0, condition, uncondition, epsilon


class CtrlDistillationModel(CtrlDistillationMixin, VideoDiffusionModelWithCtrl, ABC):
    pass
