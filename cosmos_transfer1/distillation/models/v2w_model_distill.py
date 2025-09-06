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
from typing import Dict, List, Optional, Set, Tuple

import torch
from megatron.core import parallel_state

from cosmos_transfer1.diffusion.conditioner import DataType, VideoExtendCondition
from cosmos_transfer1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_transfer1.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.diffusion.training.models.extend_model import ExtendDiffusionModel, VideoDenoisePrediction
from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationMixin
from cosmos_transfer1.utils import log


class V2WDistillationMixin(BaseDistillationMixin, ABC):
    """Video2World distillation mixin class."""

    def denoise_net(
        self,
        net: torch.nn.Module,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        feature_indices: Set[int] = None,
        return_features_early: bool = False,
    ) -> VideoDenoisePrediction | List[torch.Tensor] | Tuple[VideoDenoisePrediction, List[torch.Tensor]]:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            net (torch.nn.Module): The network to use for denoising
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (VideoExtendCondition): conditional information, generated from self.conditioner
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            feature_indices (Set[int]): indices of features to extract from the net
            return_features_early (bool): Whether to return features early with returning denoising output

        Returns:
            VideoDenoisePrediction: The denoised prediction for video, it includes clean data prediction (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
            or List[torch.Tensor]: The extracted features
            or Tuple[VideoDenoisePrediction, List[torch.Tensor]]: Both denoised prediction and extracted features.
        """

        sigma = sigma.to(**self.tensor_kwargs)
        if condition.data_type == DataType.IMAGE:
            denoise_out = super().denoise_net(net, xt, sigma, condition, feature_indices, return_features_early)
            log.debug(f"hit image denoise, noise_x shape {xt.shape}, sigma shape {sigma.shape}", rank0_only=False)
        else:
            assert condition.gt_latent is not None, (
                f"find None gt_latent in condition, "
                f"likely didn't call self.add_condition_video_indicator_and_video_input_mask "
                f"when preparing the condition or "
                f"this is an image batch but condition.data_type is wrong, get {xt.shape}"
            )
            gt_latent = condition.gt_latent
            cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

            # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
            condition, augment_latent = self.augment_conditional_latent_frames(
                condition, cfg_video_cond_bool, gt_latent, condition_video_augment_sigma_in_inference, sigma
            )
            condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
                augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
                gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

            if not condition.video_cond_bool:
                # Unconditional case, drop out the condition region
                augment_latent = self.drop_out_condition_region(augment_latent, xt, cfg_video_cond_bool)

            # Compose the model input with condition region (augment_latent) and generation region (noise_x)
            new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * xt
            # Call the base model
            denoise_out = super().denoise_net(
                net, new_noise_xt, sigma, condition, feature_indices, return_features_early
            )
        if return_features_early:
            return denoise_out  # extracted features

        if feature_indices is None or len(feature_indices) == 0:
            denoise_pred = denoise_out  # denoised prediction
        else:
            denoise_pred, features = denoise_out  # denoised prediction, extracted features

        if condition.data_type == DataType.IMAGE:
            video_denoise_pred = VideoDenoisePrediction(
                x0=denoise_pred.x0,
                eps=denoise_pred.eps,
                logvar=denoise_pred.logvar,
                xt=xt,
            )
        else:
            x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0
            if cfg_video_cond_bool.compute_loss_for_condition_region:
                # We also denoise the conditional region
                x0_pred = denoise_pred.x0
            else:
                x0_pred = x0_pred_replaced

            video_denoise_pred = VideoDenoisePrediction(
                x0=x0_pred,
                eps=batch_mul(xt - x0_pred, 1.0 / sigma),
                logvar=denoise_pred.logvar,
                net_in=batch_mul(1.0 / torch.sqrt(self.sigma_data**2 + sigma**2), new_noise_xt),
                net_x0_pred=denoise_pred.x0,
                xt=new_noise_xt,
                x0_pred_replaced=x0_pred_replaced,
            )

        if feature_indices is None or len(feature_indices) == 0:
            return video_denoise_pred
        else:
            return video_denoise_pred, features

    def _forward(
        self,
        epsilon: torch.Tensor,
        condition: VideoExtendCondition,
        use_teacher: bool = False,
        condition_video_augment_sigma_in_inference: float = 0.001,
        inference_mode: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the forward pass (from the maximum noise scale for one-step generation).

        Args:
            epsilon (torch.Tensor): Random noise with zero mean and unit variance.
            condition (VideoExtendCondition): conditional information, generated from self.conditioner
            use_teacher (bool, optional): Whether to use teacher or net. Defaults to False.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            inference_mode (bool, optional): Whether to use inference mode or not. Defaults to False.

        Returns:
            torch.Tensor: The output of the model.
        """
        sigma_max = kwargs.get("sigma_max", self.sde.sigma_max)
        xT = epsilon * sigma_max
        sigma_max = torch.tensor(sigma_max).repeat(xT.size(0))
        net = self.net
        if use_teacher:
            if hasattr(self, "teacher"):
                net = self.teacher
            else:
                log.warning("self.teacher does not exist while use_teacher=True. Use student net instead")
        video_pred = self.denoise_net(
            net=net,
            xt=xT,
            sigma=sigma_max,
            condition=condition,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
        )
        if inference_mode:
            return video_pred.x0_pred_replaced
        return video_pred.x0

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        condition_latent: torch.Tensor | None = None,
        num_condition_t: int | None = None,
        condition_video_augment_sigma_in_inference: float = None,
        use_teacher: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            seed (int): random seed
            state_shape (tuple): shape of the state, default to self.state_shape if not provided
            n_sample (int): number of samples to generate
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            use_teacher (bool, optional): Whether to use teacher or net. Defaults to False.

            use_teacher (bool, optional): Whether to use teacher or net. Defaults to False.
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call model_distill generate_samples_from_batch")
            return super().generate_samples_from_batch(
                data_batch, seed=seed, state_shape=state_shape, n_sample=n_sample, use_teacher=use_teacher
            )

        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape

        batch_shape = (n_sample, *state_shape)
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        random_noise = torch.randn(*batch_shape, generator=generator, **self.tensor_kwargs)

        if condition_latent is None:
            condition_latent = torch.zeros(batch_shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        # Add conditions for long video generation.
        _, dummy_tensor, condition = self.get_data_and_condition(data_batch, num_condition_t=num_condition_t)
        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        # Toggle CP accordingly
        data_batch["context_parallel_setup"] = False  # Force CP handling and splitting
        data_batch["validation_mode"] = True

        # We use condition in place of the uncondition argument here
        _, _, condition, _, _ = self.setup_context_parallel(
            data_batch, dummy_tensor, condition, condition, dummy_tensor
        )

        cp_enabled = self.net.is_context_parallel_enabled
        if cp_enabled:
            random_noise = split_inputs_cp(x=random_noise, seq_dim=2, cp_group=self.net.cp_group)

        samples = self._forward(
            epsilon=random_noise,
            condition=condition,
            use_teacher=use_teacher,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            inference_mode=True,
            **kwargs,
        )

        if cp_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Optional[int] = None
    ) -> VideoExtendCondition:
        """Duplicate of `extend_model::add_condition_video_indicator_and_video_input_mask` that omits the repeated call
        to broadcast the condition.

        Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        `condition_video_indicator` is a binary tensor indicating the condition region in the latent state (
        [1, 1, T, 1, 1] tensor). `condition_video_input_mask` will be concatenated with the input for the network.

        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and
                                   config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region
        if self.config.conditioner.video_cond_bool.condition_location == "first_n":
            # Only in inference to decide the condition region
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_random_n":
            # Only in training
            num_condition_t_max = self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_max
            assert (
                num_condition_t_max <= T
            ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            assert num_condition_t_max >= self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min
            num_condition_t = torch.randint(
                self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min,
                num_condition_t_max + 1,
                (1,),
            ).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        elif self.config.conditioner.video_cond_bool.condition_location == "random":
            # Only in training
            condition_rate = self.config.conditioner.video_cond_bool.random_conditon_rate
            flag = torch.ones(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype) * condition_rate
            condition_video_indicator = torch.bernoulli(flag).type(latent_dtype).to(latent_state.device)
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
            )
        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
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


class V2WDistillationModel(BaseDistillationMixin, ExtendDiffusionModel, ABC):
    pass
