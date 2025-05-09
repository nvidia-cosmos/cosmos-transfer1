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

from typing import Callable, Dict, Optional, Tuple, TypeVar, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_transfer1.diffusion.conditioner import VideoConditionerWithCtrl
from cosmos_transfer1.diffusion.inference.inference_utils import merge_patches_into_video, split_video_into_patches
from cosmos_transfer1.diffusion.model.model_t2w import DiffusionT2WModel, broadcast_condition
from cosmos_transfer1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_transfer1.diffusion.module.parallel import broadcast, cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate

T = TypeVar("T")
IS_PREPROCESSED_KEY = "is_preprocessed"


class VideoDiffusionModelWithCtrl(DiffusionV2WModel):
    def build_model(self) -> torch.nn.ModuleDict:
        log.info("Start creating base model")
        base_model = super().build_model()
        # initialize base model
        self.load_base_model(base_model)
        log.info("Done creating base model")

        log.info("Start creating ctrlnet model")
        net = lazy_instantiate(self.config.net_ctrl)
        conditioner = base_model.conditioner
        logvar = base_model.logvar
        # initialize controlnet encoder
        model = torch.nn.ModuleDict({"net": net, "conditioner": conditioner, "logvar": logvar})
        model.load_state_dict(base_model.state_dict(), strict=False)

        model.base_model = base_model
        log.info("Done creating ctrlnet model")

        self.hint_key = self.config.hint_key["hint_key"]
        return model

    @property
    def base_net(self):
        return self.model.base_model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    def load_base_model(self, base_model) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = ""

        if checkpoint_path:
            log.info(f"Loading base model checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading base model checkpoint (local): {checkpoint_path}")

            if "ema" in state_dict:
                # Copy the base model weights from ema model.
                log.info("Copying ema to base model")
                base_state_dict = {k.replace("-", "."): v for k, v in state_dict["ema"].items()}
            elif "model" in state_dict:
                # Copy the base model weights from reg model.
                log.warning("Using non-EMA base model")
                base_state_dict = state_dict["model"]
            else:
                log.info("Loading from an EMA only model")
                base_state_dict = state_dict
            base_model.load_state_dict(base_state_dict, strict=False)
        log.info("Done loading the base model checkpoint.")

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], **kwargs
    ) -> Tuple[Tensor, VideoConditionerWithCtrl]:
        # process the control input
        hint_key = self.config.hint_key["hint_key"]
        _data = {hint_key: data_batch[hint_key]}
        if IS_PREPROCESSED_KEY in data_batch:
            _data[IS_PREPROCESSED_KEY] = data_batch[IS_PREPROCESSED_KEY]
        data_batch[hint_key] = _data[hint_key]
        data_batch["hint_key"] = hint_key
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch, **kwargs)
        use_multicontrol = (
            ("control_weight" in data_batch)
            and not isinstance(data_batch["control_weight"], float)
            and data_batch["control_weight"].shape[0] > 1
        )
        if use_multicontrol:  # encode individual conditions separately
            latent_hint = []
            num_conditions = data_batch[data_batch["hint_key"]].size(1) // 3
            for i in range(num_conditions):
                cond_mask = [False] * num_conditions
                cond_mask[i] = True
                latent_hint += [self.encode_latent(data_batch, cond_mask=cond_mask)]
            latent_hint = torch.cat(latent_hint)
        else:
            latent_hint = self.encode_latent(data_batch)

        # add extra conditions
        data_batch["latent_hint"] = latent_hint
        setattr(condition, hint_key, latent_hint)
        setattr(condition, "base_model", self.model.base_model)
        return raw_state, latent_state, condition

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max

    def encode_latent(self, data_batch: dict, cond_mask: list = []) -> torch.Tensor:
        x = data_batch[data_batch["hint_key"]]
        latent = []
        # control input goes through tokenizer, which always takes 3-input channels
        num_conditions = x.size(1) // 3  # input conditions were concatenated along channel dimension
        if num_conditions > 1 and self.config.hint_dropout_rate > 0:
            if torch.is_grad_enabled():  # during training, randomly dropout some conditions
                cond_mask = torch.rand(num_conditions) > self.config.hint_dropout_rate
                if not cond_mask.any():  # make sure at least one condition is present
                    cond_mask = [True] * num_conditions
            elif not cond_mask:  # during inference, use hint_mask to indicate which conditions are used
                cond_mask = self.config.hint_mask
        else:
            cond_mask = [True] * num_conditions
        for idx in range(0, x.size(1), 3):
            x_rgb = x[:, idx : idx + 3]
            if not cond_mask[idx // 3]:  # if the condition is not selected, replace with a black image
                x_rgb = torch.zeros_like(x_rgb)
            latent.append(self.encode(x_rgb))
        latent = torch.cat(latent, dim=1)
        return latent

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        seed: int = 1,
        target_h: int = 88,
        target_w: int = 160,
        patch_h: int = 88,
        patch_w: int = 160,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
        - target_h (int): final stitched latent height
        - target_w (int): final stitched latent width
        - patch_h (int): latent patch height for each network inference
        - patch_w (int): latent patch width for each network inference

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # data_batch should be the one processed by self.get_data_and_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
            # Add conditions for long video generation.

        if condition_latent is None:
            condition_latent = torch.zeros(data_batch["latent_hint"].shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent[:1], condition, num_condition_t
        )

        uncondition.video_cond_bool = True  # Not do cfg on condition frames
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent[:1], uncondition, num_condition_t
        )

        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        # Add extra conditions for ctrlnet. 
        # Handle regional prompting information
        if "regional_contexts" in data_batch and "region_masks" in data_batch:
            setattr(condition, "regional_contexts", data_batch["regional_contexts"])
            setattr(condition, "region_masks", data_batch["region_masks"])
            # For unconditioned generation, we still need the region masks but not the regional contexts
            setattr(uncondition, "region_masks", data_batch["region_masks"])
            setattr(uncondition, "regional_contexts", None)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

            cp_group = parallel_state.get_context_parallel_group()
            latent_hint = getattr(condition, hint_key)
            seq_dim = 3 if latent_hint.ndim == 6 else 2
            latent_hint = split_inputs_cp(latent_hint, seq_dim=seq_dim, cp_group=cp_group)
            setattr(condition, hint_key, latent_hint)
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint)
            if hasattr(condition, "regional_contexts") and getattr(condition, "regional_contexts") is not None:
                regional_contexts = getattr(condition, "regional_contexts")
                regional_contexts = split_inputs_cp(regional_contexts, seq_dim=2, cp_group=cp_group)
                setattr(condition, "regional_contexts", regional_contexts)

            if hasattr(condition, "region_masks") and getattr(condition, "region_masks") is not None:
                region_masks = getattr(condition, "region_masks")
                region_masks = split_inputs_cp(region_masks, seq_dim=2, cp_group=cp_group)
                setattr(condition, "region_masks", region_masks)
                setattr(uncondition, "region_masks", region_masks)

        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        if hasattr(self, "hint_encoders"):
            self.model.net.hint_encoders = self.hint_encoders

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor):
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

            batch_images = noise_x
            batch_sigma = sigma
            output = []
            for idx, cur_images in enumerate(batch_images):
                noise_x = cur_images.unsqueeze(0)
                sigma = batch_sigma[idx : idx + 1]
                condition.gt_latent = condition_latent[idx : idx + 1]
                uncondition.gt_latent = condition_latent[idx : idx + 1]
                setattr(condition, hint_key, latent_hint[idx : idx + 1])
                if getattr(uncondition, hint_key) is not None:
                    setattr(uncondition, hint_key, latent_hint[idx : idx + 1])

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
                x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
                output.append(x0)
            output = rearrange(torch.stack(output), "(n t) b ... -> (b n t) ...", n=n_img_h, t=n_img_w)
            final_output = merge_patches_into_video(output, overlap_size_h, overlap_size_w, n_img_h, n_img_w)
            final_output = split_video_into_patches(final_output, patch_h, patch_w)
            return final_output

        return x0_fn

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
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        target_h: int = 88,
        target_w: int = 160,
        patch_h: int = 88,
        patch_w: int = 160,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half
        """
        assert patch_h <= target_h and patch_w <= target_w
        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape
        x0_fn = self.get_x0_fn_from_batch(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            seed=seed,
            target_h=target_h,
            target_w=target_w,
            patch_h=patch_h,
            patch_w=patch_w,
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
            x_sigma_max = broadcast(x_sigma_max, to_tp=False, to_cp=True)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max)

        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples


class VideoDiffusionT2VModelWithCtrl(DiffusionT2WModel):
    def build_model(self) -> torch.nn.ModuleDict:
        log.info("Start creating base model")
        base_model = super().build_model()
        # initialize base model
        config = self.config
        self.load_base_model(base_model)
        log.info("Done creating base model")

        log.info("Start creating ctrlnet model")
        net = lazy_instantiate(self.config.net_ctrl)
        conditioner = base_model.conditioner
        logvar = base_model.logvar
        # initialize controlnet encoder
        model = torch.nn.ModuleDict({"net": net, "conditioner": conditioner, "logvar": logvar})
        model.load_state_dict(base_model.state_dict(), strict=False)

        model.base_model = base_model
        log.info("Done creating ctrlnet model")

        self.hint_key = self.config.hint_key["hint_key"]
        return model

    @property
    def base_net(self):
        return self.model.base_model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    def load_base_model(self, base_model) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = ""
        if checkpoint_path:
            log.info(f"Loading base model checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading base model checkpoint (local): {checkpoint_path}")

            if "ema" in state_dict:
                # Copy the base model weights from ema model.
                log.info("Copying ema to base model")
                base_state_dict = {k.replace("-", "."): v for k, v in state_dict["ema"].items()}
            elif "model" in state_dict:
                # Copy the base model weights from reg model.
                log.warning("Using non-EMA base model")
                base_state_dict = state_dict["model"]
            else:
                log.info("Loading from an EMA only model")
                base_state_dict = state_dict
            base_model.load_state_dict(base_state_dict, strict=False)
        log.info("Done loading the base model checkpoint.")

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], **kwargs
    ) -> Tuple[Tensor, VideoConditionerWithCtrl]:
        # process the control input
        hint_key = self.config.hint_key["hint_key"]
        _data = {hint_key: data_batch[hint_key]}
        if IS_PREPROCESSED_KEY in data_batch:
            _data[IS_PREPROCESSED_KEY] = data_batch[IS_PREPROCESSED_KEY]
        data_batch[hint_key] = _data[hint_key]
        data_batch["hint_key"] = hint_key
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch, **kwargs)
        use_multicontrol = (
            ("control_weight" in data_batch)
            and not isinstance(data_batch["control_weight"], float)
            and data_batch["control_weight"].shape[0] > 1
        )
        if use_multicontrol:  # encode individual conditions separately
            latent_hint = []
            num_conditions = data_batch[data_batch["hint_key"]].size(1) // 3
            for i in range(num_conditions):
                cond_mask = [False] * num_conditions
                cond_mask[i] = True
                latent_hint += [self.encode_latent(data_batch, cond_mask=cond_mask)]
            latent_hint = torch.cat(latent_hint)
        else:
            latent_hint = self.encode_latent(data_batch)

        # add extra conditions
        data_batch["latent_hint"] = latent_hint
        setattr(condition, hint_key, latent_hint)
        setattr(condition, "base_model", self.model.base_model)
        return raw_state, latent_state, condition

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max

    def encode_latent(self, data_batch: dict, cond_mask: list = []) -> torch.Tensor:
        x = data_batch[data_batch["hint_key"]]
        latent = []
        # control input goes through tokenizer, which always takes 3-input channels
        num_conditions = x.size(1) // 3  # input conditions were concatenated along channel dimension
        if num_conditions > 1 and self.config.hint_dropout_rate > 0:
            if torch.is_grad_enabled():  # during training, randomly dropout some conditions
                cond_mask = torch.rand(num_conditions) > self.config.hint_dropout_rate
                if not cond_mask.any():  # make sure at least one condition is present
                    cond_mask = [True] * num_conditions
            elif not cond_mask:  # during inference, use hint_mask to indicate which conditions are used
                cond_mask = self.config.hint_mask
        else:
            cond_mask = [True] * num_conditions
        for idx in range(0, x.size(1), 3):
            x_rgb = x[:, idx : idx + 3]
            if not cond_mask[idx // 3]:  # if the condition is not selected, replace with a black image
                x_rgb = torch.zeros_like(x_rgb)
            latent.append(self.encode(x_rgb))
        latent = torch.cat(latent, dim=1)
        return latent

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # data_batch should be the one processed by self.get_data_and_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        # Handle regional prompting information
        if "regional_contexts" in data_batch and "region_masks" in data_batch:
            setattr(condition, "regional_contexts", data_batch["regional_contexts"])
            setattr(condition, "region_masks", data_batch["region_masks"])
            # For unconditioned generation, we still need the region masks but not the regional contexts
            setattr(uncondition, "region_masks", data_batch["region_masks"])
            setattr(uncondition, "regional_contexts", None)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

            cp_group = parallel_state.get_context_parallel_group()
            latent_hint = getattr(condition, hint_key)
            seq_dim = 3 if latent_hint.ndim == 6 else 2
            latent_hint = split_inputs_cp(latent_hint, seq_dim=seq_dim, cp_group=cp_group)
            setattr(condition, hint_key, latent_hint)
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint)

            if hasattr(condition, "regional_contexts") and getattr(condition, "regional_contexts") is not None:
                regional_contexts = getattr(condition, "regional_contexts")
                regional_contexts = split_inputs_cp(regional_contexts, seq_dim=2, cp_group=cp_group)
                setattr(condition, "regional_contexts", regional_contexts)

            if hasattr(condition, "region_masks") and getattr(condition, "region_masks") is not None:
                region_masks = getattr(condition, "region_masks")
                region_masks = split_inputs_cp(region_masks, seq_dim=2, cp_group=cp_group)
                setattr(condition, "region_masks", region_masks)
                setattr(uncondition, "region_masks", region_masks)
        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        if hasattr(self, "hint_encoders"):
            self.model.net.hint_encoders = self.hint_encoders

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
            ).x0
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
            ).x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

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
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half
        """
        if n_sample is None:
            input_key = self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            log.debug(f"Default Video state shape is used. {self.state_shape}")
            state_shape = self.state_shape

        x0_fn = self.get_x0_fn_from_batch(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
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
            x_sigma_max = broadcast(x_sigma_max, to_tp=False, to_cp=True)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max)

        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
        return samples
