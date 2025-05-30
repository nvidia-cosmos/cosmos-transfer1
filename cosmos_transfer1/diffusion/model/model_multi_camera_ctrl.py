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

from cosmos_transfer1.diffusion.model.model_t2w import broadcast_condition
from cosmos_transfer1.diffusion.model.model_v2w_multiview import DiffusionV2WMultiviewModel
from cosmos_transfer1.diffusion.module.parallel import broadcast, cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate

T = TypeVar("T")
IS_PREPROCESSED_KEY = "is_preprocessed"


class MultiVideoDiffusionModelWithCtrl(DiffusionV2WMultiviewModel):
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

    def encode_latent(self, data_batch: dict, cond_mask: list = []) -> torch.Tensor:
        x = data_batch[data_batch["hint_key"]]
        latent = []
        # control input goes through tokenizer, which always takes 3-input channels
        num_conditions = x.size(1) // 3  # input conditions were concatenated along channel dimension
        if num_conditions > 1 and self.config.hint_dropout_rate > 0:
            if not cond_mask:  # during inference, use hint_mask to indicate which conditions are used
                cond_mask = self.config.hint_mask
        else:
            cond_mask = [True] * num_conditions
        for idx in range(0, x.size(1), 3):
            x_rgb = x[:, idx : idx + 3]  # B C (V T) H W
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

        if "view_indices" in data_batch:
            comp_factor = self.vae.temporal_compression_factor
            # n_frames = data_batch['num_frames']
            view_indices = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=self.n_views)
            view_indices_B_V_0 = view_indices[:, :, :1]
            view_indices_B_V_1T = view_indices[:, :, 1:-1:comp_factor]
            view_indices_B_V_T = torch.cat([view_indices_B_V_0, view_indices_B_V_1T], dim=-1)
            condition.view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=self.n_views)
            condition.data_n_views = self.n_views
            uncondition.view_indices_B_T = condition.view_indices_B_T
            uncondition.data_n_views = self.n_views

        if condition_latent is None:
            batch_size = data_batch["latent_hint"].shape[0]
            condition_latent = torch.zeros(batch_size, *self.state_shape, **self.tensor_kwargs)
            num_condition_t = 0
            condition_video_augment_sigma_in_inference = 1000

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )

        uncondition.video_cond_bool = True  # Not do cfg on condition frames
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )

        # Add extra conditions for ctrlnet.
        latent_hint = data_batch["latent_hint"]
        hint_key = data_batch["hint_key"]
        setattr(condition, hint_key, latent_hint)
        if "use_none_hint" in data_batch and data_batch["use_none_hint"]:
            setattr(uncondition, hint_key, None)
        else:
            setattr(uncondition, hint_key, latent_hint)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():  # and not self.is_image_batch(data_batch):
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

            cp_group = parallel_state.get_context_parallel_group()
            latent_hint = getattr(condition, hint_key)
            seq_dim = 3 if latent_hint.ndim == 6 else 2
            latent_hint = split_inputs_cp(latent_hint, seq_dim=seq_dim, cp_group=cp_group)
            setattr(condition, hint_key, latent_hint)
            if getattr(uncondition, hint_key) is not None:
                setattr(uncondition, hint_key, latent_hint)

        setattr(condition, "base_model", self.model.base_model)
        setattr(uncondition, "base_model", self.model.base_model)
        if hasattr(self, "hint_encoders"):
            self.model.net.hint_encoders = self.hint_encoders

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

        is_image_batch = False  # self.is_image_batch(data_batch)

        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
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
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=sigma_max)

        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views).contiguous()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples
