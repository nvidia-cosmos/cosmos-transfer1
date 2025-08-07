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

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from einops import repeat
from megatron.core import parallel_state

from cosmos_transfer1.diffusion.conditioner import BaseVideoCondition, DataType
from cosmos_transfer1.diffusion.diffusion.types import DenoisePrediction
from cosmos_transfer1.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer1.diffusion.training.models.model import DiffusionModel, _broadcast, broadcast_condition
from cosmos_transfer1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_transfer1.diffusion.training.utils.optim_instantiate import get_base_scheduler
from cosmos_transfer1.distillation.networks.distill_controlnet_wrapper import DistillControlNet
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.easy_io import easy_io
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate


class BaseDistillationMixin(DiffusionModel, ABC):
    """Base distillation mixin class."""

    def build_model(self) -> torch.nn.ModuleDict:
        """Expand model building to initialise Teacher module and disable gradients on it."""
        config = self.config

        log.info("Instantiating the net")
        model_dict = super().build_model()

        log.info("Instantiating the teacher")
        self.teacher = lazy_instantiate(getattr(config, config.teacher_net_name))

        if config.is_ctrl_net:
            # Reload the base model for the control net setting.
            # The previous loading function in model_ctrl.py for base model is incorrect for our current checkpoint.
            self._load_pretrained_net(self.teacher, model_dict)
        else:
            self._load_pretrained_net(self.teacher)

        self.teacher.requires_grad_(False).eval()

        if config.is_ctrl_net:
            # Re-initialize the net under the controlnet setting
            log.info("Re-instantiating the student net under the controlnet setting")
            net = DistillControlNet(config)
            # Load the weight for the encoder
            net.net_ctrl.load_state_dict(self.teacher.state_dict())
            # Load the weight for the VideoExtendGeneralDIT
            net.base_model.net.load_state_dict(model_dict["base_model"].net.state_dict())
            model_dict["net"] = net
        else:
            model_dict["net"].load_state_dict(self.teacher.state_dict())

        model_dict["net"].requires_grad_(True).train()

        return model_dict

    @property
    def logvar(self):
        return self.model.logvar

    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict,
        scheduler_config: LazyDict,
    ) -> None:
        """Initialise training utilities for the Student module."""
        # instantiate the net optimizer
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        # instantiate the net scheduler
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

    def _load_pretrained_net(self, net: GeneralDIT, model_dict: torch.nn.ModuleDict = None) -> None:
        """
        Loading pre-trained base model to net.

        Args:
            net: The DiT net in pre-trained diffusion model
            model_dict: The model dictionary containing the base model in control net setting
        """
        log.info("start loading pretrained base model to net.")
        if self.config.base_load_from is None:
            log.warning("base_load_from is not set, skipping loading pretrained base model to net.")
            return
        checkpoint_path = self.config.base_load_from["load_path"]
        load_from_tp_checkpoint = False

        if "*" in checkpoint_path:
            # there might be better ways to decide if it's a converted tp checkpoint
            mp_rank = parallel_state.get_model_parallel_group().rank()
            checkpoint_path = checkpoint_path.replace("*", f"{mp_rank}")
            load_from_tp_checkpoint = True

        if checkpoint_path:
            log.info(f"Loading base model checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading base model checkpoint (local): {checkpoint_path}")

            new_state_dict_model = {}
            new_state_dict_base_model = {}
            if load_from_tp_checkpoint:
                state_dict = state_dict["ema"]
            for key, value in state_dict.items():
                if load_from_tp_checkpoint:
                    key = key.replace("-", ".")
                key_parts = key.split(".")
                if key_parts[0] == "net":
                    new_key = ".".join(key_parts[1:])
                    new_state_dict_model[new_key] = value
                elif key_parts[0] == "base_model" and model_dict is not None:
                    new_key = ".".join(key_parts[1:])
                    new_state_dict_base_model[new_key] = value

            # Using strict=False since network contains _extra_state keys not present in tp checkpoint
            if model_dict is not None:
                load_info = model_dict["base_model"].load_state_dict(new_state_dict_base_model, strict=False)
                log.info(f"Load info for base model: {load_info}")
            load_info = net.load_state_dict(new_state_dict_model, strict=False)
            log.info(f"Load info for net: {load_info}")
            torch.cuda.empty_cache()
        log.info("Done loading the base model checkpoint.")

    def _sampling_noise_schedule(self, n, num_steps=1000, min_step_percent=0.02, max_step_percent=0.98):
        noise_schedule_type = getattr(self.config, "noise_schedule_type", "inference")
        if noise_schedule_type == "inference":
            sigmas = np.exp(np.linspace(math.log(self.sde.sigma_min), math.log(self.sde.sigma_max), num_steps + 1)[1:])
            min_step = int(num_steps * min_step_percent)
            max_step = int(num_steps * max_step_percent)
            indices = np.random.randint(min_step, max_step + 1, (n,))
            return torch.tensor(sigmas[indices])

        elif noise_schedule_type == "edm_sampling":
            rho, min_t, max_t = 7.0, 0.002, 80.0
            ramp = torch.linspace(0, 1, num_steps)
            min_inv_rho = min_t ** (1 / rho)
            max_inv_rho = max_t ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

            min_step = int(num_steps * min_step_percent)
            max_step = int(num_steps * max_step_percent)
            indices = torch.randint(min_step, max_step + 1, (n,))

            return sigmas[indices]
        else:
            raise NotImplementedError(f"Noise schedule type {noise_schedule_type} not implemented")

    def draw_training_sigma_and_epsilon(
        self, size: torch.Size, condition: BaseVideoCondition
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = size[0]
        epsilon = torch.randn(size, **self.tensor_kwargs)

        sigma_B = self._sampling_noise_schedule(batch_size).to(**self.tensor_kwargs)

        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B = _broadcast(sigma_B * multiplier, to_tp=True, to_cp=is_video_batch)
        epsilon = _broadcast(epsilon, to_tp=True, to_cp=is_video_batch)
        return sigma_B, epsilon

    def _add_negative_prompt(self, data_batch, x0):
        if getattr(self.config, "use_negative_prompt", False):
            assert self.negative_prompt_data is not None
            batch_size = x0.shape[0]
            data_batch["neg_t5_text_embeddings"] = misc.to(
                repeat(
                    self.negative_prompt_data["t5_text_embeddings"],
                    "... -> b ...",
                    b=batch_size,
                ),
                **self.tensor_kwargs,
            )
            assert (
                data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape
            ), f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]
        return data_batch

    def training_step(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the distillation model.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - torch.Tensor: The computed loss for the training step as a PyTorch Tensor.
        """
        # Update trained_data_record
        input_key = self.input_data_key  # by default it is video key
        if self.is_image_batch(data_batch):
            input_key = self.input_image_key

        batch_size = data_batch[input_key].shape[0]
        self.trained_data_record["image" if self.is_image_batch(data_batch) else "video"] += (
            batch_size * self.data_parallel_size
        )
        self.trained_data_record["iteration"] += 1

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        x0_from_data_batch, x0, _ = self.get_data_and_condition(data_batch)
        if getattr(self.config, "use_negative_prompt", False):
            data_batch = self._add_negative_prompt(data_batch, x0)
            if "neg_t5_text_embeddings" in data_batch:
                data_batch["is_negative_prompt"] = True
            else:
                raise ValueError("config.use_negative_prompt=True but couldn't find neg_t5_text_embeddings in data")
        condition, uncondition = self._get_condition_uncondition(x0, data_batch)

        # Sample perturbation noise levels and N(0, 1) noises
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition)

        output_batch, loss = self.compute_loss_with_epsilon_and_sigma(
            data_batch=data_batch,
            x0_from_data_batch=x0_from_data_batch,
            x0=x0,
            condition=condition,
            uncondition=uncondition,
            epsilon=epsilon,
            sigma=sigma,
            iteration=iteration,
        )

        loss = loss * self.loss_scale

        return output_batch, loss

    def _get_condition_uncondition(self, gt_latent: torch.Tensor, data_batch: dict[str, torch.Tensor]) -> Any:
        """
        Get the condition and uncondition from data_batch

        Args:
            gt_latent: gt_latent or None
            data_batch: The data batch
        """
        del gt_latent
        if data_batch.get("is_negative_prompt", False):
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=True)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=True)
        else:
            raise RuntimeError("parallel_state is not initialized, context parallel should be turned off.")

        return condition, uncondition

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration

        Args:
            iteration (int): The current training iteration
        """
        return [self.optimizer_dict["net"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration

        Args:
            iteration (int): The current training iteration
        """
        return [self.scheduler_dict["net"]]

    def optimizers_zero_grad(self, iteration: int) -> None:
        """
        Zero the gradients of the optimizers based on the iteration
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        """
        Step the optimizer and scheduler step based on the iteration,
        and gradient scaler is also updated
        """
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
            grad_scaler.update()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    @abstractmethod
    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: BaseVideoCondition,
        uncondition: BaseVideoCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        iteration: int = 0,
    ):
        """
        Compute the diffusion distillation loss

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0_from_data_batch: raw image/video
            x0: image/video latent
            condition: text condition
            uncondition: configuration for unconditional generation
            epsilon: noise
            sigma: noise level
            iteration: training iteration (controls what networks should be updated alternatively)

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: final loss in current iteration
        """
        pass

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

        This overload handles CP for the Student and Teacher modules and splits `x0` and `epsilon`.

        Args:
            data_batch (dict[str, torch.Tensor]): The current training iteration.
            x0 (torch.Tensor): inputs in latent space
            condition (BaseVideoCondition): configuration for conditional generation.
            uncondition (BaseVideoCondition): configuration for unconditional generation.
            epsilon (torch.Tensor): initial noise state

        Returns:
            Tuple(data_batch, x0, condition, epsilon): returns input tensors possibly split over rank if CP is enabled.
        """
        if data_batch.get("context_parallel_setup", False):
            # Context Parallel already handled for current data batch
            return data_batch, x0, condition, uncondition, epsilon

        if self.is_image_batch(data_batch):
            # Turn off CP
            self.net.disable_context_parallel()
            if hasattr(self, "teacher"):
                self.teacher.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.net.enable_context_parallel(cp_group)
                    if hasattr(self, "teacher"):
                        self.teacher.enable_context_parallel(cp_group)
                    log.debug("[CP] Split x0 and epsilon")
                    x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=cp_group)
                    epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=cp_group)

        data_batch["context_parallel_setup"] = True
        return data_batch, x0, condition, uncondition, epsilon

    def denoise_net(
        self,
        net: torch.nn.Module,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        condition: BaseVideoCondition,
        feature_indices: Set[int] = None,
        return_features_early: bool = False,
    ) -> DenoisePrediction | List[torch.Tensor] | Tuple[DenoisePrediction, List[torch.Tensor]]:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            net (torch.nn.Module): The network to use for denoising
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (BaseVideoCondition): conditional information, generated from self.conditioner
            feature_indices (Set[int]): indices of features to extract from the net
            return_features_early (bool): Whether to return features early with returning denoising output

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data prediction (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
            or List[torch.Tensor]: The extracted features
            or Tuple[DenoisePrediction, List[torch.Tensor]]: Both denoised prediction and extracted features.
        """

        if getattr(self.config, "use_dummy_temporal_dim", False):
            # When using video DiT model for image, we need to use a dummy temporal dimension.
            xt = xt.unsqueeze(2)

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        # forward pass through the network
        net_output_ = net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            feature_indices=feature_indices,
            return_features_early=return_features_early,
            **condition.to_dict(),
        )
        if return_features_early:
            return net_output_  # extracted features

        if feature_indices is None or len(feature_indices) == 0:
            net_output = net_output_  # denoised prediction
        else:
            net_output = net_output_[0]  # denoised prediction, extracted features

        logvar = self.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        if getattr(self.config, "use_dummy_temporal_dim", False):
            x0_pred = x0_pred.squeeze(2)
            eps_pred = eps_pred.squeeze(2)

        if feature_indices is None or len(feature_indices) == 0:
            return DenoisePrediction(x0_pred, eps_pred, logvar)
        else:
            return DenoisePrediction(x0_pred, eps_pred, logvar), net_output_[1]

    def _forward(
        self, epsilon: torch.Tensor, condition: BaseVideoCondition, use_teacher: bool = False, **kwargs
    ) -> torch.Tensor:
        """Compute the forward pass (from the maximum noise scale for one-step generation).

        Args:
            epsilon (torch.Tensor): Random noise with zero mean and unit variance.
            condition (BaseVideoCondition): conditional information, generated from self.conditioner
            use_teacher (bool, optional): Whether to use teacher or net. Defaults to False.

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
        return self.denoise_net(net=net, xt=xT, sigma=sigma_max, condition=condition).x0

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """
        Chained initialisation of sequence parallelism on relevant modules up the class hierarchy. Extended to include
        Teacher module.
        """
        super().on_train_start(memory_format)
        if hasattr(self, "teacher"):
            self.teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if sequence_parallel:
                if hasattr(self, "teacher"):
                    self.teacher.enable_sequence_parallel()

        if getattr(self.config, "use_negative_prompt", False):
            assert self.config.negative_prompt_path, "negative_prompt_path is not set but use_negative_prompt is True"
            log.info(f"Loading negative prompt from {self.config.negative_prompt_path}")
            self.negative_prompt_data = easy_io.load(self.config.negative_prompt_path)

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        use_teacher: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image
        or video samples.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            seed (int): random seed
            state_shape (tuple): shape of the state, default to self.state_shape if not provided
            n_sample (int): number of samples to generate
            use_teacher (bool, optional): Whether to use teacher or net. Defaults to False.
        """
        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                state_shape = self.state_shape

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        random_noise = torch.randn(n_sample, *state_shape, generator=generator, **self.tensor_kwargs)

        _, _, condition = self.get_data_and_condition(data_batch)
        if self.net.is_context_parallel_enabled:
            random_noise = split_inputs_cp(x=random_noise, seq_dim=2, cp_group=self.net.cp_group)

        samples = self._forward(epsilon=random_noise, condition=condition, use_teacher=use_teacher, **kwargs)

        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples


class BaseDistillationModel(BaseDistillationMixin, DiffusionModel, ABC):
    pass
