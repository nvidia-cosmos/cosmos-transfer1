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
from typing import Tuple

import torch
from megatron.core import parallel_state

from cosmos_transfer1.diffusion.conditioner import BaseVideoCondition
from cosmos_transfer1.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.module.parallel import cat_outputs_cp, cat_outputs_cp_with_grad
from cosmos_transfer1.diffusion.training.models.model import DiffusionModel
from cosmos_transfer1.diffusion.training.utils.optim_instantiate import get_base_scheduler
from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel
from cosmos_transfer1.distillation.models.ctrl_model_distill import CtrlDistillationModel
from cosmos_transfer1.distillation.models.v2w_model_distill import V2WDistillationModel
from cosmos_transfer1.distillation.networks.distill_controlnet_wrapper import DistillControlNet
from cosmos_transfer1.distillation.utils.losses import (
    denoising_score_matching_loss,
    gan_loss_discriminator,
    gan_loss_generator,
    variational_score_distillation_loss,
)
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate


class DMD2ModelMixin(DiffusionModel, ABC):
    """DMD2 mixin class for diffusion step distillation."""

    def build_model(self) -> torch.nn.ModuleDict:
        """Expand model building to initialise fake_score and discriminator modules"""
        model_dict = super(DMD2ModelMixin, self).build_model()
        config = self.config

        if not config.ladd:
            log.info("Instantiating the fake score")
            if config.is_ctrl_net:
                self.fake_score = DistillControlNet(config)
                self.fake_score.net_ctrl.load_state_dict(self.teacher.state_dict())
                self.fake_score.base_model.net.load_state_dict(model_dict["base_model"].net.state_dict())
            else:
                self.fake_score = lazy_instantiate(getattr(config, config.fake_score_net_name))
                self.fake_score.load_state_dict(self.teacher.state_dict())

            model_dict["fake_score"] = self.fake_score

        if self.config.gan_loss_weight_gen > 0:
            log.info("Instantiating the discriminator")
            self.discriminator = lazy_instantiate(config.discriminator)
            model_dict["discriminator"] = self.discriminator

        return model_dict

    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict,
        scheduler_config: LazyDict,
    ) -> None:
        """Expand initialisation to cover fake_score and discriminator (if GAN loss enabled) modules."""
        super(DMD2ModelMixin, self).init_optimizer_scheduler(optimizer_config, scheduler_config)

        if not self.config.ladd:
            # instantiate the optimizer and lr scheduler for fake_score
            fake_score_optimizer = lazy_instantiate(self.config.fake_score_optimizer, model=self.fake_score)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, self.config.fake_score_scheduler)
            self.optimizer_dict["fake_score"] = fake_score_optimizer
            self.scheduler_dict["fake_score"] = fake_score_scheduler

        if self.config.gan_loss_weight_gen > 0 or self.config.ladd:
            # instantiate the optimizer and lr scheduler for discriminator
            discriminator_optimizer = lazy_instantiate(self.config.discriminator_optimizer, model=self.discriminator)
            discriminator_scheduler = get_base_scheduler(
                discriminator_optimizer, self, self.config.discriminator_scheduler
            )
            self.optimizer_dict["discriminator"] = discriminator_optimizer
            self.scheduler_dict["discriminator"] = discriminator_scheduler

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration

        Args:
            iteration (int): The current training iteration
        """
        if iteration % self.config.student_update_freq == 0:
            return [self.optimizer_dict["net"]]
        else:
            if self.config.gan_loss_weight_gen > 0 and not self.config.ladd:
                return [self.optimizer_dict["fake_score"], self.optimizer_dict["discriminator"]]
            elif self.config.ladd:
                return [self.optimizer_dict["discriminator"]]
            else:
                return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration

        Args:
            iteration (int): The current training iteration
        """
        if iteration % self.config.student_update_freq == 0:
            return [self.scheduler_dict["net"]]
        else:
            if self.config.gan_loss_weight_gen > 0 and not self.config.ladd:
                return [self.scheduler_dict["fake_score"], self.scheduler_dict["discriminator"]]
            elif self.config.ladd:
                return [self.scheduler_dict["discriminator"]]
            else:
                return [self.scheduler_dict["fake_score"]]

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

        This overload handles CP for the FakeScore module and skips Discriminator.

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

        data_batch, x0, condition, uncondition, epsilon = super().setup_context_parallel(
            data_batch, x0, condition, uncondition, epsilon
        )

        # Extra handling for DMD2
        if self.is_image_batch(data_batch):
            if not self.config.ladd:
                # Turn off CP for fake score
                self.fake_score.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    if not self.config.ladd:
                        self.fake_score.enable_context_parallel(cp_group)

        data_batch["context_parallel_setup"] = True
        return data_batch, x0, condition, uncondition, epsilon

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
    ) -> Tuple[dict[str, torch.Tensor], float | torch.Tensor]:
        """
        Compute the DMD2 loss according to the current distillation phase: student or discriminator.

        This method is overloaded to support a dedicated `uncondition` argument.

        Args:
            uncondition (BaseVideoCondition): configuration for unconditional generation.

        Returns:
            Tuple(output_batch, loss): pair containing an output dictionary and the scalar total loss for current batch.
        """
        # Toggle CP accordingly
        data_batch, x0, condition, uncondition, epsilon = self.setup_context_parallel(
            data_batch, x0, condition, uncondition, epsilon
        )

        if iteration % self.config.student_update_freq == 0:
            # update the student
            if not self.config.ladd:
                self.fake_score.requires_grad_(False)
            if self.config.gan_loss_weight_gen > 0 or self.config.ladd:
                self.discriminator.requires_grad_(False)
        else:
            # update the fake_score and discriminator
            if not self.config.ladd:
                self.fake_score.requires_grad_(True)
            if self.config.gan_loss_weight_gen > 0 or self.config.ladd:
                self.discriminator.requires_grad_(True)

        # Generate noisy observations (using a new eps)
        eps = torch.randn_like(epsilon).to(**self.tensor_kwargs)

        if iteration % self.config.student_update_freq == 0:
            output_batch, loss = self._student_phase(x0, condition, uncondition, epsilon, sigma, eps)
        else:
            output_batch, loss = self._discriminator_phase(x0, condition, epsilon, sigma, eps)
        return output_batch, loss

    def _student_phase(
        self,
        x0: torch.Tensor,
        condition: BaseVideoCondition,
        uncondition: BaseVideoCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        eps: torch.Tensor,
    ):
        """Perform distillation phase for tuning the Student network. Computes GAN and VSD losses wrt Student only."""
        # update the student
        x0_gen = self._forward(epsilon, condition=condition)
        if self.config.recon_loss_weight > 0 or self.config.recon_loss_only:
            recon_loss = torch.nn.functional.mse_loss(x0_gen, x0, reduction="mean")
            if self.config.recon_loss_only:
                # Update the student with the reconstruction loss only
                # This can be used as an alternative way to warm up the student before DMD2
                output_batch = {"gen_rand": x0_gen, "recon_loss": recon_loss}
                return output_batch, recon_loss

        # Get the mean and standard deviation of the marginal probability distribution.
        mean, std = self.sde.marginal_prob(x0_gen, sigma)
        # Apply the forward process to get the noisy student sample
        xt_gen = mean + batch_mul(std, eps)

        if self.config.gan_loss_weight_gen > 0 or self.config.ladd:
            teacher_output, fake_feats = self.denoise_net(
                self.teacher,
                xt_gen,
                sigma=sigma,
                condition=condition,
                feature_indices=self.discriminator.feature_indices,
            )
            # concatenate the features from different ranks in the same CP group, with gradient
            if parallel_state.is_initialized() and parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                for idx in range(len(fake_feats)):
                    fake_feats[idx] = cat_outputs_cp_with_grad(
                        fake_feats[idx].contiguous(), seq_dim=1, cp_group=cp_group
                    )  # B, T, H, W, D
            if self.config.ladd:
                # Compute the LADD loss.
                gan_gen_loss = gan_loss_generator(self.discriminator(fake_feats))
                if not self.config.recon_loss_weight > 0:
                    recon_loss = torch.zeros_like(gan_gen_loss)
                loss = gan_gen_loss + self.config.recon_loss_weight * recon_loss
                output_batch = {
                    "gen_rand": x0_gen,
                    "total_loss": loss,
                    "gan_gen_loss": gan_gen_loss,
                }
                return output_batch, loss
        else:
            teacher_output = self.denoise_net(self.teacher, xt_gen, sigma=sigma, condition=condition)
        teacher_eps = teacher_output.eps

        with torch.no_grad():
            fake_score_eps = self.denoise_net(self.fake_score, xt_gen, sigma=sigma, condition=condition).eps

        if self.config.guidance_scale > 0:
            assert uncondition is not None, "Missing uncondition, required for CFG"

            # classifier-free guidance
            with torch.no_grad():
                teacher_eps_neg = self.denoise_net(
                    self.teacher,
                    xt_gen,
                    sigma=sigma,
                    condition=uncondition,
                ).eps

            teacher_eps = teacher_eps + self.config.guidance_scale * (teacher_eps - teacher_eps_neg)

        vsd_loss = variational_score_distillation_loss(x0_gen, teacher_eps, fake_score_eps, eps, sigma)

        if self.config.gan_loss_weight_gen > 0:
            # Compute the GAN loss for the generator
            gan_gen_loss = gan_loss_generator(self.discriminator(fake_feats))
        else:
            gan_gen_loss = torch.zeros_like(vsd_loss)

        if not self.config.recon_loss_weight > 0:
            recon_loss = torch.zeros_like(vsd_loss)

        # Compute the final loss
        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_gen_loss + self.config.recon_loss_weight * recon_loss

        output_batch = {
            "gen_rand": x0_gen,
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_gen_loss": gan_gen_loss,
            "recon_loss": recon_loss,
        }
        return output_batch, loss

    def _discriminator_phase(
        self,
        x0: torch.Tensor,
        condition: BaseVideoCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        eps: torch.Tensor,
    ):
        """
        Perform distillation phase for tuning the Discriminator and Fake Score networks.

        Computes GAN loss wrt Discriminator and denoising score matching loss wrt FakeScore.
        """
        # update the fake_score and discriminator
        with torch.no_grad():
            x0_gen = self._forward(epsilon, condition=condition)
            # Get the mean and standard deviation of the marginal probability distribution.
            mean, std = self.sde.marginal_prob(x0_gen, sigma)
            # Apply the forward process to get the noisy student sample
            xt_gen_sg = mean + batch_mul(std, eps)

        if self.config.gan_loss_weight_gen > 0 or self.config.ladd:
            # Compute the GAN loss for the discriminator
            with torch.no_grad():
                fake_feats = self.denoise_net(
                    self.teacher,
                    xt_gen_sg,
                    sigma=sigma,
                    condition=condition,
                    feature_indices=self.discriminator.feature_indices,
                    return_features_early=True,
                )

                real_mean, real_std = self.sde.marginal_prob(x0, sigma)
                # Generate noisy observations (using a new eps)
                teacher_eps = torch.randn_like(epsilon).to(**self.tensor_kwargs)
                xt_real = real_mean + batch_mul(real_std, teacher_eps)

                real_feats = self.denoise_net(
                    self.teacher,
                    xt_real,
                    sigma=sigma,
                    condition=condition,
                    feature_indices=self.discriminator.feature_indices,
                    return_features_early=True,
                )

                # concatenate the features from different ranks in the same CP group, without gradient
                if parallel_state.is_initialized() and parallel_state.get_context_parallel_world_size() > 1:
                    cp_group = parallel_state.get_context_parallel_group()
                    for idx in range(len(fake_feats)):
                        fake_feats[idx] = cat_outputs_cp(
                            fake_feats[idx].contiguous(), seq_dim=1, cp_group=cp_group
                        )  # B, T, H, W, D
                        real_feats[idx] = cat_outputs_cp(
                            real_feats[idx].contiguous(), seq_dim=1, cp_group=cp_group
                        )  # B, T, H, W, D

            gan_disc_loss = gan_loss_discriminator(self.discriminator(real_feats), self.discriminator(fake_feats))
            if self.config.ladd:
                # Compute the LADD loss.
                loss = gan_disc_loss
                output_batch = {"gen_rand": x0_gen, "total_loss": loss, "gan_disc_loss": gan_disc_loss}
                return output_batch, loss

        fake_score_eps = self.denoise_net(self.fake_score, xt_gen_sg, sigma=sigma, condition=condition).eps
        loss_fakescore = denoising_score_matching_loss("eps", xt_gen_sg, eps, fake_score_eps, sigma)

        if self.config.gan_loss_weight_gen <= 0:
            gan_disc_loss = torch.zeros_like(loss_fakescore)

        loss = loss_fakescore + gan_disc_loss

        output_batch = {
            "gen_rand": x0_gen,
            "fake_score_loss": loss_fakescore,
            "gan_disc_loss": gan_disc_loss,
        }
        return output_batch, loss

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """
        Chained initialisation of sequence parallelism on relevant modules up the class hierarchy. Extended to include
        FakeScore module, skipping Discriminator at the present.
        """
        super(DMD2ModelMixin, self).on_train_start(memory_format)
        if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if sequence_parallel:
                if not self.config.ladd:
                    self.fake_score.enable_sequence_parallel()


class DMD2DistillT2WModel(DMD2ModelMixin, BaseDistillationModel):
    pass


class DMD2DistillV2WModel(DMD2ModelMixin, V2WDistillationModel):
    pass


class DMD2DistillCtrlModel(DMD2ModelMixin, CtrlDistillationModel):
    pass
