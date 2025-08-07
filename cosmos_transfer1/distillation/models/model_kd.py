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
from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.diffusion.training.models.model import DiffusionModel, _broadcast
from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel
from cosmos_transfer1.distillation.models.ctrl_model_distill import CtrlDistillationModel
from cosmos_transfer1.distillation.models.v2w_model_distill import V2WDistillationModel
from cosmos_transfer1.utils import log


class KDModelMixin(DiffusionModel, ABC):
    """Knowledge distillation mixin class for diffusion step distillation."""

    def build_model(self) -> torch.nn.ModuleDict:
        model_dict = super().build_model()
        return model_dict

    def setup_context_parallel(
        self,
        data_batch: dict[str, torch.Tensor],
        x0: torch.Tensor,
        condition: BaseVideoCondition,
        uncondition: BaseVideoCondition,
        epsilon: torch.Tensor,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor, BaseVideoCondition, torch.Tensor]:
        """Toggle context parallelism, splitting any and all tensors for which CP is relevant. Note, tensor broadcasting
        is performed at initialisation time, not here.

        This overload splits `data_batch["noise"]`.

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

        # Extra handling for KD
        if not self.is_image_batch(data_batch):
            if not data_batch.get("validation_mode", False) and parallel_state.get_context_parallel_world_size() > 1:
                # Turn on CP
                cp_group = parallel_state.get_context_parallel_group()
                log.debug("[CP] Split KD noise")
                data_batch["noise"] = split_inputs_cp(x=data_batch["noise"], seq_dim=2, cp_group=cp_group)

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
        **kwargs,
    ):
        """
        Compute the KD loss.

        Returns:
            Tuple(output_batch, loss): pair containing an output dictionary and the scaled L2 reconstruction loss.
        """

        if isinstance(condition, Tuple):
            condition, _ = condition

        # Broadcast the noise to all GPUs
        data_batch["noise"] = _broadcast(data_batch["noise"], to_tp=True, to_cp=not self.is_image_batch(data_batch))

        # Toggle CP accordingly
        data_batch, x0, condition, uncondition, epsilon = self.setup_context_parallel(
            data_batch, x0, condition, uncondition, epsilon
        )

        x0_gen = self._forward(data_batch["noise"], condition=condition)

        # Compute the l2 loss between the generated data and the denoised data
        loss = 0.5 * torch.nn.functional.mse_loss(x0_gen, x0, reduction="mean")

        # Build output dictionaries
        output_batch = {"gen_rand": x0_gen, "recon_loss": loss}

        loss = loss * self.loss_scale

        return output_batch, loss


class KDDistillT2WModel(KDModelMixin, BaseDistillationModel):
    pass


class KDDistillV2WModel(KDModelMixin, V2WDistillationModel):
    pass


class KDDistillCtrlModel(KDModelMixin, CtrlDistillationModel):
    pass
