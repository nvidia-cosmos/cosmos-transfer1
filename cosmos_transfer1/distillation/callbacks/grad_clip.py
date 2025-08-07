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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from cosmos_transfer1.diffusion.training.callbacks.grad_clip import GradClip as GradClipVideo
from cosmos_transfer1.utils import distributed

if TYPE_CHECKING:
    from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel


class GradClip(GradClipVideo):
    """
    This callback is used for applying gradient clip to BaseDistillationModel particularly.
    Unlike the DiT base model training that updates one network, BaseDistillationModel contains multiple
    submodules that are updated alternatively with separate optimizers and lr schedulers.
    Thus, `model_key` is needed to let the `GradClip` callback know what network and what optimizer to operate on.
    """

    def on_before_optimizer_step(
        self,
        model_ddp: BaseDistillationModel | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp

        # unscale the optimizer related to the `model_key`: [net, fake_score, discriminator]
        assert (
            self.model_key in model.optimizer_dict.keys()
        ), f"Keys in optimizer_dict: {list(model.optimizer_dict.keys())}."
        grad_scaler.unscale_(model.optimizer_dict[self.model_key])

        if self.model_key is not None:
            items = self.model_key.split(".")
            for item in items:
                model = getattr(model, item)

            if self.force_finite:
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0.0, out=param.grad)

            if isinstance(model, FSDP) and self.fsdp_enabled:
                total_norm = model.clip_grad_norm_(self.clip_norm)
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm, foreach=True)

            self._cur_state.update(total_norm.detach().cpu())
