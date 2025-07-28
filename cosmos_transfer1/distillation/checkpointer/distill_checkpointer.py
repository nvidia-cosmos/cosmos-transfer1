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

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from cosmos_transfer1.checkpointer.ddp_checkpointer import Checkpointer
from cosmos_transfer1.diffusion.inference.inference_utils import non_strict_load_model
from cosmos_transfer1.utils import distributed, log, misc
from cosmos_transfer1.utils.model import Model

if TYPE_CHECKING:
    from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel


class DistillCheckpointer(Checkpointer):
    """
    Checkpointer class for Distillation in distributed training (DDP).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_save_state_dict(
        self,
        model: BaseDistillationModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler],
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        state_dict = {}

        if self.rank_dp_w_cp == 0:
            trainer_state = dict(
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            model_state = model.state_dict()
            optim_state = {k: v.state_dict() for k, v in optimizer_dict.items()}
            scheduler_state = {k: v.state_dict() for k, v in scheduler_dict.items()}
            self.callbacks.on_save_checkpoint(model, state_dict=trainer_state)

            trainer_state, model_state, optim_state, scheduler_state = misc.to(
                [trainer_state, model_state, optim_state, scheduler_state], device="cpu"
            )

            state_dict = {
                "model": model_state,
                "optim": optim_state,
                "scheduler": scheduler_state,
            }
            if distributed.get_rank() == 0:  # only rank 0 saves trainer state
                state_dict["trainer"] = trainer_state
            return state_dict
        return state_dict

    def load(
        self,
        model: BaseDistillationModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (torch.nn.ModuleDict): The dict of PyTorch model.
            optimizer_dict (Dict[str, torch.optim.Optimizer]): The dict of model optimizer (default: None).
            scheduler_dict (Dict[str, torch.optim.lr_scheduler._LRScheduler]): The dict of optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        iteration = 0

        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            state_dict = self.load_broadcast_state_dict(checkpoint_path, model, set(resume_keys))

            if "trainer" in state_dict:
                trainer_state = state_dict["trainer"]
                log.critical(state_dict.keys(), rank0_only=False)
                log.critical(trainer_state, rank0_only=False)
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(trainer_state["grad_scaler"])
                self.callbacks.on_load_checkpoint(model, state_dict=trainer_state)
                iteration = trainer_state["iteration"]

            if "optim" in state_dict:
                assert optimizer_dict
                optimizer_state = state_dict["optim"]
                log.info("- Loading the optimizer...")
                for k, v in optimizer_dict.items():
                    if k in optimizer_state:
                        v.load_state_dict(optimizer_state[k])
                    else:
                        log.warning(f"Optimizer {k} not found in checkpoint.")

            if "scheduler" in state_dict:
                assert scheduler_dict
                scheduler_state = state_dict["scheduler"]
                log.info("- Loading the scheduler...")
                for k, v in scheduler_dict.items():
                    if k in scheduler_state:
                        v.load_state_dict(scheduler_state[k])
                        v.last_epoch = iteration
                    else:
                        log.warning(f"Scheduler {k} not found in checkpoint.")

            if "model" in state_dict:
                model_state = state_dict["model"]
                log.info("- Loading the model...")
                if self.strict_resume:
                    log.info("\t Strict resume mode is on.")
                    model_load_info = model.load_state_dict(model_state, strict=self.strict_resume)
                    log.info(f"\t {model_load_info}")
                else:
                    log.info("\t Strict resume mode is off.")
                    log.critical(non_strict_load_model(model, model_state))
            self.print(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model)

        return iteration


class DistillTPCheckpointer(DistillCheckpointer):
    """
    Checkpointer class for Tensor Parallelism (TP) in distributed training (DDP).

    This implementation supports the combination of Tensor Parallelism (TP) and Data Parallel Processing (DDP), with optional Context Parallelism (CP).

    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    - In principle, this implementation is also compatible with Pipeline Parallelism (PP) and Expert Parallelism (EP), which are other forms of model parallelism. However, PP and EP have not been tested yet.
    """

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: Model) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to append the TP-rank postfix to the checkpoint path.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        if key == "trainer":
            return checkpoint_path
        else:
            checkpoint_path = checkpoint_path.replace(".pt", f"_mp_{self.mp_rank}.pt")

        return checkpoint_path
