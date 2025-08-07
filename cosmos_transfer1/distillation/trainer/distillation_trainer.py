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

import argparse
import signal

import torch
import torch.utils.data

from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel
from cosmos_transfer1.distillation.utils import fsdp_distill
from cosmos_transfer1.utils import distributed, log, misc
from cosmos_transfer1.utils.lazy_config import instantiate
from cosmos_transfer1.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_transfer1.utils.trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """The trainer class for diffusion step distillation.

    It contains the basic functionality for model training (particularly suited for large-scale training),
    including data parallel (DDP/FSDP), mixed-precision training (fp16/bf16).

    Differences from BaseTrainer:
    - It supports alternative training among multiple trainable modules
    - It supports multiple optimizers and schedulers
    - No EMA support for now
    """

    def __init__(self, config):
        super().__init__(config)

    def train(
        self,
        args: argparse.Namespace,
        model: BaseDistillationModel,
    ) -> None:
        """The training function.

        Args:
            args (argparse.Namespace): The arguments passed to the training script.
            model (BaseDistillationModel): The video distillation model.
        """
        model.on_train_start(self.config.trainer.memory_format)

        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = fsdp_distill.model_to_fsdp(model)
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler(**self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()

        iteration = self.checkpointer.load(model, model.optimizer_dict, model.scheduler_dict, grad_scaler)
        grad_accum_iter = 0

        if args.mp0_only_dl:
            log.critical(
                "Using only tp_cp_pp_rank0 dataloader for faster dataloading! Make sure val dl is mock and mock data has same keys as real data."
            )
            raise NotImplementedError(
                "mp0_only_dl is not implemented correctly! Please revisit this code and propose a more robust impl that raise error timely! It does not do necessary check before training to confirm it can work with image / video data. Current impl is problematic for image training."
            )

        if is_tp_cp_pp_rank0() or not args.mp0_only_dl:
            dataloader_train = instantiate(self.config.dataloader_train)
        else:
            dataloader_train = instantiate(self.config.dataloader_val)
        dataloader_val = instantiate(self.config.dataloader_val)

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        while True:
            dataloader_train_iter = iter(dataloader_train)
            while True:
                self.callbacks.on_before_dataloading(iteration)
                with self.training_timer("dataloader_train"):
                    try:
                        data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                self.callbacks.on_after_dataloading(iteration)
                # If max_iter is reached, exit the training loop.
                if iteration >= self.config.trainer.max_iter:
                    _end_training = True
                    break
                # Move all tensors in the data batch to GPU device.
                data_batch = misc.to(data_batch, device="cuda")
                # The actual training step.
                self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                if not model.training:
                    model_ddp.train()
                assert model_ddp.training, "model_ddp is not in training mode."
                assert model.training, "model is not in training mode."
                output_batch, loss, grad_accum_iter = self.training_step(
                    model_ddp,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )
                # If the gradients are still being accumulated, continue to load the next training batch.
                if grad_accum_iter != 0:
                    continue
                # Do the following when an actual optimizer (update) step has been made.
                iteration += 1
                self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                # Validation.
                if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                    self.validate(model, dataloader_val, iteration=iteration)
                # Save checkpoint.
                if iteration % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save(
                        model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration
                    )
                # This iteration is successful; reset the timeout signal.
                signal.alarm(self.config.trainer.timeout_period)
            if _end_training:
                break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration)

        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: BaseDistillationModel | distributed.DistributedDataParallel,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
            grad_accum_iter (int): Number of gradient accumulation iterations (updated after each training_step).
        """
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp

        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                loss_scaled.backward()
                model.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp,
                    model.optimizer_dict["net"],  # back-compatibility
                    model.scheduler_dict["net"],  # back-compatibility
                    grad_scaler,
                    iteration=iteration,
                )
                model.optimizers_schedulers_step(grad_scaler, iteration=iteration)
                model.optimizers_zero_grad(iteration=iteration)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter
