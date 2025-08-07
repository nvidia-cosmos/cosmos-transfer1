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

import enum
import multiprocessing
import os
import time
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, Dict, Set, Tuple, Union

import attrs
import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from cosmos_transfer1.checkpointer.base import AbstractCheckpointer
from cosmos_transfer1.checkpointer.ema_fsdp_checkpointer import CheckpointConfig
from cosmos_transfer1.utils import callback, distributed, log, misc
from cosmos_transfer1.utils.config import JobConfig
from cosmos_transfer1.utils.ddp_config import make_freezable
from cosmos_transfer1.utils.easy_io import easy_io

if TYPE_CHECKING:
    from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel


@make_freezable
@attrs.define(slots=False)
class DistillCheckpointConfig(CheckpointConfig):
    # Load student model only. When debugging, optionally set to True to speed up checkpoint loading time.
    # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
    load_student_only_for_debug: bool = False
    # Load both model and optimizer state to maintain the optimizer's momentum values to avoid a longer warmup phase.
    # One sample use case is training from a checkpoint that used the same experiment settings except for batch size.
    # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
    load_model_optim_only: bool = False
    # Skip loading discriminator if the checkpoint has a different (or no) discriminator architecture.
    # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
    skip_load_discriminator: bool = False
    # Skip loading fake score if the checkpoint has no fake score network.
    # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
    skip_load_fake_score: bool = False


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(self.model, model_state_dict=state_dict, options=StateDictOptions(strict=False))


class OptimizerWrapper(Stateful):
    """Wrapper for optimizer state dict handling"""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(
            self.model, self.optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    pass


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config: DistillCheckpointConfig,
    job_config: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistillFSDPCheckpointer(checkpoint_config, job_config, None, disable_async=True)

    try:
        while True:
            log.debug("Checkpoint background process is ready for next task")
            sender_queue.put(SaveDone())

            log.debug("Waiting to receive new state_dict")
            received_data = receiver_queue.get()
            log.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                log.info("Received termination signal for checkpoint background process")
                return

            assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)

            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds")

    finally:
        log.info("Cleaning up: destroying distributed process group")
        torch.distributed.destroy_process_group()


class DistillFSDPCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]
    KEYS_TO_POSTFIX = {
        "model": "model",
        "optim": "optim",
        "scheduler": "scheduler.pt",
        "trainer": "",
    }

    def __init__(
        self,
        config_checkpoint: DistillCheckpointConfig,
        config_job: JobConfig,
        callbacks: callback.CallBackGroup,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> Tuple[Set, str | None, bool]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []
        resume_training = latest_checkpoint_file is not None

        if resume_training:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.save_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                elif self.config_checkpoint.load_student_only_for_debug:
                    # only load student model
                    resume_keys = ["model"]
                elif self.config_checkpoint.load_model_optim_only:
                    # load both model and optimizer state to maintain the optimizer's momentum values
                    resume_keys = ["model", "optim"]
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]
        return set(resume_keys), checkpoint_path, resume_training

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: BaseDistillationModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        self.callbacks.on_load_checkpoint_start(model)
        model_dict = model.model

        resume_keys, checkpoint_path, resume_training = self.keys_to_resume_during_load()
        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            log.info(f"Loading checkpoint from {checkpoint_path}...")

            resume_keys = sorted(resume_keys)
            for key in resume_keys:
                cur_key_ckpt_full_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path)
                torch.distributed.barrier()
                if key == "model":
                    for k in optimizer_dict.keys():
                        if self.config_checkpoint.load_student_only_for_debug and k != "net":
                            continue
                        # If we resume training, we should always load all components of the checkpoint.
                        elif (
                            self.config_checkpoint.skip_load_discriminator
                            and k == "discriminator"
                            and not resume_training
                        ):
                            continue
                        elif self.config_checkpoint.skip_load_fake_score and k == "fake_score" and not resume_training:
                            continue
                        log.info(f"- Loading the model {k}...")
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        _model_wrapper = ModelWrapper(model=model_dict[k])
                        _state_dict = _model_wrapper.state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=DefaultLoadPlanner(allow_partial_load=True),
                        )
                        _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    for k, v in optimizer_dict.items():
                        log.info(f"- Loading the optimizer {k}...")
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        _optim_wrapper = OptimizerWrapper(model_dict[k], optimizer=v)
                        _state_dict = _optim_wrapper.state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=DefaultLoadPlanner(allow_partial_load=True),
                        )
                        _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = easy_io.load(
                        cur_key_ckpt_full_path,
                        fast_backend=False,
                    )
                    for k, v in _state_dict.items():
                        scheduler_dict[k].load_state_dict(v)
                elif key == "trainer":
                    log.info("- Loading the trainer...")
                    _state_dict = easy_io.load(
                        cur_key_ckpt_full_path,
                        fast_backend=False,
                    )
                    log.critical(_state_dict.keys(), rank0_only=False)
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model)
        return iteration

    def _async_with_pinned_memory(self, checkpoint_file: str, state_dict: Dict[str, Any]) -> None:
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            log.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file

        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))

            sync_func()
            self.staging = False

    def get_storage_writer(self, checkpoint_path: str) -> Union[FileSystemWriter]:
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[FileSystemReader]:
        return FileSystemReader(checkpoint_path)

    def save_state_dict_worker(self, to_save_dict: Dict[str, Any], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            if k in ["model", "optim"]:
                for key_net, state_dict in v.items():
                    storage_writer = self.get_storage_writer(f"{full_checkpoint_path}_{key_net}")
                    dcp.save(
                        state_dict,
                        storage_writer=storage_writer,
                    )
            else:
                if distributed.get_rank() == 0:
                    easy_io.dump(
                        v,
                        full_checkpoint_path,
                    )
        self._write_latest_checkpoint_file(checkpoint_file)
        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: BaseDistillationModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler],
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (BaseDistillationModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)
        torch.cuda.empty_cache()
        model_dict = model.model

        checkpoint_file = f"iter_{iteration:09}.pt"
        to_save_dict = {
            "model": {k: ModelWrapper(model=model_dict[k]).state_dict() for k in optimizer_dict.keys()},
            "optim": {k: OptimizerWrapper(model_dict[k], optimizer=v).state_dict() for k, v in optimizer_dict.items()},
            "scheduler": {k: v.state_dict() for k, v in scheduler_dict.items()},
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        to_save_dict_cpu = misc.to(to_save_dict, device="cpu")  # save memory by moving save_dict to cpu
        del to_save_dict
        torch.cuda.empty_cache()

        for k in to_save_dict_cpu.keys():
            full_checkpoint_path = self.add_type_postfix_to_checkpoint_path(k, checkpoint_file)
            full_checkpoint_path = os.path.join(self.save_dirname, full_checkpoint_path)
            to_save_dict_cpu[k] = (to_save_dict_cpu[k], full_checkpoint_path)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict_cpu)
        else:
            self.save_state_dict_worker(to_save_dict_cpu, checkpoint_file)

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_file: str) -> str:
        assert key in self.KEYS_TO_SAVE
        post_fix = self.KEYS_TO_POSTFIX[key]

        if post_fix:
            _ckpt_path = checkpoint_file.replace(".pt", f"_{post_fix}")
        else:
            _ckpt_path = checkpoint_file
        return _ckpt_path

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
