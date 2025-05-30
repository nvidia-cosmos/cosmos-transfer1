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

import os
from typing import Any, Dict, Optional, Type, Union

import attrs
import torch

try:
    from megatron.core import ModelParallelConfig

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")

from cosmos_transfer1.utils.callback import EMAModelCallback, ProgressBarCallback
from cosmos_transfer1.utils.ddp_config import DDPConfig, make_freezable
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.utils.misc import Color


def _pretty_print_attrs_instance(obj: object, indent: int = 0, use_color: bool = False) -> str:
    """
    Recursively pretty prints attrs objects with color.
    """

    assert attrs.has(obj.__class__)

    lines: list[str] = []
    for attribute in attrs.fields(obj.__class__):
        value = getattr(obj, attribute.name)
        if attrs.has(value.__class__):
            if use_color:
                lines.append("   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ":")
            else:
                lines.append("   " * indent + "* " + attribute.name + ":")
            lines.append(_pretty_print_attrs_instance(value, indent + 1, use_color))
        else:
            if use_color:
                lines.append(
                    "   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ": " + Color.yellow(value)
                )
            else:
                lines.append("   " * indent + "* " + attribute.name + ": " + str(value))
    return "\n".join(lines)


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    # Project name.
    project: str = ""
    # Experiment name.
    group: str = ""
    # Run/job name.
    name: str = ""

    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("OUTPUT_ROOT", "checkpoints")
        return f"{local_root}/{self.path}"


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EMA decay rate.
    beta: float = 0.9999
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    # Set to True for better reproducibility of the results (only using deterministic cudnn functions).
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class JITConfig:
    # Enable exporting a JIT compiled model.
    enabled: bool = False
    # Input tensor shape, for example input.
    input_shape: Union[list[int], None] = None
    # Device to compile onto.
    device: str = "cuda"
    # # Data type to compile onto.
    dtype: str = "bfloat16"
    # Strict mode for PyTorch JIT.
    strict: bool = True


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    # possible checkpoint class
    type: Optional[Dict] = None
    # for dcp, whether to use async mode
    dcp_async_mode_enabled: bool = False
    # Save the checkpoint every N iterations.
    save_iter: int = 999999999
    # Path of model weights to resume the checkpoint from.
    load_path: str = ""
    # Whether to load the training states (optimizer/scheduler/grad-scaler) from the checkpoint path.
    load_training_state: bool = False
    # Whether to load the scheduler state only from the checkpoint path. If load_training_state is True, this will be ignored.
    only_load_scheduler_state: bool = False
    # Load state_dict to the models in strict mode.
    strict_resume: bool = True
    # Print detailed information during checkpoint saving/loading.
    verbose: bool = True
    # Configs for JIT compiling EMA model.
    jit: JITConfig = attrs.field(factory=JITConfig)
    # keys not to resume from the checkpoint, choices: ["model", "optim", "scheduler", "trainer"]
    keys_not_to_resume: list[str] = []
    # Whether to use the local filesystem for broadcasting checkpoint data (used for Tensor Parallel Checkpointer).
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False


@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    from cosmos_transfer1.utils.trainer import Trainer

    type: Type[Trainer] = Trainer
    # Set the callback class.
    # Defaults to the callbacks below.
    callbacks: LazyDict = LazyDict(
        dict(
            ema=L(EMAModelCallback)(),
            progress_bar=L(ProgressBarCallback)(),
        )
    )
    # distributed parallelism strategy
    distributed_parallelism: str = "ddp"
    # Distributed data parallel configs.
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    # cuDNN configs.
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    # Set the random seed.
    seed: int = 0
    # Gradient scaler arguments (for torch.amp.GradScaler).
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    # Maximum number of iterations to train the model.
    max_iter: int = 999999999
    # Maximum number of iterations to validate the model. If None, validate on the entire dataset.
    max_val_iter: int | None = None
    # How often we log the training stats.
    logging_iter: int = 100
    # Whether we want to run the validation routines.
    run_validation: bool = True
    # How often we evaluate on the validation set.
    validation_iter: int = 999999999
    # Kill the process after N seconds since the last iteration (usually means dead job).
    timeout_period: int = 999999999
    # Tensor memory organization format.
    memory_format: torch.memory_format = torch.preserve_format
    # Gradient accumulation (update step every N iteration).
    grad_accum_iter: int = 1
    # Whether to use the timestamp as the seed. Needed to ensure real randomness in loading data.
    timestamp_seed: bool = True
    # # Profiling config
    # profiling: Profiling = attrs.field(factory=Profiling)


@make_freezable
@attrs.define(slots=False)
class Config:
    """Config for a job.

    See /README.md/Configuration System for more info.
    """

    # Model configs.
    model: LazyDict
    # Optimizer configs.
    optimizer: LazyDict = LazyDict(dict(dummy=None))
    # Scheduler configs.
    scheduler: LazyDict = LazyDict(dict(dummy=None))
    # Training data configs.
    dataloader_train: LazyDict = LazyDict(dict(dummy=None))
    # Validation data configs.
    dataloader_val: LazyDict = LazyDict(dict(dummy=None))

    # Training job configs.
    job: JobConfig = attrs.field(factory=JobConfig)

    # Trainer configs.
    trainer: TrainerConfig = attrs.field(factory=TrainerConfig)

    # Megatron-Core configs
    if USE_MEGATRON:
        # Megatron-Core configs
        model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)
    else:
        model_parallel: None = None

    # Checkpointer configs.
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def validate(self) -> None:
        """Validate that the config has all required fields."""
        assert self.job.project != "", "Project name is required."
        assert self.job.group != "", "Group name is required."
        assert self.job.name != "", "Job name is required."
