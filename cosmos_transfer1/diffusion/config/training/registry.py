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

'''
Core training related registry.
'''

from hydra.core.config_store import ConfigStore


from cosmos_transfer1.checkpointer.ema_fsdp_checkpointer import CheckpointConfig
from cosmos_transfer1.diffusion.config.training.ema import PowerEMAConfig
from cosmos_transfer1.diffusion.config.training.optim import FusedAdamWConfig, LambdaLinearSchedulerConfig
from cosmos_transfer1.diffusion.config.training.callbacks import BASIC_CALLBACKS
from cosmos_transfer1.diffusion.config.training.checkpoint import (
    FSDP_CHECKPOINTER,
    MULTI_RANK_CHECKPOINTER,
    MODEL_PARALLEL_CHECKPOINTER,
    FAST_TP_CHECKPOINTER,
)


def register_ema(cs):
    cs.store(group="ema", package="model.ema", name="power", node=PowerEMAConfig)


def register_optimizer(cs):
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)


def register_scheduler(cs):
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)

def register_callbacks(cs):
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)

def register_checkpoint_credential(cs):
    CHECKPOINT_LOCAL = CheckpointConfig(
        save_iter=1000,
        load_path="",
        load_training_state=False,
        strict_resume=True,
    )

    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)


def register_checkpointer(cs):
    cs.store(group="ckpt_klass", package="checkpoint.type", name="fsdp", node=FSDP_CHECKPOINTER)
    cs.store(group="ckpt_klass", package="checkpoint.type", name="multi_rank", node=MULTI_RANK_CHECKPOINTER)
    cs.store(group="ckpt_klass", package="checkpoint.type", name="tp", node=MODEL_PARALLEL_CHECKPOINTER)
    cs.store(group="ckpt_klass", package="checkpoint.type", name="fast_tp", node=FAST_TP_CHECKPOINTER)
    

def register_configs():
    cs = ConfigStore.instance()

    register_optimizer(cs)
    register_scheduler(cs)
    register_ema(cs)
    register_checkpoint_credential(cs)
    register_checkpointer(cs)
    register_callbacks(cs)