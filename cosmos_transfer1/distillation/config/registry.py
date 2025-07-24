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

from hydra.core.config_store import ConfigStore

from cosmos_transfer1.diffusion.config.transfer.registry import register_experiment_ctrlnet
from cosmos_transfer1.diffusion.config.registry import register_net, register_conditioner, register_tokenizer
from cosmos_transfer1.diffusion.config.training.registry import register_checkpoint_credential
from cosmos_transfer1.diffusion.config.base.data import register_data_ctrlnet
from cosmos_transfer1.diffusion.config.training.callbacks import BASIC_CALLBACKS
from cosmos_transfer1.distillation.config.base.checkpoint import DISTILL_CHECKPOINTER, DISTILL_FSDP_CHECKPOINTER
from cosmos_transfer1.distillation.config.base.discriminator import (
    CONV3D_POOL_FADITV2_Config,
    CONV3D_POOL_TINY_FA_Config,
)
from cosmos_transfer1.distillation.config.base.fsdp import FULL_FSDP_CONFIG, HYBRID_FSDP_CONFIG
from projects.cosmos.nano.v1.config.debug import (
    DEBUG_LOCAL_CP_EXP,
    DEBUG_LOCAL_CP_FSDP_EXP,
    DEBUG_LOCAL_DDP_EXP,
    DEBUG_LOCAL_FSDP_EXP,
    DEBUG_LOCAL_KD_TPSP_CP_EXP,
    DEBUG_LOCAL_TINY_DDP_EXP,
    DEBUG_LOCAL_TP_EXP,
    DEBUG_LOCAL_TPSP_CP_EXP,
    DEBUG_V2W_LOCAL_CP_FSDP_EXP,
    DEBUG_V2W_LOCAL_TINY_DDP_EXP,
    DEBUG_V2W_LOCAL_TPSP_CP_EXP,
)
from cosmos_transfer1.diffusion.config.training.optim import FusedAdamWConfig, LambdaLinearSchedulerConfig


def register_fsdp(cs):
    cs.store(group="fsdp", package="_global_", name="full", node=FULL_FSDP_CONFIG)
    cs.store(group="fsdp", package="_global_", name="hybrid", node=HYBRID_FSDP_CONFIG)


def register_experiment(cs):
    # base model
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_tiny_ddp",
        node=DEBUG_LOCAL_TINY_DDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_ddp",
        node=DEBUG_LOCAL_DDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_cp",
        node=DEBUG_LOCAL_CP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_fsdp",
        node=DEBUG_LOCAL_FSDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_cp_fsdp",
        node=DEBUG_LOCAL_CP_FSDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_tp",
        node=DEBUG_LOCAL_TP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_tpsp_cp",
        node=DEBUG_LOCAL_TPSP_CP_EXP,
    )
    # v2w model
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_v2w_local_tiny_ddp",
        node=DEBUG_V2W_LOCAL_TINY_DDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_v2w_local_cp_fsdp",
        node=DEBUG_V2W_LOCAL_CP_FSDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_v2w_local_tpsp_cp",
        node=DEBUG_V2W_LOCAL_TPSP_CP_EXP,
    )
    # kd model
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_local_kd_tpsp_cp",
        node=DEBUG_LOCAL_KD_TPSP_CP_EXP,
    )


def register_callbacks(cs):
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)


def register_discriminator(cs):
    cs.store(
        group="discriminator",
        package="model.discriminator",
        name="conv3d_pool_tiny_fa",
        node=CONV3D_POOL_TINY_FA_Config,
    )
    cs.store(
        group="discriminator",
        package="model.discriminator",
        name="conv3d_pool_faditv2",
        node=CONV3D_POOL_FADITV2_Config,
    )


def register_checkpointer(cs):
    cs.store(group="ckpt_klass", package="checkpoint.type", name="multi_rank", node=DISTILL_CHECKPOINTER)
    cs.store(group="ckpt_klass", package="checkpoint.type", name="fsdp", node=DISTILL_FSDP_CHECKPOINTER)


def register_optimizers(cs):
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
    cs.store(
        group="discriminator_optimizer",
        package="model.discriminator_optimizer",
        name="fusedadamw",
        node=FusedAdamWConfig,
    )
    cs.store(
        group="fake_score_optimizer",
        package="model.fake_score_optimizer",
        name="fusedadamw",
        node=FusedAdamWConfig,
    )


def register_schedulers(cs):
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)
    cs.store(
        group="discriminator_scheduler",
        package="model.discriminator_scheduler",
        name="lambdalinear",
        node=LambdaLinearSchedulerConfig,
    )
    cs.store(
        group="fake_score_scheduler",
        package="model.fake_score_scheduler",
        name="lambdalinear",
        node=LambdaLinearSchedulerConfig,
    )


def register_configs():
    cs = ConfigStore.instance()

    register_net(cs)
    register_conditioner(cs)
    register_tokenizer(cs)

    register_fsdp(cs)
    register_callbacks(cs)
    register_checkpoint_credential(cs)
    register_checkpointer(cs)
    register_discriminator(cs)
    register_optimizers(cs)
    register_schedulers(cs)
    register_experiment(cs)

    register_data_ctrlnet(cs)
    register_experiment_ctrlnet(cs)
