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

from cosmos_transfer1.diffusion.config.base.data import register_data_ctrlnet
from cosmos_transfer1.diffusion.config.registry import register_conditioner
from cosmos_transfer1.diffusion.config.training.optim import FusedAdamWConfig, LambdaLinearSchedulerConfig
from cosmos_transfer1.diffusion.config.training.registry import register_checkpoint_credential
from cosmos_transfer1.diffusion.config.training.registry_extra import (
    register_conditioner_ctrlnet,
    register_net_train,
    register_tokenizer,
)
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS
from cosmos_transfer1.distillation.config.base.callbacks import BASIC_CALLBACKS, TRAIN_VIS_CALLBACK
from cosmos_transfer1.distillation.config.base.checkpoint import DISTILL_CHECKPOINTER, DISTILL_FSDP_CHECKPOINTER
from cosmos_transfer1.distillation.config.base.data import (
    get_dmd2_transfer_dataset,
    get_kd_transfer_dataset,
    get_mock_dataset,
)
from cosmos_transfer1.distillation.config.base.debug import (
    DEBUG_CTRLNET_LOCAL_CP_FSDP_EXP,
    DEBUG_CTRLNET_LOCAL_DDP_EXP,
    DEBUG_LOCAL_CP_EXP,
    DEBUG_LOCAL_CP_FSDP_EXP,
    DEBUG_LOCAL_DDP_EXP,
    DEBUG_LOCAL_FSDP_EXP,
)
from cosmos_transfer1.distillation.config.base.discriminator import (
    CONV3D_POOL_FADITV2_Config,
    CONV3D_POOL_TINY_FA_Config,
)
from cosmos_transfer1.distillation.config.base.fsdp import FULL_FSDP_CONFIG, HYBRID_FSDP_CONFIG


def register_fsdp(cs):
    cs.store(group="fsdp", package="_global_", name="full", node=FULL_FSDP_CONFIG)
    cs.store(group="fsdp", package="_global_", name="hybrid", node=HYBRID_FSDP_CONFIG)


def register_debug_experiments(cs):
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
        name="debug_ctrlnet_local_ddp",
        node=DEBUG_CTRLNET_LOCAL_DDP_EXP,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="debug_ctrlnet_local_cp_fsdp",
        node=DEBUG_CTRLNET_LOCAL_CP_FSDP_EXP,
    )


def register_callbacks(cs):
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="train_vis", node=TRAIN_VIS_CALLBACK)


def register_data(cs):
    # Mock dataset for debugging base model distillation with debug tokenizer
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock_distill_debug",
        node=get_mock_dataset(is_debug_tokenizer=True),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock_distill_debug",
        node=get_mock_dataset(is_debug_tokenizer=True),
    )

    for hint_key in CTRL_HINT_KEYS:
        # Mock dataset for debugging ctrlnet distillation with full tokenizer
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"mock_ctrl_distill_{hint_key}",
            node=get_mock_dataset(hint_key=hint_key, is_debug_tokenizer=False),
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"mock_ctrl_distill_{hint_key}",
            node=get_mock_dataset(hint_key=hint_key, is_debug_tokenizer=False),
        )

        # Mock dataset for debugging ctrlnet distillation with debug tokenizer
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"mock_ctrl_distill_debug_{hint_key}",
            node=get_mock_dataset(hint_key=hint_key, is_debug_tokenizer=True),
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"mock_ctrl_distill_debug_{hint_key}",
            node=get_mock_dataset(hint_key=hint_key, is_debug_tokenizer=True),
        )

        # Custom KD dataset
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"kd_transfer_train_data_{hint_key}",
            node=get_kd_transfer_dataset(hint_key=hint_key, is_train=True),
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"kd_transfer_val_data_{hint_key}",
            node=get_kd_transfer_dataset(hint_key=hint_key, is_train=False),
        )

        # Custom DMD2 dataset
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"dmd2_transfer_train_data_{hint_key}",
            node=get_dmd2_transfer_dataset(hint_key=hint_key, is_train=True),
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"dmd2_transfer_val_data_{hint_key}",
            node=get_dmd2_transfer_dataset(hint_key=hint_key, is_train=False),
        )


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

    # register all the basic configs: net, conditioner, tokenizer, checkpoint
    register_net_train(cs)
    register_conditioner(cs)
    register_conditioner_ctrlnet(cs)
    register_tokenizer(cs)
    register_checkpoint_credential(cs)

    # register distillation training configs
    register_callbacks(cs)
    register_checkpointer(cs)
    register_debug_experiments(cs)
    register_discriminator(cs)
    register_fsdp(cs)
    register_data(cs)
    register_optimizers(cs)
    register_schedulers(cs)

    # register ctrlnet data
    register_data_ctrlnet(cs)

    # register ctrlnet hint keys
    for hint_key in CTRL_HINT_KEYS:
        cs.store(
            group="hint_key",
            package="model",
            name=hint_key,
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=False)),
        )
        cs.store(
            group="hint_key",
            package="model",
            name=f"{hint_key}_grayscale",
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=True)),
        )
