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

from typing import Any, List

import attrs

from cosmos_transfer1.distillation.checkpointer.distill_fsdp_checkpointer import DistillCheckpointConfig
from cosmos_transfer1.distillation.config.base.model import DMD2ModelConfig
from cosmos_transfer1.distillation.config.registry import register_configs
from cosmos_transfer1.distillation.models.model_dmd2 import DMD2DistillT2WModel
from cosmos_transfer1.distillation.trainer.distillation_trainer import Trainer
from cosmos_transfer1.utils import config
from cosmos_transfer1.utils.config_helper import import_all_modules_from_package
from cosmos_transfer1.utils.lazy_config import PLACEHOLDER
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"discriminator_optimizer": "fusedadamw"},
            {"fake_score_optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"discriminator_scheduler": "lambdalinear"},
            {"fake_score_scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"net": None},
            {"discriminator": "conv3d_pool_faditv2"},
            {"conditioner": "add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"tokenizer": None},
            {"checkpoint": "local"},
            {"ckpt_klass": "fsdp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(DMD2DistillT2WModel)(
        config=PLACEHOLDER,
    )

    checkpoint: DistillCheckpointConfig = attrs.field(factory=DistillCheckpointConfig)


def make_config():
    c = Config(
        model=DMD2ModelConfig(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_transfer1_distill"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None
    c.trainer.ddp.static_graph = False
    c.trainer.ddp.find_unused_parameters = True

    # Call this function to register config groups for advanced overriding.
    register_configs()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_transfer1.distillation.config.experiment", reload=True)
    return c
