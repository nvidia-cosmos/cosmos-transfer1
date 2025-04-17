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

from cosmos_transfer1.diffusion.config.transfer.model import CtrlModelConfig
from cosmos_transfer1.checkpointer.ema_fsdp_checkpointer import CheckpointConfig
from cosmos_transfer1.diffusion.config.training.registry_extra import register_configs
from cosmos_transfer1.diffusion.training.models.model_ctrl import VideoDiffusionModelWithCtrl
from cosmos_transfer1.utils import config
from cosmos_transfer1.utils.config_helper import import_all_modules_from_package
from cosmos_transfer1.utils.lazy_config import PLACEHOLDER
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.utils.trainer import Trainer


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
            {"scheduler": "lambdalinear"},
            {"callbacks": None},
            #
            {"net": None},
            {"net_ctrl": None},
            {"hint_key": "control_input_edge"},
            {"conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            {"pixel_corruptor": None},
            {"fsdp": None},
            {"ema": "power"},
            {"checkpoint": "local"},
            {"ckpt_klass": "multi_rank"},
            {"tokenizer": "vae1"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(VideoDiffusionModelWithCtrl)(
        config=PLACEHOLDER,
    )
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)



def make_config():
    c = Config(
        model=CtrlModelConfig(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "cosmos_transfer1"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    # c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    register_configs()
    import_all_modules_from_package("cosmos_transfer1.diffusion.config.training.experiment", reload=True)
    return c
