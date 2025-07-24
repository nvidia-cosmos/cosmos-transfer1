# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import Any, List

import attrs

from imaginaire import config
from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.utils.config_helper import import_all_modules_from_package
from projects.cosmos.nano.v1.checkpointers.distill_fsdp_checkpointer import DistillCheckpointConfig
from projects.cosmos.nano.v1.config.base.model import DMD2CtrlModelConfig
from projects.cosmos.nano.v1.config.registry import register_configs
from projects.cosmos.nano.v1.models import DMD2Video2WorldCtrlDistillationModel
from projects.cosmos.nano.v1.trainer.distillation_trainer import Trainer


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock_video"},
            {"data_val": "mock"},
            {"hint_key": None},
            {"optimizer": "fusedadamw"},
            {"discriminator_optimizer": "fusedadamw"},
            {"fake_score_optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"discriminator_scheduler": "lambdalinear"},
            {"fake_score_scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"net": None},
            {"net_ctrl": None},
            {"discriminator": "conv3d_pool_faditv2"},
            {"conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"vae": "vae1"},
            {"checkpoint": "s3"},
            {"ckpt_klass": "fsdp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(DMD2Video2WorldCtrlDistillationModel)(
        config=PLACEHOLDER,
    )

    checkpoint: DistillCheckpointConfig = attrs.field(factory=DistillCheckpointConfig)


def make_config():
    c = Config(
        model=DMD2CtrlModelConfig(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_nano_v1"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
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
    import_all_modules_from_package("projects.cosmos.nano.v1.config.experiment.v2w_ctrl_7B_dmd2", reload=True)
    return c
