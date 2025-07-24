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
from projects.cosmos.diffusion.v1.checkpointers.ema_fsdp_checkpointer import CheckpointConfig
from projects.cosmos.nano.v1.config.base.model import DistillModelConfig
from projects.cosmos.nano.v1.config.registry import register_configs
from projects.cosmos.nano.v1.models.model_kd import KDWorldDistillationModel
from projects.cosmos.nano.v1.trainer.distillation_trainer import Trainer


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock_video_noise"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"vae": "vae1"},
            {"checkpoint": "s3"},
            {"ckpt_klass": "fsdp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(KDWorldDistillationModel)(
        config=PLACEHOLDER,
    )

    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)


def make_config():
    c = Config(
        model=DistillModelConfig(),
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
    import_all_modules_from_package("projects.cosmos.nano.v1.config.experiment", reload=True)
    return c
