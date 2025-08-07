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

import argparse
import importlib
import os
from typing import TYPE_CHECKING

import torch.distributed as dist
from loguru import logger as logging
from omegaconf import OmegaConf

from cosmos_transfer1.diffusion.config.config import Config
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.config_helper import get_config_module, override
from cosmos_transfer1.utils.lazy_config import instantiate
from cosmos_transfer1.utils.lazy_config.lazy import LazyConfig

if TYPE_CHECKING:
    from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel

"""
The training entry script for distillation. Works for both DDP and FSDP training.
Due to the new FSDP implementation for distillation training, the `config.model_obj` does not take `fsdp_checkpointer`
as an attribute, which differs from `cosmos_transfer1/diffusion/training/train.py`.
"""


@misc.timer("instantiate model")
def instantiate_model(config: Config) -> BaseDistillationModel:
    misc.set_random_seed(seed=config.trainer.seed, by_rank=False)
    config.model_obj.config = config.model
    if config.model.fsdp_enabled:
        log.critical("FSDP enabled")
    model = instantiate(config.model_obj)
    config.model_obj.config = None
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
    return model


def destroy_distributed():
    log.info("Destroying distributed environment...")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except ValueError as e:
            print(f"Error destroying default process group: {e}")


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)

    # Create the model
    model = instantiate_model(config)

    # Start training
    trainer.train(
        args,
        model,
    )
    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config", default="cosmos_transfer1/distillation/config/config_ctrl_dmd2.py", help="Path to the config file"
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--mp0_only_dl",
        action="store_true",
        help="Use only model parallel rank 0 dataloader for faster dataloading! Make sure mock data has same keys as real data.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(OmegaConf.to_yaml(OmegaConf.load(f"{config.job.path_local}/config.yaml")))
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
