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

from loguru import logger as logging
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType

from cosmos_transfer1.diffusion.config.config import Config
from cosmos_transfer1.distillation.train import destroy_distributed, instantiate_model
from cosmos_transfer1.distillation.utils import fsdp_distill
from cosmos_transfer1.utils import distributed, log
from cosmos_transfer1.utils.config_helper import get_config_module, override
from cosmos_transfer1.utils.easy_io import easy_io

"""
Script to convert a distributed FSDP distilled checkpoint to a native checkpoint.

Must specify an experiment specifying the model configuration and pass the checkpoint to convert.

Example usage:
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 cosmos_transfer1/distillation/scripts/convert_fsdp_dcp_to_native_ckpt.py \
        --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- \
        experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train \
        job.name=checkpoint_conversion \
        checkpoint.load_path=<path_to_checkpoint>
"""


def convert_and_save(config, trainer, model) -> None:
    if "fake_score" in model.model:
        log.info("Deleting fake_score module")
        del model.model["fake_score"]
    if "discriminator" in model.model:
        log.info("Deleting discriminator module")
        del model.model["discriminator"]
    if "base_model" in model.model:
        log.info("Deleting duplicate base_model module")
        del model.model["base_model"]

    model.on_train_start(trainer.config.trainer.memory_format)
    fsdp_distill.model_to_fsdp(model)
    trainer.checkpointer.load(model, {"net": None}, None, None)

    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        state_dict = model.state_dict()
    if distributed.is_rank0():
        save_path = f'{config.checkpoint.load_path.replace(".pt", "_full.pt")}'
        log.info(f"Uploading consolidated checkpoint to: {save_path}")
        easy_io.dump(state_dict, save_path, fast_backend=True)
    distributed.barrier()

    trainer.checkpointer.finalize()
    distributed.barrier()
    trainer.callbacks.on_app_end()


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)

    # Create the model
    model = instantiate_model(config)

    # Start conversion
    convert_and_save(config, trainer, model)
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
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    launch(config, args)
