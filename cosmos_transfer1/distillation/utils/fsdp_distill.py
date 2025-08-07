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

import functools
from typing import TYPE_CHECKING, Any, Callable, List

from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from cosmos_transfer1.diffusion.training.utils.fsdp_helper import apply_fsdp_checkpointing, hsdp_device_mesh
from cosmos_transfer1.utils import distributed, log

if TYPE_CHECKING:
    from cosmos_transfer1.distillation.models.base_model_distill import BaseDistillationModel


def model_to_fsdp(model: BaseDistillationModel) -> BaseDistillationModel:
    """Convert the networks in model (BaseDistillationModel) to FSDP separately."""
    # Collect model dict
    model_dict = model.model
    config = model.config

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
    }[config.fsdp.sharding_strategy]
    wrap_policy = get_wrap_policy(model)
    if config.fsdp.sharding_strategy == "full":
        device_mesh = init_device_mesh("cuda", (distributed.get_world_size(),))
    elif config.fsdp.sharding_strategy == "hybrid":
        device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp.sharding_group_size)
    else:
        raise NotImplementedError(f"Unsupported sharding strategy {config.fsdp.sharding_strategy}.")
    log.critical(f"Using {sharding_strategy} sharding strategy for FSDP")

    for k, v in model_dict.items():
        if k in ["logvar", "conditioner"]:
            continue
        elif k == "base_model":
            wrapped_v = module_dict_to_fsdp(v, sharding_strategy, wrap_policy, device_mesh)
        else:
            wrapped_v = FSDP(
                v,
                device_mesh=device_mesh,
                auto_wrap_policy=wrap_policy,
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                use_orig_params=True,
            )
        setattr(model, k, wrapped_v)
        model_dict[k] = wrapped_v
        log.info(f"Wrapped model {k}")

    if config.fsdp.checkpoint:
        fsdp_blocks_cls = get_fsdp_wrap_block_cls(model)
        log.critical(f"Applying FSDP checkpointing with FSDP blocks: {fsdp_blocks_cls}")
        apply_fsdp_checkpointing(model, list_block_cls=fsdp_blocks_cls)

    return model


def module_dict_to_fsdp(module_dict: dict, sharding_strategy, wrap_policy, device_mesh) -> dict:
    """Convert the nested modules in a ControlNet model to FSDP separately."""
    log.critical(f"Wrapping nested modules of base_model: {module_dict.keys()}")
    for k, v in module_dict.items():
        if k in ["logvar", "conditioner"]:
            continue
        wrapped_v = FSDP(
            v,
            device_mesh=device_mesh,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=sharding_strategy,
            sync_module_states=True,
            use_orig_params=True,
        )
        module_dict[k] = wrapped_v
        log.info(f"Wrapped nested model {k}")
    return module_dict


def get_fsdp_wrap_block_cls(model: BaseDistillationModel) -> List[Any]:
    """Get the list of FSDP wrap block classes (used when config.fsdp.policy=block)."""
    config = model.config
    if not hasattr(model.net, "fsdp_wrap_block_cls"):
        raise ValueError("model.net does not have fsdp_wrap_block_cls attribute")

    fsdp_blocks_cls = model.net.fsdp_wrap_block_cls
    if getattr(config, "gan_loss_weight_gen", 0) > 0:
        if not hasattr(model.discriminator, "fsdp_wrap_block_cls"):
            raise ValueError("model.discriminator does not have fsdp_wrap_block_cls attribute")
        fsdp_blocks_cls = [model.net.fsdp_wrap_block_cls, model.discriminator.fsdp_wrap_block_cls]

    fsdp_blocks_cls = list(fsdp_blocks_cls) if isinstance(fsdp_blocks_cls, (list, tuple, set)) else [fsdp_blocks_cls]
    log.critical(f"Using FSDP blocks {fsdp_blocks_cls}")
    return fsdp_blocks_cls


def get_wrap_policy(model: BaseDistillationModel) -> Callable:
    """Get the FSDP wrap policy based on the model config."""
    config = model.config

    log.critical(f"Using wrap policy {config.fsdp.policy}")
    if config.fsdp.policy == "size":
        min_num_params = getattr(config.fsdp, "min_num_params", 3000)
        log.critical(f"Using {min_num_params} as the minimum number of parameters for auto-wrap policy")
        wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        fsdp_blocks_cls = get_fsdp_wrap_block_cls(model)
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(fsdp_blocks_cls),
        )
    return wrap_policy
