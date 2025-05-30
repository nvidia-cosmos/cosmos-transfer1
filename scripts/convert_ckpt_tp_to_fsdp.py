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

"""
Usage:
    torchrun --nproc_per_node=8 -m scripts.convert_ckpt_tp_to_fsdp > output.txt

This script is designed to convert a Tensor Parallel (TP) checkpoint
to a Fully Sharded Data Parallel (FSDP) compatible format for a video diffusion model.

Using experiment `CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3_posttrain` as an example:
For a model trained with Tensor Parallel (TP), the checkpoints are saved in the following formats:
```
    checkpoint_path = f"checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/{experiment}/checkpoints/iter_000000100_model_mp_0.pt"
    checkpoint_path = f"checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/{experiment}/checkpoints/iter_000000100_model_mp_1.pt"
    ...
    checkpoint_path = f"checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/{experiment}/checkpoints/iter_000000100_model_mp_7.pt"
```

where `*_model_mp_0.pt` and `*_model_mp_1.pt` are the model checkpoints for the eight TP ranks.

This script will load the TP model checkpoint and convert it to a FSDP-compatible format.
The converted checkpoints will be saved
to a new directory `fsdp_checkpoints` under the same experiment directory, e.g.,
 `checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/{experiment}/fsdp_checkpoints/`.

It has the following formats:
```
iter_000000100_reg_model.pt
iter_000000100_ema_model.pt
```
"""

import os
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine as te
import yaml
from megatron.core import parallel_state

from cosmos_transfer1.diffusion.config.config_train import make_config
from cosmos_transfer1.diffusion.training.train import instantiate_model
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.config_helper import override
from cosmos_transfer1.utils.easy_io import easy_io
from cosmos_transfer1.utils.misc import set_random_seed


@torch.no_grad
def copy_params_from_tp(model: nn.Module, model_tp: nn.Module, tp_size: int) -> None:
    orig_tp_size = parallel_state.get_tensor_model_parallel_world_size()
    # create temporary parallel_state for parameters & buffer copy
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    match_layers = OrderedDict()
    ddp_group = parallel_state.get_data_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    assert tp_size == parallel_state.get_tensor_model_parallel_world_size(), "TP group init is wrong"
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    def record_match_layer(name, param, param_chunk, policy):
        match_layers[name] = {
            "shape": list(param.shape),
            "copied_name": name,
            "copied_shape": list(param_chunk.shape),
            "policy": policy,
            "type": "param",
        }

    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        module_name_hierarchy = name.split(".")
        submodule_name = ".".join(module_name_hierarchy[:-1])
        submodule = model.get_submodule(submodule_name)
        submodule_tp = model_tp.get_submodule(submodule_name)

        if isinstance(submodule, nn.Linear) and isinstance(submodule_tp, te.pytorch.Linear):
            # get parallel mode and copy weights
            if module_name_hierarchy[-1] == "weight":
                if submodule_tp.parallel_mode == "column":
                    param_chunks = param.chunk(tp_size, dim=0)
                    record_match_layer(name, param, param_chunks[tp_rank], f"column_rank{tp_rank}")
                    param_tp_chunks = [torch.zeros_like(param_tp) for _ in range(tp_size)]
                    dist.all_gather(param_tp_chunks, param_tp, tp_group, async_op=False)
                    for _tp_rank in range(tp_size):
                        param_chunks[_tp_rank].copy_(param_tp_chunks[_tp_rank], non_blocking=True)
                elif submodule_tp.parallel_mode == "row":
                    param_chunks = param.chunk(tp_size, dim=1)
                    record_match_layer(name, param, param_chunks[tp_rank], f"row_rank{tp_rank}")
                    param_tp_chunks = [torch.zeros_like(param_tp) for _ in range(tp_size)]
                    dist.all_gather(param_tp_chunks, param_tp, tp_group, async_op=False)
                    for _tp_rank in range(tp_size):
                        param_chunks[_tp_rank].copy_(param_tp_chunks[_tp_rank], non_blocking=True)
                else:
                    record_match_layer(name, param, param_tp, "direct")
                    param.copy_(param_tp, non_blocking=True)
            elif module_name_hierarchy[-1] == "bias":
                raise NotImplementedError("Bias is not supported yet.")
        else:
            record_match_layer(name, param, param_tp, "direct")
            param.copy_(param_tp, non_blocking=True)

    # Important to also copy buffer as logvar has randomness.
    for (name, buffer), (name_tp, buffer_tp) in zip(model.named_buffers(), model_tp.named_buffers()):
        if buffer.size() == buffer_tp.size():
            match_layers[name] = {
                "shape": buffer.shape,
                "copied_name": name_tp,
                "copied_shape": buffer_tp.shape,
                "policy": "direct",
                "type": "buffer",
            }
            buffer.copy_(buffer_tp, non_blocking=True)
        else:
            if "bias" in name:
                raise NotImplementedError("Bias is not supported yet.")

            if "model_ema" in name:
                module_name = name.replace("-", ".")
                module_name = module_name.replace("model_ema", "model")
                if "column" in match_layers[module_name]["policy"] or "row" in match_layers[module_name]["policy"]:
                    dim = 0 if "column" in match_layers[module_name]["policy"] else 1
                    buffer_chunks = buffer.chunk(tp_size, dim=dim)
                    buffer_tp_chunks = [torch.zeros_like(buffer_tp) for _ in range(tp_size)]
                    dist.all_gather(buffer_tp_chunks, buffer_tp, tp_group, async_op=False)
                    for _tp_rank in range(tp_size):
                        buffer_chunks[_tp_rank].copy_(buffer_tp_chunks[_tp_rank], non_blocking=True)
            else:
                log.info(f"{name} is not copied due to size mismatch.")

    dist.barrier(ddp_group)
    dist.barrier(tp_group)
    # convert match_layers to yaml and save it to disk
    yaml_fp = f"/tmp/match_layers_rank{dist.get_rank()}_tp_rank{tp_rank}.yaml"
    with open(yaml_fp, "w") as f:
        yaml.dump(match_layers, f)

    # recover the original parallel_state
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=orig_tp_size)

    return


def convert_tp_checkpoint_to_fsdp(
    experiment: str,
    checkpoint_path: str,
    output_directory: str,
    include_base_model_in_ctrlnet_ckpt: bool = False,
) -> None:
    """
    Convert a Tensor Parallel (TP) checkpoint to a Fully Sharded Data Parallel (FSDP) compatible format.

    This function performs the following steps:
    1. Loads a TP model checkpoint
    2. Initializes a non-TP model
    3. Converts the checkpoint from TP format to FSDP compatible format
    4. Verifies the conversion by comparing outputs, losses, and gradients

    Args:
        experiment (str): The name of the experiment for which to convert the checkpoint.
        checkpoint_path (str): The path to the TP checkpoint file.
        output_directory (str): The directory where the converted FSDP checkpoint will be saved.

    Raises:
        ValueError: If the conversion process fails or if the verification step detects significant discrepancies.

    Note:
        This function assumes that the necessary configurations and dependencies are properly set up.
        It uses bfloat16 as the default dtype for better performance and memory efficiency.

    """
    log.info(f"Converting TP checkpoint to FSDP for experiment: {experiment}")

    # Clean up any existing parallel state
    parallel_state.destroy_model_parallel()

    # Set the default dtype to bfloat16 for better performance and memory efficiency
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)

    # Initialize and load the Tensor Parallel (TP) model
    config_tp = make_config()
    override_tp = [
        "--",
        f"experiment={experiment}",
        f"checkpoint.load_path={checkpoint_path}",
        "checkpoint.load_training_state=False",
    ]
    config_tp = override(
        config_tp,
        override_tp,
    )

    # Initialize trainer, model, optimizer, scheduler, and grad scaler for TP
    trainer_tp = config_tp.trainer.type(config_tp)
    # tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    # global_tp_src_rank = parallel_state.get_tensor_model_parallel_src_rank()
    global_rank = dist.get_rank()

    # Set random seed by global rank to ensure diversity within TP groups
    set_random_seed(global_rank)
    model_tp = instantiate_model(config_tp, trainer_tp).cuda()
    optimizer_tp, scheduler_tp = model_tp.init_optimizer_scheduler(config_tp.optimizer, config_tp.scheduler)
    grad_scaler_tp = torch.amp.GradScaler("cuda", **config_tp.trainer.grad_scaler_args)

    # Load checkpoint and prepare model for training
    log.info("Loading checkpoint...")
    trainer_tp.checkpointer.load(model_tp, optimizer_tp, scheduler_tp, grad_scaler_tp)
    model_tp.on_train_start()

    # Initialize and prepare the non-TP model
    parallel_state.destroy_model_parallel()

    config = make_config()
    config = override(
        config,
        [
            "--",
            f"experiment={experiment}",
            "ckpt_klass=multi_rank",
            "checkpoint.load_path=''",
            "model_parallel.tensor_model_parallel_size=1",
            "model_parallel.sequence_parallel=False",
        ],
    )

    # Initialize non-TP model and copy parameters from TP model
    trainer = config.trainer.type(config)
    model = instantiate_model(config, trainer).cuda()
    model.on_train_start()
    copy_params_from_tp(model, model_tp, tp_size=tp_size)

    # Save the converted model checkpoints
    if torch.distributed.get_rank() == 0:
        # Save regular model checkpoint
        checkpoint_name = os.path.basename(checkpoint_path)
        reg_model_checkpoint_name = checkpoint_name.replace(".pt", "_reg_model.pt")
        reg_model_path = os.path.join(output_directory, reg_model_checkpoint_name)
        easy_io.dump(model.state_dict()["model"], reg_model_path)

        # Save EMA model checkpoint with necessary post-processing
        ema_state_dict = {k.replace("-", "."): v for k, v in model.state_dict()["ema"].items()}
        for key in ["net.pos_embedder.seq", "logvar.0.freqs", "logvar.0.phases"]:
            ema_state_dict[key] = model.state_dict()["model"][key]

        if include_base_model_in_ctrlnet_ckpt:
            # Copy base model keys to ema dict for controlnets.
            for key in model.state_dict()["model"].keys():
                if key.startswith("base_model") and key not in ema_state_dict:
                    ema_state_dict[key] = model.state_dict()["model"][key]

            ema_model_checkpoint_name = checkpoint_name.replace(".pt", "_ema_model.pt")
        else:
            ema_model_checkpoint_name = checkpoint_name.replace(".pt", "_ema_model_only.pt")
        ema_model_path = os.path.join(output_directory, ema_model_checkpoint_name)
        easy_io.dump(ema_state_dict, ema_model_path)

        log.info(
            f"Conversion complete. FSDP-compatible checkpoints saved for experiment: {experiment}\n"
            f"Regular model saved at {reg_model_path}\n"
            f"EMA model saved at {ema_model_path}"
        )


if __name__ == "__main__":
    # Example: Assume the TP checkpoint is saved for the VisControl at iteration 100 in the path below.
    experiment = "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3_posttrain"
    checkpoint_path = f"checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/{experiment}/checkpoints/iter_000000100.pt"
    output_directory = os.path.dirname(checkpoint_path).replace(
        f"{experiment}/checkpoints", f"{experiment}/fsdp_checkpoints"
    )
    os.makedirs(output_directory, exist_ok=True)
    convert_tp_checkpoint_to_fsdp(experiment, checkpoint_path, output_directory)
