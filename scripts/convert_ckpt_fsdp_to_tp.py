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
Converting an FSDP checkpoint to a TP checkpoint.
"""
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.easy_io import easy_io

TP_SIZE = 8


def is_column(key: str) -> bool:
    """Check if the given key corresponds to a column-parallel parameter."""
    return (
        key.endswith("to_q.0.weight")
        or key.endswith("to_k.0.weight")
        or key.endswith("to_v.0.weight")
        or key.endswith("block.layer1.weight")
    )


def is_row(key: str) -> bool:
    """Check if the given key corresponds to a row-parallel parameter."""
    return key.endswith("to_out.0.weight") or key.endswith("block.layer2.weight")


def native_to_tp(reg_state_dict: Dict[str, Any], tp_size: int) -> List[OrderedDict]:
    """Convert a regular state dict to tensor parallel state dicts.

    Args:
        reg_state_dict: The regular state dictionary.
        tp_size: The number of tensor parallel partitions.

    Returns:
        A list of OrderedDicts, each representing a tensor parallel partition.
    """
    tp_state_dict = [OrderedDict() for _ in range(tp_size)]
    log.info("Converting to TP checkpoint..")
    for key, value in reg_state_dict.items():
        if key.endswith("_extra_state"):
            continue

        if is_column(key):
            for i, item in enumerate(value.chunk(tp_size, dim=0)):
                tp_state_dict[i][key] = item
        elif is_row(key):
            for i, item in enumerate(value.chunk(tp_size, dim=1)):
                tp_state_dict[i][key] = item
        else:
            for i in range(tp_size):
                tp_state_dict[i][key] = value

    return tp_state_dict


def convert_fsdp_to_tp(path_in: str, path_out: str) -> None:
    """Convert an FSDP checkpoint to TP format.

    Args:
        path_in: Path to input checkpoint (without _reg_model.pt suffix)
        path_out: Path for output checkpoint (without _model_mp_X.pt suffix)
        tp_size: Number of tensor parallel partitions
        verbose: Whether to show progress bar

    Raises:
        FileNotFoundError: If input checkpoint doesn't exist
        ValueError: If paths are invalid or tp_size <= 0
        RuntimeError: For other conversion errors
    """
    try:
        log.info(f"Loading checkpoint from {path_in}..")
        native_ckpt = torch.load(
            path_in,
            map_location=torch.device("cpu"),
            weights_only=False,  # Load to CPU first; weights_only=False required for newer PyTorch versions
        )
        state_dicts = native_to_tp(native_ckpt, TP_SIZE)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file {path_in} not found")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")

    log.info("Saving TP checkpoints..")
    # Add a dummy grad_scaler and iteration to the checkpoint. Required by the training script.
    easy_io.dump({"grad_scaler": {}, "iteration": 0}, f"{path_out}.pt")
    for i in tqdm(range(TP_SIZE)):
        state_dict = {"model": state_dicts[i]}
        easy_io.dump(state_dict, f"{path_out}_model_mp_{i}.pt")


if __name__ == "__main__":
    """
    Example usage: converting a viscontrol model to a TP checkpoint.

    Command:
        python convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B/vis_control.pt

    This will save the Tensor Parallel (TP) checkpoints as 8 files in the same directory:
        checkpoints/nvidia/Cosmos-Transfer1-7B/vis_control_model_mp_0.pt
        ...
        checkpoints/nvidia/Cosmos-Transfer1-7B/vis_control_model_mp_7.pt
    """
    if len(sys.argv) != 2:
        print("Usage: python convert_ckpt_fsdp_to_tp.py <path_to_checkpoint.pt>")
        print("Example: python convert_ckpt_fsdp_to_tp.py checkpoints/model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Create checkpoints_tp directory in the same parent directory as the input checkpoint
    input_dir = os.path.dirname(checkpoint_path)
    tp_ckpt_dir = os.path.join(input_dir, "checkpoints_tp")
    os.makedirs(tp_ckpt_dir, exist_ok=True)

    # Use the same basename as input but in the checkpoints_tp directory
    out_tp_checkpoint_path = os.path.join(tp_ckpt_dir, os.path.basename(checkpoint_path).replace(".pt", ""))
    try:
        convert_fsdp_to_tp(checkpoint_path, out_tp_checkpoint_path)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)
