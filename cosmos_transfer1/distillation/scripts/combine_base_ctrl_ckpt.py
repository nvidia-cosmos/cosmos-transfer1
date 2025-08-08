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
Script to combine base model and control model checkpoints into a single checkpoint.

Use this script to prepare the teacher model checkpoint for distillation.

Below is an example of the expected checkpoint structure (inside the checkpoints/nvidia/Cosmos-Transfer1-7B/ directory):
    - base_model.pt: The base model checkpoint.
    - edge_control.pt: The control model checkpoint for edge control.
    - checkpoints_teacher/edge_control.pt: The combined checkpoint for distillation.

Usage:
    PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/scripts/combine_base_ctrl_ckpt.py --ctrl_type edge
"""

import argparse
from pathlib import Path

import torch


def combine_checkpoints(base_model_path: str, control_model_path: str, output_path: str):
    """
    Combine base model and control model checkpoints into a single checkpoint.

    Args:
        base_model_path: Path to the base model checkpoint
        control_model_path: Path to the control model checkpoint
        output_path: Path where to save the combined checkpoint
    """
    print(f"Loading base model from: {base_model_path}")
    base_state_dict = torch.load(base_model_path, map_location="cpu", weights_only=False)

    print(f"Loading control model from: {control_model_path}")
    control_state_dict = torch.load(control_model_path, map_location="cpu", weights_only=False)

    print("Merging base and control model weights...")
    combined_state_dict = {**base_state_dict, **control_state_dict}
    print(f"Added {len(base_state_dict)} base model parameters")
    print(f"Added {len(control_state_dict)} control model parameters")

    print(f"Saving combined checkpoint to: {output_path}")
    torch.save(combined_state_dict, output_path)

    print("✅ Successfully combined checkpoints!")
    print(f"  Total parameters: {len(combined_state_dict)}")

    # Print some example keys for verification
    print("\nExample combined keys:")
    sample_keys = list(combined_state_dict.keys())[:5]
    for key in sample_keys:
        print(f"  {key}")
    if len(combined_state_dict) > 5:
        print(f"  ... and {len(combined_state_dict) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="Combine base model and control model checkpoints")
    parser.add_argument("--ctrl_type", required=True, help="Control type: edge, vis, depth, seg, keypoint")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Path to the checkpoint directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    base_model_path = f"{args.checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/base_model.pt"
    ctrl_model_path = f"{args.checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/{args.ctrl_type}_control.pt"
    output_path = f"{args.checkpoint_dir}/nvidia/Cosmos-Transfer1-7B/checkpoints_teacher/{args.ctrl_type}_control.pt"

    # Validate input files exist
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"Base model checkpoint not found: {base_model_path}")

    if not Path(ctrl_model_path).exists():
        raise FileNotFoundError(f"Control model checkpoint not found: {ctrl_model_path}")

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        combine_checkpoints(base_model_path, ctrl_model_path, output_path)
    except Exception as e:
        print(f"❌ Error combining checkpoints: {e}")
        raise


if __name__ == "__main__":
    main()
