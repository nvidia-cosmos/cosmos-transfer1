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

""" Download checkpoints from HuggingFace Hub.

This file downloads the checkpoints specified in the `cosmos_transfer1.checkpoints` module.

Usage:

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir <output_dir> --hf_token <hf_token>
    If the token is not provided, it will try to read from the `HF_TOKEN` environment variable.

"""

import argparse
import hashlib
import os
import pathlib

from huggingface_hub import login, snapshot_download

# Import the checkpoint paths
from cosmos_transfer1 import checkpoints
from cosmos_transfer1.utils import log


def download_checkpoint(checkpoint: str, output_dir: str) -> None:
    """Download a single checkpoint from HuggingFace Hub."""
    try:
        # Parse the checkpoint path to get repo_id and filename
        checkpoint, revision = checkpoint.split(":") if ":" in checkpoint else (checkpoint, None)
        checkpoint_dir = os.path.join(output_dir, checkpoint)
        if get_md5_checksum(output_dir, checkpoint):
            log.warning(f"Checkpoint {checkpoint_dir} EXISTS, skipping download... ")
            return
        else:
            print(f"Downloading {checkpoint} to {checkpoint_dir}")
        # Create the output directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Downloading {checkpoint}...")
        # Download the files
        snapshot_download(repo_id=checkpoint, local_dir=checkpoint_dir, revision=revision)
        print(f"Successfully downloaded {checkpoint}")

    except Exception as e:
        print(f"Error downloading {checkpoint}: {str(e)}")


MD5_CHECKSUM_LOOKUP = {
    f"{checkpoints.GROUNDING_DINO_MODEL_CHECKPOINT}/pytorch_model.bin": "0fcf0d965ca9baec14bb1607005e2512",
    f"{checkpoints.GROUNDING_DINO_MODEL_CHECKPOINT}/model.safetensors": "0739b040bb51f92464b4cd37f23405f9",
    f"{checkpoints.T5_MODEL_CHECKPOINT}/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    f"{checkpoints.T5_MODEL_CHECKPOINT}/tf_model.h5": "e081fc8bd5de5a6a9540568241ab8973",
    f"{checkpoints.SAM2_MODEL_CHECKPOINT}/sam2_hiera_large.pt": "08083462423be3260cd6a5eef94dc01c",
    f"{checkpoints.DEPTH_ANYTHING_MODEL_CHECKPOINT}/model.safetensors": "14e97d7ed2146d548c873623cdc965de",
    checkpoints.BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: "2006e158f8a17a3b801c661f0c01e9f2",
    checkpoints.HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "2ddd781560d221418c2ed9258b6ca829",
    checkpoints.LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "184beee5414bcb6c0c5c0f09d8f8b481",
    checkpoints.UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: "b28378d13f323b49445dc469dfbbc317",
    checkpoints.BASE_7B_CHECKPOINT_PATH: "356497b415f3b0697f8bb034d22b6807",
    checkpoints.VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "69fdffc5006bc5d6acb29449bb3ffdca",
    checkpoints.EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "a0642e300e9e184077d875e1b5920a61",
    checkpoints.EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH: "bf7def0bb5ffb5beff1f376d2404fcb4",
    checkpoints.DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "80999ed60d89a8dfee785c544e0ccd54",
    checkpoints.SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "3e4077a80c836bf102c7b2ac2cd5da8c",
    checkpoints.KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "26619fb1686cff0e69606a9c97cac68e",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit": "7f658580d5cf617ee1a1da85b1f51f0d",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit": "ff21a63ed817ffdbe4b6841111ec79a8",
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit": "f5834d03645c379bc0f8ad14b9bc0299",
    f"{checkpoints.COSMOS_UPSAMPLER_CHECKPOINT}/consolidated.safetensors": "d06e6366e003126dcb351ce9b8bf3701",
    f"{checkpoints.COSMOS_GUARDRAIL_CHECKPOINT}/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    f"{checkpoints.LLAMA_GUARD_3_MODEL_CHECKPOINT}/model-00001-of-00004.safetensors": "5748060ae47b335dc19263060c921a54",
    checkpoints.SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "4f8a4340d48ebedaa9e7bab772e0203d",
    checkpoints.SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "89b82db1bc1dc859178154f88b6ca0f2",
    checkpoints.SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "a9592d232a7e5f7971f39918c18eaae0",
    checkpoints.SV2MV_v2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "cb27af88ec7fb425faec32f4734d99cf",
    checkpoints.BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: "a3fb13e8418d8bb366b58e4092bd91df",
    checkpoints.BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: "48b2080ca5be66c05fac44dea4989a04",
}


def get_md5_checksum(output_dir, model_name):
    print("---------------------")
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name):
            print(f"Verifying checkpoint {key}...")
            file_path = os.path.join(output_dir, key)
            # File must exist
            if not pathlib.Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match give MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--output_dir", type=str, help="Directory to store the downloaded checkpoints", default="./checkpoints"
    )
    parser.add_argument(
        "--model", type=str, help="Model type to download", default="all", choices=["all", "7b", "7b_av"]
    )
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    checkpoint_vars = []
    # Get all variables from the checkpoints module
    for name in dir(checkpoints):
        obj = getattr(checkpoints, name)
        if isinstance(obj, str) and "CHECKPOINT" in name and "PATH" not in name:
            if args.model != "all" and name in [
                "COSMOS_TRANSFER1_7B_CHECKPOINT",
                "COSMOS_TRANSFER1_7B_SAMPLE_AV_CHECKPOINT",
            ]:
                if args.model == "7b" and name == "COSMOS_TRANSFER1_7B_CHECKPOINT":
                    checkpoint_vars.append(obj)
                elif args.model == "7b_av" and name in [
                    "COSMOS_TRANSFER1_7B_SAMPLE_AV_CHECKPOINT",
                    "COSMOS_TRANSFER1_7B_MV_SAMPLE_AV_CHECKPOINT",
                ]:
                    checkpoint_vars.append(obj)
            else:
                checkpoint_vars.append(obj)

    print(f"Found {len(checkpoint_vars)} checkpoints to download")

    # Download each checkpoint
    for checkpoint in checkpoint_vars:
        download_checkpoint(checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
