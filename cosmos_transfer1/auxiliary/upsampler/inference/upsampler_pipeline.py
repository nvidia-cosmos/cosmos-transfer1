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

import argparse

from cosmos_transfer1.auxiliary.upsampler.model.upsampler import PixtralPromptUpsampler
from cosmos_transfer1.utils.misc import extract_video_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt upsampler pipeline")
    parser.add_argument("--prompt", type=str, required=False, help="Prompt to upsample")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--offload_prompt_upsampler", action="store_true", help="Offload prompt upsampler model after inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = PixtralPromptUpsampler(args.checkpoint_dir, offload_prompt_upsampler=args.offload_prompt_upsampler)

    # Upsample the prompt with the given video
    frame_paths = extract_video_frames(args.input_video)
    upsampled_prompt = model._prompt_upsample_with_offload(args.prompt, frame_paths)
    print("Upsampled prompt:", upsampled_prompt)


if __name__ == "__main__":
    import os

    rank = int(os.environ["RANK"])

    dist_keys = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "OMP_NUM_THREADS",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCHELASTIC_ERROR_FILE",
    ]

    for dist_key in dist_keys:
        del os.environ[dist_key]

    if rank == 0:
        main()
