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
import os

import torch

from cosmos_transfer1.diffusion.datasets.augmentors.text_transforms_for_video import pad_and_resize
from cosmos_transfer1.utils.easy_io import easy_io
from scripts.get_t5_embeddings import encode_for_batch, init_t5

"""
Script to generate T5 embeddings for a given text prompt and save as .pkl file.

Example usage:
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/scripts/save_negative_prompt.py
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T5 embedding for text and save as .pkl file")
    parser.add_argument(
        "--text",
        type=str,
        # Recommended negative prompt for Cosmos-Transfer1
        default=(
            "The video captures a game playing, with bad crappy graphics and cartoonish frames. "
            "It represents a recording of old outdated games. The lighting looks very fake. "
            "The textures are very raw and basic. The geometries are very primitive. "
            "The images are very pixelated and of poor CG quality. "
            "There are many subtitles in the footage. "
            "Overall, the video is unrealistic at all."
        ),
        help="Input text to encode",
    )
    parser.add_argument(
        "--output_path", type=str, default="datasets/negative_prompt/transfer1.pkl", help="Output .pkl file path"
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument("--t5_model_path", type=str, default="checkpoints/google-t5/t5-11b", help="T5 model local path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Encoding text: '{args.text}'")
    print(f"Output file: {args.output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize T5 model using the existing function
    print("Loading T5 model...")
    tokenizer, text_encoder = init_t5(
        pretrained_model_name_or_path=args.t5_model_path,
        max_length=args.max_length,
    )

    # Encode the text using the existing function
    print("Encoding text...")
    encoded_text = encode_for_batch(tokenizer, text_encoder, [args.text], args.max_length)

    # Extract the single embedding from the batch result
    embedding, _ = pad_and_resize(encoded_text[0], 512)
    embedding = embedding.to(torch.bfloat16)
    to_save = {"t5_text_embeddings": embedding}

    # Save embedding as pickle file
    print(f"Saving embedding to {args.output_path}")
    easy_io.dump(to_save, args.output_path)
    print(f"Successfully saved embedding with shape {embedding.shape} to {args.output_path}")


if __name__ == "__main__":
    main()
