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

from PIL import Image

from cosmos_transfer1.auxiliary.depth_anything.model.depth_anything import DepthAnythingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Estimation using Depth Anything V2")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image or video")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Processing mode: 'image' for a single image, 'video' for a video file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = DepthAnythingModel()

    if args.mode == "image":
        # Load the input image and predict its depth
        image = Image.open(args.input).convert("RGB")
        depth_image = model.predict_depth(image)
        depth_image.save(args.output)
        print(f"Depth image saved to {args.output}")
    elif args.mode == "video":
        # Process the video and save the output
        out_path = model.predict_depth_video(args.input, args.output)
        if out_path:
            print(f"Depth video saved to {out_path}")


if __name__ == "__main__":
    main()
