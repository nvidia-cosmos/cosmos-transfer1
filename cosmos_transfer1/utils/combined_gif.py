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
import math
import os
import re
from glob import glob
from typing import Tuple

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from cosmos_transfer1.utils import log


def extract_number(filename):
    # Extracts the last or first group of digits in the filename (before extension)
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # Try to find a number at the end or start
    match = re.search(r"(\d+)(?!.*\d)", name)  # last number
    if not match:
        match = re.search(r"^(\d+)", name)  # first number
    return int(match.group(1)) if match else float("inf")


def get_video_files(folder: str):
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    files = [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(video_exts)]
    # Sort by number if present, else lexicographically
    files.sort(key=lambda x: (extract_number(x), x))
    return files


def get_best_grid(n: int) -> Tuple[int, int]:
    # Find the grid (rows, cols) closest to square for n videos
    best_r, best_c = 1, n
    min_diff = n
    for r in range(1, n + 1):
        c = math.ceil(n / r)
        if r * c >= n:
            diff = abs(r - c)
            if diff < min_diff:
                min_diff = diff
                best_r, best_c = r, c
    return best_r, best_c


def read_video_frames(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames), fps


def resize_frames(frames, size):
    return np.array([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])


def tile_frames(frames_list, grid_shape, tile_size):
    # frames_list: list of [num_frames, H, W, C]
    num_frames = frames_list[0].shape[0]
    rows, cols = grid_shape
    H, W = tile_size
    tiled_frames = []
    for i in tqdm(range(num_frames), desc="Tiling frames"):
        row_imgs = []
        for r in range(rows):
            col_imgs = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(frames_list):
                    col_imgs.append(frames_list[idx][i])
                else:
                    col_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
            row_imgs.append(np.concatenate(col_imgs, axis=1))
        tiled = np.concatenate(row_imgs, axis=0)
        tiled_frames.append(tiled)
    return tiled_frames


def create_gif(input_folder, output_gif, output_res=(1080, 720), quality=10):
    video_files = get_video_files(input_folder)
    if not video_files:
        log.error("No videos found in the folder.")
        return
    log.info(f"Found {len(video_files)} videos.")

    # Read all videos
    all_frames = []
    min_frames = float("inf")
    for vf in tqdm(video_files, desc="Reading videos"):
        frames, fps = read_video_frames(vf)
        all_frames.append(frames)
        min_frames = min(min_frames, len(frames))
    # Truncate all to min_frames
    all_frames = [f[:min_frames] for f in all_frames]
    # Check aspect ratios
    h0, w0 = all_frames[0][0].shape[:2]
    for f in all_frames:
        h, w = f[0].shape[:2]
        if abs((w / h) - (w0 / h0)) > 1e-2:
            raise ValueError("All videos must have the same aspect ratio.")
    # Determine grid
    rows, cols = get_best_grid(len(all_frames))
    out_w, out_h = output_res
    tile_w, tile_h = out_w // cols, out_h // rows
    # Resize all frames
    all_frames = [resize_frames(f, (tile_w, tile_h)) for f in all_frames]
    # Tile frames
    tiled_frames = tile_frames(all_frames, (rows, cols), (tile_h, tile_w))
    # Save as GIF
    with imageio.get_writer(
        output_gif, mode="I", fps=int(fps), loop=0, quantizer="median_cut", quality=quality
    ) as writer:
        for frame in tqdm(tiled_frames, desc="Writing frames to GIF"):
            writer.append_data(frame)

    log.info(f"Saved tiled GIF to {output_gif}")


def main():
    parser = argparse.ArgumentParser(description="Tile videos into a GIF.")
    parser.add_argument("--input_folder", type=str, help="Folder containing videos")
    parser.add_argument("--output_gif", type=str, help="Output GIF path")
    parser.add_argument("--res", type=int, nargs=2, default=[1080, 720], help="Output resolution (width height)")
    parser.add_argument("--quality", type=int, default=10, help="GIF quality (1-100)")
    args = parser.parse_args()
    create_gif(args.input_folder, args.output_gif, tuple(args.res), args.quality)


if __name__ == "__main__":
    main()
