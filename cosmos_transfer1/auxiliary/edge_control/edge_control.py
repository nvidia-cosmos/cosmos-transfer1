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

import os

import cv2
import imageio
import numpy as np
import torch

from cosmos_transfer1.utils import log


class EdgeControlModel:
    def __init__(self, canny_threshold="medium", use_random=True):
        self.use_random = use_random
        self.preset_strength = canny_threshold

    def _load_frame(self, video_path: str) -> np.ndarray:
        log.info(f"Processing video: {video_path} for edge control")
        assert os.path.exists(video_path)
        try:
            reader = imageio.get_reader(video_path, "ffmpeg")
            frames = np.array([frame for frame in reader])
            reader.close()
        except Exception as e:
            raise ValueError(f"Failed to load video frames from {video_path}") from e

        # Convert from (T, H, W, C) to (C, T, H, W) format
        frames = frames.transpose((3, 0, 1, 2))
        return frames

    def __call__(self, input_video, output_video):
        log.info(f"EdgeControlModel: {input_video=} {output_video=}")
        # Resize the frames to target size before blurring.
        frames = self._load_frame(input_video)

        if self.use_random:
            t_lower = np.random.randint(20, 100)  # Get a random lower thre within [0, 255]
            t_diff = np.random.randint(50, 150)  # Get a random diff between lower and upper
            t_upper = min(255, t_lower + t_diff)  # The upper thre is lower added by the diff
        else:
            if self.preset_strength == "none" or self.preset_strength == "very_low":
                t_lower, t_upper = 20, 50
            elif self.preset_strength == "low":
                t_lower, t_upper = 50, 100
            elif self.preset_strength == "medium":
                t_lower, t_upper = 100, 200
            elif self.preset_strength == "high":
                t_lower, t_upper = 200, 300
            elif self.preset_strength == "very_high":
                t_lower, t_upper = 300, 400
            else:
                raise ValueError(f"Preset {self.preset_strength} not recognized.")
        frames = np.array(frames)
        # Simple resize for edge detection - keep original resolution
        # frames = resize_frames(frames, is_image=False, data_dict={})  # Commented out as not needed

        # Compute the canny edge map by the two thresholds.
        edge_maps = [cv2.Canny(img, t_lower, t_upper) for img in frames.transpose((1, 2, 3, 0))]
        edge_maps = np.stack(edge_maps)[None]
        edge_maps = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)

        # Save the output video
        self._save_output_video(edge_maps, output_video)
        return edge_maps

    def _save_output_video(self, frames: torch.Tensor, output_path: str) -> None:
        """Save processed frames as a video file."""
        # Convert tensor to numpy and change format from (C, T, H, W) to (T, H, W, C)
        if isinstance(frames, torch.Tensor):
            frames_np = frames.detach().cpu().numpy()
        else:
            frames_np = frames

        # Transpose from (C, T, H, W) to (T, H, W, C)
        frames_np = frames_np.transpose((1, 2, 3, 0))

        # Ensure values are in the correct range [0, 255] and convert to uint8
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

        # Ensure RGB format (drop alpha channel if present)
        if frames_np.shape[-1] == 4:
            frames_np = frames_np[:, :, :, :3]
        elif frames_np.shape[-1] == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)

        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with imageio.get_writer(output_path, fps=30, macro_block_size=8) as writer:
                for frame in frames_np:
                    writer.append_data(frame)

            log.info(f"Successfully saved edge control video to: {output_path}")
        except Exception as e:
            log.error(f"Failed to save edge control video to {output_path}: {e}")
            raise
