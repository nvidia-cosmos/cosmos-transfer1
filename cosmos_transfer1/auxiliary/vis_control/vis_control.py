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
import random

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_transfer1.diffusion.config.transfer.augmentors import BilateralOnlyBlurAugmentorConfig
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import Blur
from cosmos_transfer1.utils import log


class VisControlModel:
    def __init__(self, blur_strength="medium", use_random=True):
        self.use_random = use_random
        self.min_downup_factor = 4
        self.max_downup_factor = 16
        self.blur_kernel_size = 15
        self.blur_sigma = 7.0
        self.blur = Blur(config=BilateralOnlyBlurAugmentorConfig[blur_strength], output_key="vis_control")

    def _load_frame(self, video_path: str) -> np.ndarray:
        log.info(f"Processing video: {video_path} for vis control")
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
        log.info(f"VisControlModel: {input_video=} {output_video=}")
        # Resize the frames to target size before blurring.
        frames = self._load_frame(input_video)
        H, W = frames.shape[2], frames.shape[3]

        scale_factor = random.randint(self.min_downup_factor, self.max_downup_factor + 1)

        frames = self.blur(frames)
        # turn into tensor
        controlnet_img = torch.from_numpy(frames)
        # Resize image
        controlnet_img = transforms_F.resize(
            controlnet_img,
            size=(int(H / scale_factor), int(W / scale_factor)),
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )
        controlnet_img = transforms_F.resize(
            controlnet_img,
            size=(H, W),
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )

        # Save the output video
        self._save_output_video(controlnet_img, output_video)
        return controlnet_img

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

            log.info(f"Successfully saved vis control video to: {output_path}")
        except Exception as e:
            log.error(f"Failed to save vis control video to {output_path}: {e}")
            raise
