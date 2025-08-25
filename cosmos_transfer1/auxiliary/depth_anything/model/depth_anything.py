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
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cosmos_transfer1.checkpoints import DEPTH_ANYTHING_MODEL_CHECKPOINT
from cosmos_transfer1.utils import log


class DepthAnythingModel:
    def __init__(self):
        """
        Initialize the Depth Anything model and its image processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load image processor and model with half precision
        log.info(f"Loading Depth Anything model - {DEPTH_ANYTHING_MODEL_CHECKPOINT}...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            DEPTH_ANYTHING_MODEL_CHECKPOINT,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_ANYTHING_MODEL_CHECKPOINT,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)

    def predict_depth(self, image: Image.Image) -> Image.Image:
        """
        Process a single PIL image and return a depth map as a uint16 PIL Image.
        """
        # Prepare inputs for the model
        inputs = self.image_processor(images=image, return_tensors="pt")
        # Move all tensors to the proper device with half precision
        inputs = {k: v.to(self.device, dtype=torch.float16) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate the predicted depth to the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # PIL image size is (width, height), interpolate expects (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Convert the output tensor to a numpy array and save as a depth image
        output = prediction.squeeze().cpu().numpy()
        depth_image = DepthAnythingModel.save_depth(output)
        return depth_image

    def __call__(self, input_video: str, output_video: str = "depth.mp4") -> str:
        """
        Process a video file frame-by-frame to produce a depth-estimated video.
        The output video is saved as an MP4 file.
        """

        log.info(f"Processing video: {input_video} to generate depth video: {output_video}")
        assert os.path.exists(input_video)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            log.info("Error: Cannot open video file.")
            return

        # Retrieve video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        depths = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR to RGB and then to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=torch.float16) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # For video processing, take the first output and interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth[0].unsqueeze(0).unsqueeze(0),
                size=(frame_height, frame_width),
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().numpy()
            depths += [depth]
        cap.release()

        depths = np.stack(depths)
        depths_normed = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8) * 255.0
        depths_normed = depths_normed.astype(np.uint8)

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        self.write_video(depths_normed, output_video, fps=fps)
        return output_video

    @staticmethod
    def save_depth(output: np.ndarray) -> Image.Image:
        """
        Convert the raw depth output (float values) into a uint16 PIL Image.
        """
        depth_min = output.min()
        depth_max = output.max()
        max_val = (2**16) - 1  # Maximum value for uint16

        if depth_max - depth_min > np.finfo("float").eps:
            out_array = max_val * (output - depth_min) / (depth_max - depth_min)
        else:
            out_array = np.zeros_like(output)

        formatted = out_array.astype("uint16")
        depth_image = Image.fromarray(formatted, mode="I;16")
        return depth_image

    @staticmethod
    def write_video(frames, output_path, fps=30):
        with imageio.get_writer(output_path, fps=fps, macro_block_size=8) as writer:
            for frame in frames:
                if len(frame.shape) == 2:  # single channel
                    frame = frame[:, :, None].repeat(3, axis=2)
                writer.append_data(frame)
