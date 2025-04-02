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
import numpy as np
from rtmlib import Wholebody

from cosmos_transfer1.utils import log
from cosmos_transfer1.diffusion.datasets.augmentors.human_keypoint_utils import (
    coco_wholebody_133_skeleton,
    openpose134_skeleton,
)

class HumanKeypointModel:
    def __init__(self, to_openpose=True, conf_thres=0.6):
        self.model = Wholebody(
            to_openpose=to_openpose,
            mode="performance",
            backend="onnxruntime",
            device="cuda",
        )
        self.to_openpose = to_openpose
        self.conf_thres = conf_thres

    def __call__(self, input_video: str, output_video: str = "keypoint.mp4") -> str:
        """
            Generate the human body keypoint plot for the keypointControlNet video2world model.
            Input: mp4 video
            Output: mp4 keypoint video, of the same spatial and temporal dimensions as the input video.
        """

        log.info(f"Processing video: {input_video} to generate keypoint video: {output_video}")
        assert os.path.exists(input_video)

        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        # vid writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        skeleton_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        log.info(f"frame width: {frame_width}, frame height: {frame_height}, fps: {fps}")
        log.info("start pose estimation for frames..")

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create a black background frame
            black_frame = np.zeros_like(frame)

            # Run pose estimation
            keypoints, scores = self.model(frame)

            if keypoints is not None and len(keypoints) > 0:
                skeleton_frame = self.plot_person_kpts(
                    black_frame,
                    keypoints,
                    scores,
                    kpt_thr=self.conf_thres,
                    openpose_format=True,
                    line_width=4,
                )  # (h, w, 3)
            else:
                skeleton_frame = black_frame

            skeleton_writer.write(skeleton_frame[:, :, ::-1])

        cap.release()
        skeleton_writer.release()

    def draw_skeleton(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.6,
        openpose_format: bool = True,
        radius: int = 2,
        line_width: int = 4,
    ):
        skeleton_topology = openpose134_skeleton if openpose_format else coco_wholebody_133_skeleton
        assert len(keypoints.shape) == 2
        keypoint_info, skeleton_info = (
            skeleton_topology["keypoint_info"],
            skeleton_topology["skeleton_info"],
        )
        vis_kpt = [s >= kpt_thr for s in scores]
        link_dict = {}
        for i, kpt_info in keypoint_info.items():
            kpt_color = tuple(kpt_info["color"])
            link_dict[kpt_info["name"]] = kpt_info["id"]

            kpt = keypoints[i]

            if vis_kpt[i]:
                img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)

        for i, ske_info in skeleton_info.items():
            link = ske_info["link"]
            pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

            if vis_kpt[pt0] and vis_kpt[pt1]:
                link_color = ske_info["color"]
                kpt0 = keypoints[pt0]
                kpt1 = keypoints[pt1]

                img = cv2.line(
                    img, (int(kpt0[0]), int(kpt0[1])), (int(kpt1[0]), int(kpt1[1])), link_color, thickness=line_width
                )

        return img

    def plot_person_kpts(
        self,
        pose_vis_img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.6,
        openpose_format: bool = True,
        line_width: int = 4,
    ) -> np.ndarray:
        """
        plot a single person
        in-place update the pose image
        """
        for kpts, ss in zip(keypoints, scores):
            try:
                pose_vis_img = self.draw_skeleton(
                    pose_vis_img, kpts, ss, kpt_thr=kpt_thr, openpose_format=openpose_format, line_width=line_width
                )
            except ValueError as e:
                log.error(f"Error in draw_skeleton func, {e}")

        return pose_vis_img
