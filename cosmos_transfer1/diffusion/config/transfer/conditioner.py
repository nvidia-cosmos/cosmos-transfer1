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

from cosmos_transfer1.diffusion.conditioner import VideoConditionerWithCtrl
from cosmos_transfer1.diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import (
    AddControlInput,
    AddControlInputDepth,
    AddControlInputEdge,
    AddControlInputHDMAP,
    AddControlInputLIDAR,
    AddControlInputKeypoint,
    AddControlInputSeg,
    AddControlInputUpscale,
)
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

CTRL_HINT_KEYS = [
    "control_input_edge",
    "control_input_vis",
    "control_input_depth",
    "control_input_seg",
    "control_input_keypoint",
    "control_input_upscale",
    "control_input_hdmap",
    "control_input_lidar",
]

CTRL_HINT_KEYS_COMB = {
    "control_input_vis": [AddControlInput],
    "control_input_edge": [AddControlInputEdge],
    "control_input_depth": [AddControlInputDepth],
    "control_input_seg": [AddControlInputSeg],
    "control_input_keypoint": [AddControlInputKeypoint],
    "control_input_upscale": [AddControlInputUpscale],
    "control_input_hdmap": [AddControlInputHDMAP],
    "control_input_lidar": [AddControlInputLIDAR],
}


BaseVideoConditionerWithCtrlConfig: LazyDict = L(VideoConditionerWithCtrl)(
    text=TextConfig(),
)

VideoConditionerFpsSizePaddingWithCtrlConfig: LazyDict = L(VideoConditionerWithCtrl)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)
