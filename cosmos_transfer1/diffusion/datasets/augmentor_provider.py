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

from cosmos_transfer1.diffusion.config.transfer.blurs import BlurAugmentorConfig, random_blur_config
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_AUG_KEYS, CTRL_HINT_KEYS, CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.datasets.augmentors.basic_augmentors import (
    ReflectionPadding,
    ResizeLargestSideAspectPreserving,
)
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import (
    VIDEO_RES_SIZE_INFO,
    AddControlInput,
    AddControlInputComb,
)
from cosmos_transfer1.diffusion.datasets.augmentors.merge_datadict import DataDictMerger
from cosmos_transfer1.diffusion.datasets.augmentors.text_transforms_for_video import TextTransformForVideo
from cosmos_transfer1.utils.lazy_config import LazyCall as L

AUGMENTOR_OPTIONS = {}


def augmentor_register(key):
    def decorator(func):
        AUGMENTOR_OPTIONS[key] = func
        return func

    return decorator


@augmentor_register("video_basic_augmentor")
def get_video_augmentor(
    resolution: str,
    blur_config=None,
):
    return {
        "merge_datadict": L(DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "frame_start",
                "frame_end",
                "orig_num_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
    }


"""
register all the video ctrlnet augmentors for data loading
"""
for hint_key in CTRL_HINT_KEYS:

    def get_video_ctrlnet_augmentor(hint_key, is_kd_data: bool = False, use_random: bool = True):
        def _get_video_ctrlnet_augmentor(
            resolution: str,
            blur_config: BlurAugmentorConfig = random_blur_config,
        ):
            if hint_key == "control_input_keypoint":
                add_control_input = L(AddControlInputComb)(
                    input_keys=["", "video"],
                    output_keys=[hint_key],
                    args={
                        "comb": CTRL_HINT_KEYS_COMB[hint_key],
                        "use_openpose_format": True,
                        "kpt_thr": 0.6,
                        "human_kpt_line_width": 4,
                    },
                    use_random=use_random,
                    blur_config=blur_config,
                )
            elif hint_key in CTRL_HINT_KEYS_COMB:
                add_control_input = L(AddControlInputComb)(
                    input_keys=["", "video"],
                    output_keys=[hint_key],
                    args={"comb": CTRL_HINT_KEYS_COMB[hint_key]},
                    use_random=use_random,
                    blur_config=blur_config,
                    is_kd_data=is_kd_data,
                )
            else:
                add_control_input = L(AddControlInput)(
                    input_keys=["", "video"],
                    output_keys=[hint_key],
                    use_random=use_random,
                    blur_config=blur_config,
                    is_kd_data=is_kd_data,
                )
            input_keys = ["video"]
            output_keys = [
                "video",
                "fps",
                "num_frames",
                "frame_start",
                "frame_end",
                "orig_num_frames",
            ]
            for key, value in CTRL_AUG_KEYS.items():
                if key in hint_key:
                    input_keys.append(value)
                    output_keys.append(value)

            augmentation = {
                # "merge_datadict": L(DataDictMerger)(
                #     input_keys=input_keys,
                #     output_keys=output_keys,
                # ),
                # this addes the control input tensor to the data dict
                "add_control_input": add_control_input,
                # this resizes both the video and the control input to the model's required input size
                "resize_largest_side_aspect_ratio_preserving": L(ResizeLargestSideAspectPreserving)(
                    input_keys=["video", hint_key],
                    args={"size": VIDEO_RES_SIZE_INFO[resolution]},
                ),
                "reflection_padding": L(ReflectionPadding)(
                    input_keys=["video", hint_key],
                    args={"size": VIDEO_RES_SIZE_INFO[resolution]},
                ),
                "text_transform": L(TextTransformForVideo)(
                    input_keys=["t5_text_embeddings"],
                    args={
                        "t5_tokens": {"num": 512, "dim": 1024},
                        "is_mask_all_ones": True,
                    },
                ),
            }
            return augmentation

        return _get_video_ctrlnet_augmentor

    augmentor_register(f"video_ctrlnet_augmentor_{hint_key}")(get_video_ctrlnet_augmentor(hint_key))
    augmentor_register(f"kd_video_ctrlnet_augmentor_{hint_key}")(get_video_ctrlnet_augmentor(hint_key, is_kd_data=True))
