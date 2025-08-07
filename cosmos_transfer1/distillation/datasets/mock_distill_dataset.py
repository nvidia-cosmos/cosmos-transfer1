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

"""
Run this command to sanity check the dataset:
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/datasets/mock_distill_dataset.py
"""

from functools import partial

import torch

from cosmos_transfer1.distillation.datasets.mock_dataset import CombinedDictDataset, LambdaDataset


def get_mock_distill_dataset(
    h: int,
    w: int,
    num_video_frames: int,
    len_t5: int = 512,
    is_debug_tokenizer: bool = False,
):
    """
    Mock dataset for distillation model training.

    Args:
        h (int): height of the video
        w (int): width of the video
        num_video_frames (int): number of video frames
        len_t5 (int): size of t5 embedding
        is_debug_tokenizer (bool): whether debug tokenizer is used
    """

    def video_fn():
        return torch.randint(0, 255, size=(3, num_video_frames, h, w)).to(dtype=torch.uint8)

    if is_debug_tokenizer:
        noise_fn = partial(torch.randn, size=(16, num_video_frames // 17 * 3, h // 16, w // 16))
    else:
        noise_fn = partial(torch.randn, size=(16, (num_video_frames - 1) // 8 + 1, h // 8, w // 8))

    return CombinedDictDataset(
        **{
            "video": LambdaDataset(video_fn),
            # Noise tensor is used for KD training only
            "noise": LambdaDataset(noise_fn),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, 1024))),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 24.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: num_video_frames),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w))),
            "ai_caption": LambdaDataset(lambda: "placeholder"),
            "dataset_name": LambdaDataset(lambda: "video_data"),
            "frame_end": LambdaDataset(lambda: 0),
            "frame_start": LambdaDataset(lambda: 0),
        }
    )


def get_mock_distill_ctrlnet_dataset(
    h: int,
    w: int,
    num_video_frames: int,
    len_t5: int = 512,
    hint_key: str = "control_input_edge",
    is_debug_tokenizer: bool = False,
):
    """
    Mock dataset for ctrlnet distillation model training.

    Args:
        h (int): height of the video
        w (int): width of the video
        num_video_frames (int): number of video frames
        len_t5 (int): size of t5 embedding
        hint_key (str): control hint key
        is_debug_tokenizer (bool): whether debug tokenizer is used
    """

    def video_fn():
        return torch.randint(0, 255, size=(3, num_video_frames, h, w)).to(dtype=torch.uint8)

    def hint_fn():
        return torch.randint(0, 255, size=(3, num_video_frames, h, w)).to(dtype=torch.uint8)

    if is_debug_tokenizer:
        noise_fn = partial(torch.randn, size=(16, num_video_frames // 17 * 3, h // 16, w // 16))
    else:
        noise_fn = partial(torch.randn, size=(16, (num_video_frames - 1) // 8 + 1, h // 8, w // 8))

    return CombinedDictDataset(
        **{
            "video": LambdaDataset(video_fn),
            hint_key: LambdaDataset(hint_fn),
            # Noise tensor is used for KD training only
            "noise": LambdaDataset(noise_fn),
            "t5_text_embeddings": LambdaDataset(partial(torch.randn, size=(len_t5, 1024))),
            "t5_text_mask": LambdaDataset(partial(torch.randint, low=0, high=2, size=(len_t5,), dtype=torch.int64)),
            "fps": LambdaDataset(lambda: 24.0),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "num_frames": LambdaDataset(lambda: num_video_frames),
            "padding_mask": LambdaDataset(partial(torch.zeros, size=(1, h, w))),
            "ai_caption": LambdaDataset(lambda: "placeholder"),
            "dataset_name": LambdaDataset(lambda: "video_data"),
            "frame_end": LambdaDataset(lambda: 0),
            "frame_start": LambdaDataset(lambda: 0),
        }
    )


if __name__ == "__main__":
    """
    Sanity check for the dataset.
    """
    control_input_key = "control_input_edge"
    dataset = get_mock_distill_ctrlnet_dataset(
        h=704,
        w=1280,
        num_video_frames=121,
        hint_key=control_input_key,
    )
    indices = [0, 5, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data[control_input_key].shape=}\n"  # should match the video shape
                f"{data['noise'].shape=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                "---"
            )
        )
