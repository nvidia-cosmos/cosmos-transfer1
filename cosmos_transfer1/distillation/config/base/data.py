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

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.distillation.datasets.example_kd_dataset import KDTransferDataset
from cosmos_transfer1.distillation.datasets.mock_distill_dataset import (
    get_mock_distill_ctrlnet_dataset,
    get_mock_distill_dataset,
)

MOCK_DISTILL_DATA_LOADER = L(DataLoader)(
    dataset=L(get_mock_distill_dataset)(
        h=704,
        w=1280,
        num_video_frames=121,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# Mock dataloader for debugging purposes, works with debug tokenizer
MOCK_DISTILL_DATA_LOADER_DEBUG = L(DataLoader)(
    dataset=L(get_mock_distill_dataset)(
        h=704,
        w=1280,
        num_video_frames=136,
        is_debug_tokenizer=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

MOCK_DISTILL_CTRLNET_DATA_LOADER = L(DataLoader)(
    dataset=L(get_mock_distill_ctrlnet_dataset)(
        h=704,
        w=1280,
        num_video_frames=121,
        hint_key="control_input_edge",
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# Mock dataloader for debugging purposes, works with debug tokenizer
MOCK_DISTILL_CTRLNET_DATA_LOADER_DEBUG = L(DataLoader)(
    dataset=L(get_mock_distill_ctrlnet_dataset)(
        h=704,
        w=1280,
        num_video_frames=136,
        hint_key="control_input_edge",
        is_debug_tokenizer=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_kd_transfer_dataset(hint_key, is_train=True):
    dataset = L(KDTransferDataset)(
        dataset_dir="datasets/kd",
        num_frames=121,
        resolution="720",
        hint_key=hint_key,
        is_train=is_train,
    )

    return L(DataLoader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=1,
        drop_last=True,
        num_workers=8,  # adjust as needed
        prefetch_factor=2,  # adjust as needed
        pin_memory=True,
    )
