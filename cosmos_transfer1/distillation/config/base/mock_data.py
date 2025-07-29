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

from torch.utils.data import DataLoader

from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.distillation.datasets.mock_data_distill import (
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
