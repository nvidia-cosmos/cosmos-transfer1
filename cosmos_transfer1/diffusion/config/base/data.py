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

from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS
from cosmos_transfer1.diffusion.datasets.example_transfer_dataset import ExampleTransferDataset
from cosmos_transfer1.utils.lazy_config import LazyCall as L


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_example_transfer_dataset(hint_key, is_train=True):
    dataset = L(ExampleTransferDataset)(
        dataset_dir="datasets/hdvila",
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


#  NOTE 1: For customized post train: add your dataloader registration here.
#  NOTE 2: The loop below simply registers a dataset for all hint_keys in CTRL_HINT_KEYS. The actual data might not exist.
def register_data_ctrlnet(cs):
    for hint_key in CTRL_HINT_KEYS:
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"example_transfer_train_data_{hint_key}",
            node=get_example_transfer_dataset(hint_key=hint_key, is_train=True),
        )
        cs.store(
            group="data_val",
            package="dataloader_val",
            name=f"example_transfer_val_data_{hint_key}",
            node=get_example_transfer_dataset(hint_key=hint_key, is_train=False),
        )
