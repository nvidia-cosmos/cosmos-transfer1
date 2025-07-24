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

import copy

from cosmos_transfer1.utils.lazy_config import LazyDict

FULL_FSDP_CONFIG = LazyDict(
    dict(
        trainer=dict(
            distributed_parallelism="fsdp",
        ),
        model=dict(
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=False,
                min_num_params=1024,
                sharding_strategy="full",
            ),
        ),
    )
)

HYBRID_FSDP_CONFIG = copy.deepcopy(FULL_FSDP_CONFIG)
HYBRID_FSDP_CONFIG.model.fsdp.sharding_strategy = "hybrid"
