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

from typing import Dict

from cosmos_transfer1.checkpointer.fast_tp import Checkpointer as FastTPCheckpointer
from cosmos_transfer1.checkpointer.fsdp_checkpointer import FSDPCheckpointer
from cosmos_transfer1.checkpointer.multi_rank_checkpointer import MultiRankCheckpointer
from cosmos_transfer1.checkpointer.tp_checkpointer import Checkpointer as TPCheckpointer
from cosmos_transfer1.utils.lazy_config import LazyCall as L

MULTI_RANK_CHECKPOINTER: Dict[str, str] = L(MultiRankCheckpointer)()
FSDP_CHECKPOINTER: Dict[str, str] = L(FSDPCheckpointer)()
MODEL_PARALLEL_CHECKPOINTER: Dict[str, str] = L(TPCheckpointer)()
FAST_TP_CHECKPOINTER: Dict[str, str] = L(FastTPCheckpointer)()
