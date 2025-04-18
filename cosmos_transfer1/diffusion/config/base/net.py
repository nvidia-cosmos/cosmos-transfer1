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

from cosmos_transfer1.diffusion.networks.general_dit import GeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

FADITV2Config: LazyDict = L(GeneralDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    block_config="FA-CA-MLP",
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    block_x_format="THWBD",
    affline_emb_norm=True,
    use_adaln_lora=True,
    adaln_lora_dim=256,
)