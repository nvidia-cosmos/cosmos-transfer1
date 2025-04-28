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

from typing import List

import attrs

from cosmos_transfer1.diffusion.config.training.ema import PowerEMAConfig
from cosmos_transfer1.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class FSDPConfig:
    policy: str = "block"
    checkpoint: bool = False
    min_num_params: int = 1024
    sharding_group_size: int = 8
    sharding_strategy: str = "full"


@attrs.define(slots=False)
class DefaultModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    sigma_data: float = 0.5
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames

    # training related
    ema: LazyDict = PowerEMAConfig
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    camera_sample_weight: LazyDict = LazyDict(
        dict(
            enabled=False,
            weight=5.0,
        )
    )
    aesthetic_finetuning: LazyDict = LazyDict(
        dict(
            enabled=False,
        )
    )
    loss_mask_enabled: bool = False
    loss_masking: LazyDict = None
    loss_add_logvar: bool = True
    input_image_key: str = "images_1024"  # key to fetch input image from data_batch
    loss_reduce: str = "sum"
    loss_scale: float = 1.0
    fsdp_enabled: bool = False
    use_torch_compile: bool = False
    fsdp: FSDPConfig = attrs.field(factory=FSDPConfig)
    use_dummy_temporal_dim: bool = False  # Whether to use dummy temporal dimension in data
    adjust_video_noise: bool = False  # whether or not adjust video noise accroding to the video length
    context_parallel_size: int = 1  # Number of context parallel groups

    # `num_latents_to_drop` is mechanism to satisfy the CP%8==0 and (1I,N*P,1I) latents setup.
    # Since our tokenizer is causal and has the `T+1` input frames setup, it makes it
    # a little challenging to sample exact number of frames from file, and encode those.
    # Instead, we sample as many frame from file, run the tokenizer twice, and discard the second
    # chunk's P-latents, ensuring the above two requirements. By default, this flag does not have any effect.
    num_latents_to_drop: int = 0  # number of latents to drop


@attrs.define(slots=False)
class MultiviewModelConfig(DefaultModelConfig):
    n_views: int = 6


@attrs.define(slots=False)
class LatentDiffusionDecoderModelConfig(DefaultModelConfig):
    tokenizer_corruptor: LazyDict = None
    latent_corruptor: LazyDict = None
    pixel_corruptor: LazyDict = None
    diffusion_decoder_cond_sigma_low: float = None
    diffusion_decoder_cond_sigma_high: float = None
    diffusion_decoder_corrupt_prob: float = None
    condition_on_tokenizer_corruptor_token: bool = False
