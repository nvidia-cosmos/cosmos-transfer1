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

from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_transfer1.diffusion.config.training.ema import PowerEMAConfig


@attrs.define(slots=False)
class FSDPConfig:
    policy: str = "block"
    checkpoint: bool = False
    min_num_params: int = 1024
    sharding_group_size: int = 8
    sharding_strategy: str = "full"


@attrs.define(slots=False)
class DistillModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    teacher_net_name: str = "net"  # Pointer to config property to base module on
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    sigma_data: float = 0.5
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images_1024"  # key to fetch input image from data_batch
    loss_scale: float = 1.0
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames
    fsdp_enabled: bool = False
    use_torch_compile: bool = False
    fsdp: FSDPConfig = attrs.field(factory=FSDPConfig)
    use_dummy_temporal_dim: bool = False  # Whether to use dummy temporal dimension in data
    adjust_video_noise: bool = False  # whether adjust video noise according to the video length

    base_load_from: LazyDict = None  # specify where to load net

    # flag for control net
    is_ctrl_net: bool = False

    # back-compatibility but unused
    ema: LazyDict = PowerEMAConfig
    loss_mask_enabled: bool = False
    loss_masking: LazyDict = None


@attrs.define(slots=False)
class DistillCtrlModelConfig(DistillModelConfig):
    net_ctrl: LazyDict = None
    hint_key: str = None
    finetune_base_model: bool = True
    hint_mask: list = [True]
    hint_dropout_rate: float = 0.0
    num_control_blocks: int = 5
    random_drop_control_blocks: bool = False
    pixel_corruptor: LazyDict = None
    is_ctrl_net: bool = True


@attrs.define(slots=False)
class DMD2ModelConfig(DistillModelConfig):
    # optimizer and scheduler for the fake score net
    fake_score_optimizer: LazyDict = None
    fake_score_scheduler: LazyDict = None
    fake_score_net_name: str = "net"  # Pointer to config property to base module on

    discriminator: LazyDict = None
    # optimizer and scheduler for the discriminator
    discriminator_optimizer: LazyDict = None
    discriminator_scheduler: LazyDict = None

    # student update frequency
    student_update_freq: int = 5
    # guidance scale for cfg in teacher diffusion model
    guidance_scale: float = 0

    # weight for the GAN loss in the student update phase
    gan_loss_weight_gen: float = 0.001

    # weight for the regression loss in the student update phase
    recon_loss_weight: float = 0.0

    # flag for reconstruction loss only
    recon_loss_only: bool = False

    # use negative prompt or not for CFG in DMD2
    use_negative_prompt: bool = True

    # noise schedule type ["edm_sampling", "inference"]
    noise_schedule_type: str = "edm_sampling"

    # flag for LADD training
    ladd: bool = False


@attrs.define(slots=False)
class DMD2CtrlModelConfig(DMD2ModelConfig, DistillCtrlModelConfig):
    is_ctrl_net: bool = True
