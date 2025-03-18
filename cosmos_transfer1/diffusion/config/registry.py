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

from hydra.core.config_store import ConfigStore

from cosmos_transfer1.diffusion.config.base.conditioner import (
    BaseVideoConditionerConfig,
    VideoConditionerFpsSizePaddingConfig,
    VideoExtendConditionerConfig,
)
from cosmos_transfer1.diffusion.config.base.net import FADITV2Config
from cosmos_transfer1.diffusion.config.base.tokenizer import get_cosmos_diffusion_tokenizer_comp8x8x8


def register_net(cs):
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_7b",
        node=FADITV2Config,
    )


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="basic",
        node=BaseVideoConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond",
        node=VideoExtendConditionerConfig,
    )


def register_tokenizer(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",
        node=get_cosmos_diffusion_tokenizer_comp8x8x8(resolution="720", chunk_duration=121),
    )


def register_configs():
    cs = ConfigStore.instance()

    register_net(cs)
    register_conditioner(cs)
    register_tokenizer(cs)
