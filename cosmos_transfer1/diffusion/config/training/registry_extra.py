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
Registry for training experiments, callbacks and data.
"""

import copy

from hydra.core.config_store import ConfigStore

import cosmos_transfer1.diffusion.config.training.registry as base_training_registry
from cosmos_transfer1.diffusion.config.base.data import register_data_ctrlnet
from cosmos_transfer1.diffusion.config.registry import register_conditioner
from cosmos_transfer1.diffusion.config.training.tokenizer import (
    DebugTokenizerConfig,
    get_cosmos_diffusion_tokenizer_comp8x8x8,
)
from cosmos_transfer1.diffusion.config.transfer.conditioner import (
    CTRL_HINT_KEYS,
    BaseVideoConditionerWithCtrlConfig,
    VideoConditionerFpsSizePaddingWithCtrlConfig,
    ViewConditionedVideoConditionerFpsSizePaddingWithCtrlConfig,
)
from cosmos_transfer1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_transfer1.diffusion.training.networks.general_dit_ctrl_enc import GeneralDITEncoder
from cosmos_transfer1.diffusion.training.networks.general_dit_ctrl_enc_multicamera import GeneralDITMulticamEncoder

# from cosmos_transfer1.diffusion.config.registry import register_tokenizer
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

FADITV2ConfigTrain: LazyDict = L(GeneralDIT)(
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
    additional_timestamp_channels=None,
    affline_emb_norm=True,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    legacy_patch_emb=False,
)

DebugDITConfigTrain: LazyDict = copy.deepcopy(FADITV2ConfigTrain)
DebugDITConfigTrain.patch_spatial = 2
DebugDITConfigTrain.patch_temporal = 1
DebugDITConfigTrain.model_channels = 128 * 2
DebugDITConfigTrain.num_blocks = 4
DebugDITConfigTrain.num_heads = 2

num_blocks = FADITV2ConfigTrain["num_blocks"]
FADITV2EncoderConfigTrain = copy.deepcopy(FADITV2ConfigTrain)
FADITV2EncoderConfigTrain["_target_"] = GeneralDITEncoder
FADITV2EncoderConfigTrain["layer_mask"] = [True if i > num_blocks // 2 else False for i in range(num_blocks)]

num_blocks = FADITV2ConfigTrain["num_blocks"]
FADITV2MultiCamEncoderConfig = copy.deepcopy(FADITV2ConfigTrain)
FADITV2MultiCamEncoderConfig["_target_"] = GeneralDITMulticamEncoder
FADITV2MultiCamEncoderConfig["layer_mask"] = [True if i > num_blocks // 2 else False for i in range(num_blocks)]

num_blocks = DebugDITConfigTrain["num_blocks"]
DebugDITEncoderConfigTrain = copy.deepcopy(DebugDITConfigTrain)
DebugDITEncoderConfigTrain["_target_"] = GeneralDITEncoder
DebugDITEncoderConfigTrain["layer_mask"] = [True if i > 2 else False for i in range(num_blocks)]


def register_net_train(cs):
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_7b",
        node=FADITV2ConfigTrain,
    )
    cs.store(
        group="net",
        package="model.net",
        name="tiny_fa",
        node=DebugDITConfigTrain,
    )

    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_7b", node=FADITV2EncoderConfigTrain)
    cs.store(group="net_ctrl", package="model.net_ctrl", name="faditv2_sv2mv", node=FADITV2MultiCamEncoderConfig)
    cs.store(group="net_ctrl", package="model.net_ctrl", name="tiny_fa", node=DebugDITEncoderConfigTrain)


def register_conditioner_ctrlnet(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="ctrlnet",
        node=BaseVideoConditionerWithCtrlConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="ctrlnet_add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingWithCtrlConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="view_cond_ctrlnet_add_fps_image_size_padding_mask",
        node=ViewConditionedVideoConditionerFpsSizePaddingWithCtrlConfig,
    )


def register_tokenizer(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",
        node=get_cosmos_diffusion_tokenizer_comp8x8x8(resolution="720", chunk_duration=121),
    )
    cs.store(group="tokenizer", package="model.tokenizer", name="debug_tokenizer", node=DebugTokenizerConfig)


def register_configs():
    cs = ConfigStore.instance()

    # register all the basic configs: net, conditioner, tokenizer.
    register_net_train(cs)
    register_conditioner(cs)
    register_conditioner_ctrlnet(cs)
    register_tokenizer(cs)

    # register training configs: optimizer, scheduler, callbacks, etc.
    base_training_registry.register_configs()

    # register data, experiment, callbacks
    register_data_ctrlnet(cs)

    # register hint keys
    for hint_key in CTRL_HINT_KEYS:
        cs.store(
            group="hint_key",
            package="model",
            name=hint_key,
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=False)),
        )
        cs.store(
            group="hint_key",
            package="model",
            name=f"{hint_key}_grayscale",
            node=dict(hint_key=dict(hint_key=hint_key, grayscale=True)),
        )
