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

from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH,
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
)
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.model.model_ctrl import (
    VideoDiffusionModelWithCtrl,
    VideoDiffusionT2VModelWithCtrl,
    VideoDistillModelWithCtrl,
)
from cosmos_transfer1.diffusion.model.model_multi_camera_ctrl import MultiVideoDiffusionModelWithCtrl
from cosmos_transfer1.diffusion.networks.general_dit_multi_view import MultiViewVideoExtendGeneralDIT
from cosmos_transfer1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Base configuration for 7B model
Base_7B_Config = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "add_fps_image_size_padding_mask"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        model=dict(
            latent_shape=[16, 16, 88, 160],
            net=dict(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
        ),
        job=dict(
            group="Control2World",
            name="Base_7B_Config",
        ),
    )
)


def make_ctrlnet_config_7b(
    hint_key: str = "control_input_seg",
    num_control_blocks: int = 3,
) -> LazyDict:
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return LazyDict(
        dict(
            defaults=[
                "/experiment/Base_7B_Config",
                {"override /hint_key": hint_key},
                {"override /net_ctrl": "faditv2_7b"},
                {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            ],
            job=dict(
                group="CTRL_7Bv1_lvg",
                name=f"CTRL_7Bv1pt3_lvg_tp_121frames_{hint_key}_block{num_control_blocks}",
                project="cosmos_transfer1",
            ),
            model=dict(
                hint_mask=hint_mask,
                hint_dropout_rate=0.3,
                conditioner=dict(video_cond_bool=dict()),
                net=L(VideoExtendGeneralDIT)(
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                net_ctrl=dict(
                    in_channels=17,
                    hint_channels=128,
                    num_blocks=28,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(28)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
            ),
            model_obj=L(VideoDiffusionModelWithCtrl)(),
        )
    )


def make_ctrlnet_config_7b_t2v(
    hint_key: str = "control_input_seg",
    num_control_blocks: int = 3,
) -> LazyDict:
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return LazyDict(
        dict(
            defaults=[
                "/experiment/Base_7B_Config",
                {"override /hint_key": hint_key},
                {"override /net_ctrl": "faditv2_7b"},
                {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            ],
            job=dict(
                group="CTRL_7Bv1_t2v",
                name=f"CTRL_7Bv1pt3_t2v_121frames_{hint_key}_block{num_control_blocks}",
                project="cosmos_ctrlnet1",
            ),
            model=dict(
                base_load_from=dict(
                    load_path=f"checkpoints/{BASE_7B_CHECKPOINT_AV_SAMPLE_PATH}",
                ),
                hint_mask=hint_mask,
                hint_dropout_rate=0.3,
                net=dict(
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                net_ctrl=dict(
                    in_channels=16,
                    hint_channels=16,
                    num_blocks=28,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(28)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
            ),
            model_obj=L(VideoDiffusionT2VModelWithCtrl)(),
        )
    )


def make_ctrlnet_config_7b_mv(
    hint_key: str = "control_input_seg",
    num_control_blocks: int = 3,
    t2w: bool = True,
) -> LazyDict:
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return LazyDict(
        dict(
            defaults=[
                "/experiment/Base_7B_Config",
                {"override /hint_key": hint_key},
                {"override /net_ctrl": "faditv2_7b_mv"},
                {"override /conditioner": "view_cond_ctrlnet_add_fps_image_size_padding_mask"},
            ],
            job=dict(
                group="CTRL_7Bv1_mv",
                name=f"CTRL_7Bv1pt3_sv2mv_{'t2w' if t2w else 'v2w'}_57frames_{hint_key}_block{num_control_blocks}",
                project="cosmos_ctrlnet1",
            ),
            model=dict(
                n_views=6,
                base_load_from=dict(
                    load_path=(
                        f"checkpoints/{BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH}"
                        if t2w
                        else f"checkpoints/{BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH}"
                    ),
                ),
                hint_mask=hint_mask,
                hint_dropout_rate=0.3,
                conditioner=dict(
                    video_cond_bool=dict(
                        condition_location="first_cam" if t2w else "first_cam_and_first_n",
                    )
                ),
                net=L(MultiViewVideoExtendGeneralDIT)(
                    n_views=6,
                    n_views_emb=7,
                    camera_condition_dim=6,
                    add_repeat_frame_embedding=True,
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                ),
                net_ctrl=dict(
                    in_channels=16,
                    hint_channels=16,
                    num_blocks=28,
                    n_views=6,
                    n_views_emb=7,
                    camera_condition_dim=6,
                    add_repeat_frame_embedding=True,
                    is_extend_model=True,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(28)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                ),
                tokenizer=dict(
                    video_vae=dict(
                        pixel_chunk_duration=57,
                    )
                ),
            ),
            model_obj=L(MultiVideoDiffusionModelWithCtrl)(),
        )
    )


def make_ctrlnet_config_7b_mv_waymo(
    hint_key: str = "control_input_seg",
    num_control_blocks: int = 3,
    t2w: bool = True,
) -> LazyDict:
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return LazyDict(
        dict(
            defaults=[
                "/experiment/Base_7B_Config",
                {"override /hint_key": hint_key},
                {"override /net_ctrl": "faditv2_7b_mv"},
                {"override /conditioner": "view_cond_ctrlnet_add_fps_image_size_padding_mask"},
            ],
            job=dict(
                group="CTRL_7Bv1_mv",
                name=f"CTRL_7Bv1pt3_sv2mv_{'t2w' if t2w else 'v2w'}_57frames_{hint_key}_waymo_block{num_control_blocks}",
                project="cosmos_ctrlnet1",
            ),
            model=dict(
                n_views=5,
                base_load_from=dict(
                    load_path=(
                        f"checkpoints/{BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH}"
                        if t2w
                        else f"checkpoints/{BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH}"
                    ),
                ),
                hint_mask=hint_mask,
                hint_dropout_rate=0.15,
                conditioner=dict(
                    video_cond_bool=dict(
                        condition_location="first_cam" if t2w else "first_cam_and_first_n",
                        cfg_unconditional_type="zero_condition_region_condition_mask",
                        apply_corruption_to_condition_region="noise_with_sigma",
                        condition_on_augment_sigma=False,
                        dropout_rate=0.0,
                        first_random_n_num_condition_t_max=0 if t2w else 2,
                        normalize_condition_latent=False,
                        augment_sigma_sample_p_mean=-3.0,
                        augment_sigma_sample_p_std=2.0,
                        augment_sigma_sample_multiplier=1.0,
                    )
                ),
                net=L(MultiViewVideoExtendGeneralDIT)(
                    n_views=5,
                    n_views_emb=7,
                    camera_condition_dim=6,
                    add_repeat_frame_embedding=True,
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                    num_blocks=28,
                ),
                adjust_video_noise=True,
                net_ctrl=dict(
                    in_channels=16,
                    hint_channels=16,
                    num_blocks=28,
                    n_views=5,
                    n_views_emb=7,
                    camera_condition_dim=6,
                    add_repeat_frame_embedding=True,
                    is_extend_model=True,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(28)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                tokenizer=dict(
                    video_vae=dict(
                        pixel_chunk_duration=57,
                    )
                ),
            ),
            model_obj=L(MultiVideoDiffusionModelWithCtrl)(),
        )
    )


# Register base configs
cs.store(group="experiment", package="_global_", name=Base_7B_Config["job"]["name"], node=Base_7B_Config)
# Register all control configurations
num_control_blocks = 3
for key in CTRL_HINT_KEYS_COMB.keys():
    # Register 7B configurations
    config_7b = make_ctrlnet_config_7b(hint_key=key, num_control_blocks=num_control_blocks)
    cs.store(group="experiment", package="_global_", name=config_7b["job"]["name"], node=config_7b)

# Register t2v based control net
num_control_blocks = 3
for key in ["control_input_hdmap", "control_input_lidar"]:
    # Register 7B configurations
    config_7b = make_ctrlnet_config_7b_t2v(hint_key=key, num_control_blocks=num_control_blocks)
    cs.store(group="experiment", package="_global_", name=config_7b["job"]["name"], node=config_7b)

num_control_blocks = 3
for key in ["control_input_hdmap", "control_input_lidar"]:
    for t2w in [True, False]:
        # Register 7B sv2mv configurations
        config_7b = make_ctrlnet_config_7b_mv(hint_key=key, num_control_blocks=num_control_blocks, t2w=t2w)
        cs.store(group="experiment", package="_global_", name=config_7b["job"]["name"], node=config_7b)


# Register waymo example
num_control_blocks = 3
for key in ["control_input_hdmap", "control_input_lidar"]:
    for t2w in [True, False]:
        # Register 7B sv2mv configurations
        config_7b = make_ctrlnet_config_7b_mv_waymo(hint_key=key, num_control_blocks=num_control_blocks, t2w=t2w)
        cs.store(group="experiment", package="_global_", name=config_7b["job"]["name"], node=config_7b)


def make_ctrlnet_config_7b_distilled(
    hint_key: str = "control_input_edge",
    num_control_blocks: int = 3,
) -> LazyDict:
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return LazyDict(
        dict(
            defaults=[
                "/experiment/Base_7B_Config",
                {"override /hint_key": hint_key},
                {"override /net": "faditv2_7b"},
                {"override /net_ctrl": "faditv2_7b"},
                {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
                "_self_",
            ],
            job=dict(
                group="DISTILL_CTRL_7Bv1",
                name=f"CTRL_7Bv1pt3_lvg_fsdp_distilled_121frames_{hint_key}_block{num_control_blocks}",
                project="cosmos_nano_v1",
            ),
            model=dict(
                base_load_from=dict(
                    load_path=f"checkpoints/{EDGE2WORLD_CONTROLNET_7B_DISTILLED_CHECKPOINT_PATH}",
                ),
                hint_mask=hint_mask,
                hint_dropout_rate=0.0,
                conditioner=dict(
                    video_cond_bool=dict(
                        condition_location="first_random_n",
                        cfg_unconditional_type="zero_condition_region_condition_mask",
                        apply_corruption_to_condition_region="noise_with_sigma_fixed",
                        condition_on_augment_sigma=False,
                        dropout_rate=0.0,
                        first_random_n_num_condition_t_max=2,
                        first_random_n_num_condition_t_min=0,
                        normalize_condition_latent=False,
                        augment_sigma_sample_p_mean=-3.0,
                        augment_sigma_sample_p_std=2.0,
                        augment_sigma_sample_multiplier=1.0,
                    )
                ),
                net=L(VideoExtendGeneralDIT)(
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                    rope_t_extrapolation_ratio=2,
                ),
                net_ctrl=dict(
                    in_channels=17,
                    hint_channels=128,
                    num_blocks=28,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(28)],
                    num_control_blocks=num_control_blocks,
                    dropout_ctrl_branch=0,
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
            ),
            model_obj=L(VideoDistillModelWithCtrl)(),
        )
    )


# Register the specific distilled configuration
distilled_config = make_ctrlnet_config_7b_distilled(hint_key="control_input_edge", num_control_blocks=3)
cs.store(
    group="experiment",
    package="_global_",
    name="CTRL_7Bv1pt3_lvg_fsdp_distilled_121frames_control_input_edge_block3",
    node=distilled_config,
)
