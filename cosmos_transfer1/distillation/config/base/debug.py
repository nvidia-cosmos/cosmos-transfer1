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

from cosmos_transfer1.diffusion.training.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L

"""
Sample command to run the debug base experiment:
torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_base_dmd2.py -- experiment=debug_local_ddp trainer.max_iter=5 trainer.logging_iter=1

Sample command to run the debug ctrlnet experiment:
torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=debug_ctrlnet_local_ddp trainer.max_iter=5 trainer.logging_iter=1
"""

# ------------------------------------------------------
# Debug experiments for base model distillation

DEBUG_LOCAL_DDP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_distill_debug"},
        {"override /data_val": "mock_distill_debug"},
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /callbacks": ["basic"]},
        {"override /ckpt_klass": "multi_rank"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_base",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
        net=dict(
            num_blocks=2,
        ),
        discriminator=dict(
            num_blocks=2,
        ),
    ),
)

DEBUG_LOCAL_CP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_distill_debug"},
        {"override /data_val": "mock_distill_debug"},
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /callbacks": ["basic"]},
        {"override /ckpt_klass": "multi_rank"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_base",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
)

DEBUG_LOCAL_FSDP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_distill_debug"},
        {"override /data_val": "mock_distill_debug"},
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /callbacks": ["basic"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_base",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
        distributed_parallelism="fsdp",
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
        fsdp_enabled=True,
        fsdp=dict(
            policy="block",
            checkpoint=False,
            min_num_params=3000,
            sharding_strategy="full",
        ),
        net=dict(
            num_blocks=4,
        ),
        discriminator=dict(
            num_blocks=4,
        ),
    ),
)

DEBUG_LOCAL_CP_FSDP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_distill_debug"},
        {"override /data_val": "mock_distill_debug"},
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /callbacks": ["basic"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_base",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
        distributed_parallelism="fsdp",
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
        fsdp_enabled=True,
        fsdp=dict(
            policy="block",
            checkpoint=False,
            min_num_params=3000,
            sharding_strategy="full",
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
)

# ------------------------------------------------------
# Debug experiments for ctrlnet model distillation

DEBUG_CTRLNET_LOCAL_DDP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_ctrl_distill_debug_control_input_edge"},
        {"override /data_val": "mock_ctrl_distill_debug_control_input_edge"},
        {"override /hint_key": "control_input_edge"},
        {"override /net": "tiny_fa"},
        {"override /net_ctrl": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /callbacks": ["basic"]},
        {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
        {"override /ckpt_klass": "multi_rank"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_ctrlnet",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
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
            dropout_ctrl_branch=0,
            extra_per_block_abs_pos_emb=True,
            pos_emb_learnable=True,
            extra_per_block_abs_pos_emb_type="learnable",
        ),
        teacher_net_name="net_ctrl",
        fake_score_net_name="net_ctrl",
    ),
)

DEBUG_CTRLNET_LOCAL_CP_FSDP_EXP = dict(
    defaults=[
        {"override /data_train": "mock_ctrl_distill_debug_control_input_edge"},
        {"override /data_val": "mock_ctrl_distill_debug_control_input_edge"},
        {"override /hint_key": "control_input_edge"},
        {"override /net": "tiny_fa"},
        {"override /net_ctrl": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /callbacks": ["basic"]},
        {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
        {"override /tokenizer": "debug_tokenizer"},
        "_self_",
    ],
    job=dict(
        group="debug_ctrlnet",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
        distributed_parallelism="fsdp",
    ),
    model=dict(
        ema=dict(
            enabled=False,
        ),
        fsdp_enabled=True,
        fsdp=dict(
            policy="block",
            checkpoint=False,
            min_num_params=3000,
            sharding_strategy="full",
        ),
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
            dropout_ctrl_branch=0,
            extra_per_block_abs_pos_emb=True,
            pos_emb_learnable=True,
            extra_per_block_abs_pos_emb_type="learnable",
        ),
        teacher_net_name="net_ctrl",
        fake_score_net_name="net_ctrl",
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
)
