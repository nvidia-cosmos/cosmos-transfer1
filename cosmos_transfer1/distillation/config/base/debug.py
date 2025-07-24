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
Debug configs for for developers and testers to quickly verify model behavior during development stages.
Not intended for reproducible training.
"""

from cosmos_transfer1.utils.lazy_config import PLACEHOLDER
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.distillation.models.model_dmd2 import DMD2DistillCtrlModel

"""
torchrun --nproc_per_node=1 --master_port=12341 -m projects.edify_image.v4.train --config=projects/cosmos/nano/v1/config/config.py -- experiment=debug_local_basic_cb_ddp trainer.max_iter=5
"""

# ------------------------------------------------------
# Debug base model distillation

DEBUG_LOCAL_TINY_DDP_EXP = dict(
    defaults=[
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /ckpt_klass": "multi_rank"},
        {"override /vae": "dummy_vae1"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
    upload_reproducible_setup=False,
)

DEBUG_LOCAL_DDP_EXP = dict(
    defaults=[
        {"override /net": "full_xl2"},
        {"override /discriminator": "conv3d_pool_xl2"},
        {"override /callbacks": ["basic", "wandb"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
            num_blocks=4,
        ),
        discriminator=dict(
            num_blocks=4,
        ),
    ),
    upload_reproducible_setup=False,
)

DEBUG_LOCAL_CP_EXP = dict(
    defaults=[
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /callbacks": ["basic"]},
        {"override /ckpt_klass": "multi_rank"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
        {"override /net": "full_xl2"},
        {"override /discriminator": "conv3d_pool_xl2"},
        {"override /callbacks": ["basic", "wandb"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
        {"override /net": "full_xl2"},
        {"override /discriminator": "conv3d_pool_xl2"},
        {"override /callbacks": ["basic", "wandb"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
# Debug v2w model distillation

DEBUG_V2W_LOCAL_TINY_DDP_EXP = dict(
    defaults=[
        {"override /net": "tiny_fa"},
        {"override /discriminator": "conv3d_pool_tiny_fa"},
        {"override /conditioner": "video_cond"},
        {"override /ckpt_klass": "multi_rank"},
        {"override /vae": "dummy_vae1"},
        "_self_",
    ],
    job=dict(
        group="debug_v2w",
    ),
    checkpoint=dict(
        save_iter=10,
    ),
    trainer=dict(
        max_iter=100,
        logging_iter=10,
    ),
    model=dict(
        conditioner=dict(
            video_cond_bool=dict(
                condition_location="first_random_n",
                cfg_unconditional_type="zero_condition_region_condition_mask",
                apply_corruption_to_condition_region="noise_with_sigma",
                condition_on_augment_sigma=False,
            )
        ),
        ema=dict(
            enabled=False,
        ),
        net=L(VideoExtendGeneralDIT)(
            num_blocks=2,
        ),
        discriminator=dict(
            num_blocks=2,
        ),
    ),
    # using the v2w model for training
    model_obj=L(DMD2DistillCtrlModel)(
        config=PLACEHOLDER,
    ),
    upload_reproducible_setup=False,
)

DEBUG_V2W_LOCAL_CP_FSDP_EXP = dict(
    defaults=[
        {"override /net": "full_xl2"},
        {"override /discriminator": "conv3d_pool_xl2"},
        {"override /callbacks": ["basic", "wandb"]},
        {"override /conditioner": "video_cond"},
        "_self_",
    ],
    job=dict(
        group="debug_v2w",
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
        conditioner=dict(
            video_cond_bool=dict(
                condition_location="first_random_n",
                cfg_unconditional_type="zero_condition_region_condition_mask",
                apply_corruption_to_condition_region="noise_with_sigma",
                condition_on_augment_sigma=False,
            )
        ),
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
        net=L(VideoExtendGeneralDIT)(),
    ),
    # using the v2w model for training
    model_obj=L(DMD2DistillCtrlModel)(
        config=PLACEHOLDER,
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
)

# ------------------------------------------------------
# Debug KD model distillation

DEBUG_LOCAL_KD_TPSP_CP_EXP = dict(
    defaults=[
        {"override /net": "full_xl2"},
        {"override /callbacks": ["basic", "wandb"]},
        {"override /conditioner": "add_fps_image_size_padding_mask"},
        {"override /ckpt_klass": "tp"},
        "_self_",
    ],
    job=dict(
        group="debug",
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
            block_x_format="THWBD",
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
    ),
)
