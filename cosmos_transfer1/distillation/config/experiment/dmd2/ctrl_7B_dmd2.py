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
This script will make + register the DMD2 configs for all the control modalities (one config per modality).
The configs are registered under the group "experiment" and can be used in training by passing the experiment name as an argument.

Example usage:
    - [dryrun, generate and inspect edge control DMD2 config]:
            torchrun --nproc_per_node=1 -m cosmos_transfer1.distillation.train --dryrun --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train
    - [real run, 8 gpu, train edge control DMD2 from released checkpoint]:
            torchrun --nproc_per_node=8 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names
from cosmos_transfer1.diffusion.training.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()

NUM_FRAMES = 121
NUM_BLOCKS = 28
NUM_CONTROL_BLOCKS = 3


def make_ctrl_dmd2_experiment(hint_key: str = "control_input_edge") -> dict:
    # Get the checkpoint path for the teacher model
    hint_key_short = hint_key.replace("control_input_", "")  # "control_input_edge" -> "edge"
    pretrain_ckpt_path = default_model_names[hint_key_short]
    teacher_checkpoint_path = os.path.join(
        "checkpoints", os.path.dirname(pretrain_ckpt_path), "checkpoints_teacher", os.path.basename(pretrain_ckpt_path)
    )

    # Infer the hint mask based on the hint key
    hint_mask = [True] * len(CTRL_HINT_KEYS_COMB[hint_key])

    return dict(
        defaults=[
            {"override /callbacks": ["basic", "train_vis"]},
            # For demo purposes, we use mock data that you can find in cosmos_transfer1/distillation/datasets/mock_distill_dataset.py.
            # Replace with f"dmd2_transfer_train_data_{hint_key}" to use your own data.
            {"override /data_train": f"mock_ctrl_distill_{hint_key}"},
            # Replace with f"dmd2_transfer_val_data_{hint_key}" to use your own data.
            {"override /data_val": f"mock_ctrl_distill_{hint_key}"},
            {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            {"override /discriminator": "conv3d_pool_faditv2"},
            {"override /hint_key": hint_key},
            {"override /net": "faditv2_7b"},
            {"override /net_ctrl": "faditv2_7b"},
            "_self_",
        ],
        job=dict(
            group="DISTILL_CTRL_7Bv1",
            name=f"DISTILL_CTRL_7Bv1_{hint_key_short}_fsdp_dmd2_train",
        ),
        checkpoint=dict(
            save_iter=500,
            # If load_path is specified, the student model will be initialized from the given checkpoint.
            # Otherwise, the student model will be initialized from the teacher model.
            load_path="",
            # Skip loading discriminator if the checkpoint has a different (or no) discriminator architecture.
            # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
            # If loading from a KD checkpoint, this should be set to True.
            skip_load_discriminator=False,
            # Skip loading fake score if the checkpoint has no fake score network.
            # If resuming training, this setting will be ignored and all components of the previous checkpoint will be loaded.
            # If loading from a KD checkpoint, this should be set to True.
            skip_load_fake_score=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            distributed_parallelism="fsdp",
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=10.0,
                ),
                train_sample=dict(
                    every_n=500,
                ),
            ),
            # Set to 1 to disable gradient accumulation.
            grad_accum_iter=1,
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        optimizer=dict(
            lr=5e-7,
            weight_decay=0.01,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        model=dict(
            discriminator_optimizer=dict(
                lr=5e-7,
                weight_decay=0.01,
            ),
            discriminator_scheduler=dict(
                warm_up_steps=[0],
            ),
            fake_score_optimizer=dict(
                lr=5e-7,
                weight_decay=0.01,
            ),
            fake_score_scheduler=dict(
                warm_up_steps=[0],
            ),
            gan_loss_weight_gen=0.001,
            guidance_scale=5.0,
            ema=dict(
                enabled=False,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=False,
                min_num_params=3000,
                # By default, we use full FSDP. Change to "hybrid" to use hybrid FSDP.
                sharding_strategy="full",
                # Applies to the hybrid sharding strategy. Unused for full sharding strategy.
                sharding_group_size=16,
            ),
            base_load_from=dict(
                load_path=teacher_checkpoint_path,
            ),
            latent_shape=[
                16,
                (NUM_FRAMES - 1) // 8 + 1,
                88,
                160,
            ],
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
                # Enable gradient checkpointing to reduce memory usage. Set to False to disable.
                use_checkpoint=True,
            ),
            adjust_video_noise=True,
            net_ctrl=dict(
                in_channels=17,
                hint_channels=128,
                num_blocks=NUM_BLOCKS,
                layer_mask=[True if (i >= NUM_CONTROL_BLOCKS) else False for i in range(NUM_BLOCKS)],
                num_control_blocks=NUM_CONTROL_BLOCKS,
                dropout_ctrl_branch=0,
                extra_per_block_abs_pos_emb=True,
                pos_emb_learnable=True,
                extra_per_block_abs_pos_emb_type="learnable",
                # Enable gradient checkpointing to reduce memory usage. Set to False to disable.
                use_checkpoint=True,
            ),
            teacher_net_name="net_ctrl",
            fake_score_net_name="net_ctrl",
            use_negative_prompt=True,
            negative_prompt_path="datasets/negative_prompt/transfer1.pkl",
        ),
    )


for key in CTRL_HINT_KEYS_COMB.keys():
    if key in ["control_input_upscale", "control_input_hdmap", "control_input_lidar"]:
        continue
    exp = make_ctrl_dmd2_experiment(key)
    cs.store(
        group="experiment",
        package="_global_",
        name=exp["job"]["name"],
        node=exp,
    )
