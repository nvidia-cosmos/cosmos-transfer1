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
This script will make + register the architecture + training-related configs for all the control modalities (one config per modality).
The configs are registered under the group "experiment" and can be used in training by passing the experiment name as an argument.

Example usage:
    - [dryrun, generate and inspect EdgeControl config] torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3
    - [real run, 8 gpu, train SegControl] torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3
    - [real run, 8 gpu, train DepthControl] torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict
from cosmos_transfer1.diffusion.config.transfer.blurs import random_blur_config
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.training.models.model_ctrl import VideoDiffusionModelWithCtrl  # this one has training support
from cosmos_transfer1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT

cs = ConfigStore.instance()

num_frames = 121
num_blocks = 28
num_control_blocks = 3

# TODO (qianlim) add data config
def get_data_train_name(hint_key: str) -> str:
    pass

def get_data_val_name(hint_key: str) -> str:
    pass

def make_ctrlnet_config_7b_training(
    hint_key: str = "control_input_canny",
    num_control_blocks: int = 3,
) -> LazyDict:

    data_train = get_data_train_name(hint_key)
    data_val = get_data_val_name(hint_key)

    # Create the complete configuration in one step
    config = LazyDict(
        dict(
            defaults=[
                {"override /net": "faditv2_7b"},
                {"override /net_ctrl": "faditv2_7b"},
                {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
                {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
                #
                {"override /hint_key": hint_key},
                {"override /callbacks": "basic"},
                {"override /checkpoint": "local"},
                {"override /ckpt_klass": "fast_tp"},
                #
                {"override /data_train": data_train},
                {"override /data_val": data_val},
                "_self_",
            ],
            job=dict(
                group="CTRL_7Bv1_lvg",
                name=f"CTRL_7Bv1pt3_lvg_tp_121frames_{hint_key}_block{num_control_blocks}",
                project="cosmos_transfer1_posttrain",
            ),
            optimizer=dict(
                lr=2 ** (-14.3),  # ~5e-5
                weight_decay=0.1,
                betas=[0.9, 0.99],
                eps=1e-10,
            ),
            checkpoint=dict(
                load_path="checkpoints/nvidia/Cosmos-Transfer1-7B/vis_control.pt",  # modify as needed. Here we assume post-train our pre-trained VisControl model.
                broadcast_via_filesystem=True,
                save_iter=1000,
                load_training_state=False,
                strict_resume=True,
                keys_not_to_resume=[],
            ),
            trainer=dict(
                distributed_parallelism="ddp",
                logging_iter=200,
                max_iter=999_999_999,
            ),
            model_parallel=dict(
                tensor_model_parallel_size=8,
                sequence_parallel=True,
            ),
            model=dict(
                fsdp_enabled=False,
                context_parallel_size=1,
                loss_reduce='mean',
                latent_shape=[
                    16,
                    (num_frames - 1) // 8 + 1,
                    88,
                    160,
                ],
                base_load_from=dict(
                    load_path="checkpoints/nvidia/Cosmos-Transfer1-7B/base_model.pt",  # modify as needed. This is the base model (that's frozen during training).
                ),
                finetune_base_model=False,
                hint_mask=[True] * len(CTRL_HINT_KEYS_COMB[hint_key]),
                hint_dropout_rate=0.3,
                conditioner=dict(
                    video_cond_bool=dict(
                        condition_location="first_random_n",
                        cfg_unconditional_type="zero_condition_region_condition_mask",
                        apply_corruption_to_condition_region="noise_with_sigma",
                        condition_on_augment_sigma=False,
                        dropout_rate=0.0,
                        first_random_n_num_condition_t_max=2,
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
                    rope_h_extrapolation_ratio=1,
                    rope_t_extrapolation_ratio=2,
                    rope_w_extrapolation_ratio=1,
                ),
                adjust_video_noise=True,
                net_ctrl=dict(
                    in_channels=17,
                    hint_channels=128,
                    num_blocks=num_blocks,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(num_blocks)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                ema=dict(
                    enabled=True,
                ),
            ),
            model_obj=L(VideoDiffusionModelWithCtrl)(),
            scheduler=dict(
                warm_up_steps=[2500],
                cycle_lengths=[10000000000000],
                f_start=[1.0e-6],
                f_max=[1.0],
                f_min=[1.0],
            ),
            dataloader_val=dict(
                dataset=dict(
                    resolution="720",
                    num_video_frames=num_frames,
                ),
            ),
            dataloader_train=dict(
                dataloaders=dict(
                    image_data=dict(
                        dataloader=dict(
                            batch_size=1,
                            dataset=dict(
                                resolution="720",
                                blur_config=random_blur_config,
                            ),
                        ),
                        ratio=0,  # only use video data for training.
                    ),
                    video_data=dict(
                        dataloader=dict(
                            batch_size=1,
                            dataset=dict(
                                resolution="720",
                                num_video_frames=num_frames,
                                blur_config=random_blur_config,
                            ),
                        ),
                        ratio=1,
                    ),
                ),
            ),
        )
    )
    return config


"""
Register configurations
The loop below will register all experiments CTRL_7Bv1pt3_lvg_tp_121frames_control_input_{hint_key_name}_block3 for each hint_key_name
and then in training command, simply need to pass the "experiment" arg to override the configs. See the docstring at top of this script
for an example.
"""
for key in CTRL_HINT_KEYS_COMB.keys():
    config = make_ctrlnet_config_7b_training(hint_key=key, num_control_blocks=num_control_blocks)
    cs.store(
        group="experiment",
        package="_global_",
        name=config["job"]["name"],
        node=config,
    )