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
    - [dryrun, generate and inspect EdgeControl config]:
            torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain
    - [real run, 8 gpu, train SegControl from scratch]:
            torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3_pretrain
    - [real run, 8 gpu, train SegControl from released checkpoint]:
            torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3_posttrain
"""

import copy
from hydra.core.config_store import ConfigStore
import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer1.checkpoints import COSMOS_TRANSFER1_7B_CHECKPOINT
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names
from cosmos_transfer1.diffusion.training.models.model_ctrl import (  # this one has training support
    VideoDiffusionModelWithCtrl,
)
from cosmos_transfer1.diffusion.training.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

num_frames = 121
num_blocks = 28
num_control_blocks = 3


def make_ctrlnet_config_7b_training(
    hint_key: str = "control_input_canny", num_control_blocks: int = 3, pretrain_model_path: str = ""
) -> LazyDict:
    if pretrain_model_path == "":
        job_name = f"CTRL_7Bv1pt3_lvg_tp_121frames_{hint_key}_block{num_control_blocks}_pretrain"
        job_project = "cosmos_transfer1_pretrain"
    else:
        job_name = f"CTRL_7Bv1pt3_lvg_tp_121frames_{hint_key}_block{num_control_blocks}_posttrain"
        job_project = "cosmos_transfer1_posttrain"

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
                # data: register your own data at cosmos_transfer1/diffusion/config/base/data.py
                {"override /data_train": f"example_transfer_train_data_{hint_key}"},
                {"override /data_val": f"example_transfer_val_data_{hint_key}"},
                "_self_",
            ],
            # ckpt, config yaml files etc. will be saved under checkpoints/<project>/<group>/<name>/
            job=dict(
                project=job_project,
                group="CTRL_7Bv1_lvg",
                name=job_name,
            ),
            optimizer=dict(
                lr=2 ** (-14.3),  # ~5e-5
                weight_decay=0.1,
                betas=[0.9, 0.99],
                eps=1e-10,
            ),
            checkpoint=dict(
                load_path=pretrain_model_path,  # Modify load_path as needed if you do post-training (fine-tuning). If training from scratch, leave it empty.
                broadcast_via_filesystem=True,
                save_iter=1000,  # 1000 iterations per checkpoint. Update as needed.
                load_training_state=False,
                strict_resume=False,  # TODO (qianlim): temporary hack: We have excluded the base model ckpt from each full controlnet. The base model weights are loaded below, see 'base_load_from'.
                keys_not_to_resume=[],
            ),
            trainer=dict(
                distributed_parallelism="ddp",
                logging_iter=200,  # will log iter speed, loss, etc. every 200 iterations. (Will log per-iteration speed for the first 1000 iterations.)
                max_iter=999_999_999,
                timestamp_seed=True,
            ),
            model_parallel=dict(
                tensor_model_parallel_size=8,
                sequence_parallel=True,
            ),
            model=dict(
                fsdp_enabled=False,
                context_parallel_size=1,
                loss_reduce="mean",
                latent_shape=[
                    16,
                    (num_frames - 1) // 8 + 1,  # for 121 frames, this is 16
                    88,
                    160,
                ],
                base_load_from=dict(
                    load_path=os.path.join(
                        "checkpoints", COSMOS_TRANSFER1_7B_CHECKPOINT, "checkpoints_tp", "base_model_model_mp_*.pt"
                    )
                ),  # modify as needed. This is the TP version of base model ckpt (that's frozen during training).
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
        )
    )
    return config


"""
Register configurations
The loop below will register ALL experiments CTRL_7Bv1pt3_lvg_tp_121frames_control_input_{hint_key_name}_block3_{pretrain_or_posttrain} for ALL hint_key_name.
Then in training command, simply need to pass the "experiment" arg to override the configs. See the docstring at top of this script for an example.

# NOTE: To launch real post-training, convert the checkpoints to TP checkpoints first. See scripts/convert_ckpt_fsdp_to_tp.py.
"""
for key in CTRL_HINT_KEYS_COMB.keys():
    if key in ["control_input_hdmap", "control_input_lidar"]:
        continue
    # Register experiments for pretraining from scratch
    config = make_ctrlnet_config_7b_training(
        hint_key=key, num_control_blocks=num_control_blocks, pretrain_model_path=""
    )
    cs.store(
        group="experiment",
        package="_global_",
        name=config["job"]["name"],
        node=config,
    )
    # Register experiments for post-training from TP checkpoints.
    hint_key_short = key.replace("control_input_", "")  # "control_input_vis" -> "vis"
    pretrain_ckpt_path = default_model_names[hint_key_short]
    # note: The TP ckpt path are specified as <name>.pt to the script, but actually the <name>_model_mp_*.pt files will be loaded.
    tp_ckpt_path = os.path.join(
        "checkpoints", os.path.dirname(pretrain_ckpt_path), "checkpoints_tp", os.path.basename(pretrain_ckpt_path)
    )
    config = make_ctrlnet_config_7b_training(
        hint_key=key, num_control_blocks=num_control_blocks, pretrain_model_path=tp_ckpt_path
    )
    cs.store(
        group="experiment",
        package="_global_",
        name=config["job"]["name"],
        node=config,
    )
