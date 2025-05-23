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
Usage:
    - [real run, 8 gpu] torchrun --nproc_per_node=8 -m projects.edify_image.v4.train --dryrun --config=projects/edify_video/v4/config/ctrl/config.py -- experiment=CTRL_tp_121frames_control_input_bbox_image_block3
    - [debug small model, 1 gpu] torchrun --nproc_per_node=8 -m projects.edify_image.v4.train --config=projects/edify_video/v4/config/ctrl/config.py -- experiment=CTRL_tp_121frames_control_input_bbox_image_block3  model.net.num_blocks=1 model.context_parallel_size=1 checkpoint.load_path="" job.group=debug trainer.logging_iter=5
"""

import copy
import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_transfer1.checkpoints import COSMOS_TRANSFER1_7B_CHECKPOINT, COSMOS_TRANSFER1_7B_SAMPLE_AV_CHECKPOINT
from cosmos_transfer1.diffusion.config.base.data import get_sampler
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS_COMB
from cosmos_transfer1.diffusion.datasets.example_transfer_dataset import AVTransferDataset, ExampleTransferDataset
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names
from cosmos_transfer1.diffusion.training.models.model_ctrl import (  # this one has training support
    ShortVideoDiffusionModelWithCtrl,
    VideoDiffusionModelWithCtrl,
)
from cosmos_transfer1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_transfer1.diffusion.training.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_transfer1.utils.lazy_config import LazyCall as L
from cosmos_transfer1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

num_blocks = 28
num_control_blocks = 3
ckpt_root = "checkpoints/"
data_root = "datasets/waymo_transfer1/"


def make_ctrlnet_config(
    hint_key: str = "control_input_hdmap",
    num_control_blocks: int = 3,
    pretrain_model_path: str = "",
    t2w: bool = True,
    num_frames=121,
) -> LazyDict:
    if pretrain_model_path == "":
        if t2w:
            job_name = f"CTRL_7Bv1pt3_t2w_{num_frames}frames_{hint_key}_block{num_control_blocks}_pretrain"
            job_project = "cosmos_transfer1_pretrain"
        else:
            job_name = f"CTRL_7Bv1pt3_lvg_{num_frames}frames_{hint_key}_block{num_control_blocks}_pretrain"
            job_project = "cosmos_transfer1_pretrain"
    else:
        if t2w:
            job_name = f"CTRL_7Bv1pt3_t2w_{num_frames}frames_{hint_key}_block{num_control_blocks}_posttrain"
            job_project = "cosmos_transfer1_posttrain"
        else:
            job_name = f"CTRL_7Bv1pt3_lvg_{num_frames}frames_{hint_key}_block{num_control_blocks}_posttrain"
            job_project = "cosmos_transfer1_posttrain"
    example_multiview_dataset_waymo = L(AVTransferDataset)(
        dataset_dir=data_root,
        num_frames=num_frames,
        hint_key=hint_key,
        resolution="720",
        view_keys=["front"],
        sample_n_views=-1,
        load_mv_emb=False,
    )

    ctrl_config = LazyDict(
        dict(
            defaults=[
                {"override /net": "faditv2_7b"},
                {"override /net_ctrl": "faditv2_7b"},
                {"override /conditioner": "ctrlnet_add_fps_image_size_padding_mask"},
                {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
                {"override /hint_key": hint_key},
                {"override /callbacks": "basic"},
                {"override /checkpoint": "local"},
                {"override /ckpt_klass": "fast_tp"},
                "_self_",
            ],
            job=dict(group="CTRL_7Bv1_sampleAV", project=job_project, name=job_name),
            optimizer=dict(
                lr=2 ** (-14.3),  # ~5e-5
                weight_decay=0.1,
                betas=[0.9, 0.99],
                eps=1e-10,
            ),
            checkpoint=dict(
                load_path=pretrain_model_path,
                # Modify load_path as needed if you do post-training (fine-tuning). If training from scratch, leave it empty.
                broadcast_via_filesystem=True,
                save_iter=1000,
                load_training_state=False,
                strict_resume=False,
                keys_not_to_resume=[],
            ),
            trainer=dict(
                distributed_parallelism="ddp",
                logging_iter=200,
                max_iter=999_999_999,
                callbacks=dict(
                    iter_speed=dict(hit_thres=5),
                ),
                timestamp_seed=True,  # important for dataver dataloader!!!
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
                    (num_frames - 1) // 8 + 1,
                    88,
                    160,
                ],
                base_load_from=dict(
                    load_path=os.path.join(
                        ckpt_root,
                        COSMOS_TRANSFER1_7B_SAMPLE_AV_CHECKPOINT,
                        "checkpoints_tp",
                        "base_model_model_mp_*.pt",
                    )
                ),
                finetune_base_model=False,
                hint_mask=[True],
                hint_dropout_rate=0.15,
                conditioner=dict(
                    video_cond_bool=dict(
                        condition_location="first_random_n",
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
                net=dict(
                    in_channels=16,
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                adjust_video_noise=True,
                net_ctrl=dict(
                    in_channels=16,
                    hint_channels=16,
                    num_blocks=num_blocks,
                    layer_mask=[True if (i >= num_control_blocks) else False for i in range(num_blocks)],
                    extra_per_block_abs_pos_emb=True,
                    pos_emb_learnable=True,
                    extra_per_block_abs_pos_emb_type="learnable",
                ),
                tokenizer=dict(
                    pixel_chunk_duration=num_frames,
                ),
            ),
            model_obj=L(ShortVideoDiffusionModelWithCtrl)(),
            dataloader_train=L(DataLoader)(
                dataset=example_multiview_dataset_waymo,
                sampler=L(get_sampler)(dataset=example_multiview_dataset_waymo),
                batch_size=1,
                drop_last=True,
                pin_memory=True,
                num_workers=8,
            ),
            dataloader_val=L(DataLoader)(
                dataset=example_multiview_dataset_waymo,
                sampler=L(get_sampler)(dataset=example_multiview_dataset_waymo),
                batch_size=1,
                drop_last=True,
                pin_memory=True,
                num_workers=8,
            ),
        )
    )
    return ctrl_config


all_hint_key = [
    "control_input_hdmap",
    "control_input_lidar",
]

for key in all_hint_key:
    for num_frames in [57, 121]:
        # Register experiments for pretraining from scratch
        t2w_config = make_ctrlnet_config(
            hint_key=key, num_control_blocks=num_control_blocks, pretrain_model_path="", t2w=True, num_frames=num_frames
        )
        cs.store(
            group="experiment",
            package="_global_",
            name=t2w_config["job"]["name"],
            node=t2w_config,
        )

        # Register experiments for post-training from TP checkpoints.
        hint_key_short = key.replace("control_input_", "")  # "control_input_vis" -> "vis"
        pretrain_ckpt_path = default_model_names[hint_key_short]
        # note: The TP ckpt path are specified as <name>.pt to the script, but actually the <name>_model_mp_*.pt files will be loaded.
        tp_ckpt_path = os.path.join(
            ckpt_root, os.path.dirname(pretrain_ckpt_path), "checkpoints_tp", os.path.basename(pretrain_ckpt_path)
        )
        config = make_ctrlnet_config(
            hint_key=key,
            num_control_blocks=num_control_blocks,
            pretrain_model_path=tp_ckpt_path,
            t2w=True,
            num_frames=num_frames,
        )
        cs.store(
            group="experiment",
            package="_global_",
            name=config["job"]["name"],
            node=config,
        )
