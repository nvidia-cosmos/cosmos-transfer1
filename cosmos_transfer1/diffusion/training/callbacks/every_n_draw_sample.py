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

import math
import os
from contextlib import nullcontext
from functools import partial
from typing import IO, Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
from einops import rearrange
from megatron.core import parallel_state
from PIL import Image as PILImage

from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.diffusion.training.callbacks.every_n import EveryN
from cosmos_transfer1.diffusion.training.utils.fsdp_helper import possible_fsdp_scope
from cosmos_transfer1.utils import distributed, log, misc
from cosmos_transfer1.utils.easy_io import easy_io
from cosmos_transfer1.utils.model import Model
from cosmos_transfer1.utils.parallel_state_helper import is_tp_cp_pp_rank0


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


def save_img_or_video(
    sample_C_T_H_W_in01: torch.Tensor,
    save_fp_wo_ext: Union[str, IO[Any]],
    fps: int = 24,
    quality=None,
    ffmpeg_params=None,
) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample_C_T_H_W_in01 (Tensor): Input tensor with shape (C, T, H, W) in [0, 1] range.
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample_C_T_H_W_in01.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(
        save_fp_wo_ext, "write"
    ), "save_fp_wo_ext must be a string or file-like object"

    if torch.is_floating_point(sample_C_T_H_W_in01):
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.clamp(0, 1)
    else:
        assert sample_C_T_H_W_in01.dtype == torch.uint8, "Only support uint8 tensor"
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.float().div(255)

    kwargs = {}
    if quality is not None:
        kwargs["quality"] = quality
    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if sample_C_T_H_W_in01.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85,
            **kwargs,
        )
    else:
        save_obj = rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
            **kwargs,
        )


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        is_x0: bool = True,
        is_sample: bool = True,
        save_verbose: bool = False,
        is_ema: bool = False,
        show_all_frames: bool = False,
        text_embedding_type: str = "t5_xxl",
    ):
        super().__init__(every_n, step_size)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_verbose = save_verbose
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.text_embedding_type = text_embedding_type
        self.show_all_frames = show_all_frames
        self.rank = distributed.get_rank()

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with misc.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = misc.co(easy_io.load(self.fix_batch), "cpu")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        if not model.is_image_batch(data_batch):
            if parallel_state.get_context_parallel_world_size() > 1:
                raw_data = split_inputs_cp(raw_data, seq_dim=2, cp_group=model.net.cp_group)
                x0 = split_inputs_cp(x0, seq_dim=2, cp_group=model.net.cp_group)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_x0_level + 1)[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )
        # check if it has control input
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            to_show.append(hint.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicaID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_verbose and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"{self.local_dir}/BatchInfo_ReplicaID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            log.debug("entering, ema", rank0_only=False)
            with possible_fsdp_scope(model.model):
                # we only use rank0 and rank to generate images and save
                # other rank run forward pass to make sure it works for FSDP
                log.debug("entering, fsdp", rank0_only=False)
                if self.is_x0:
                    log.debug("entering, x0_pred", rank0_only=False)
                    x0_img_fp, mse_loss, sigmas = self.x0_pred(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                    log.debug("done, x0_pred", rank0_only=False)
                    if self.save_verbose and self.rank == 0:
                        easy_io.dump(
                            {
                                "mse_loss": mse_loss.tolist(),
                                "sigmas": sigmas.tolist(),
                                "iteration": iteration,
                            },
                            f"{self.local_dir}/{tag}_MSE_Iter{iteration:09d}.json",
                        )
                if self.is_sample:
                    log.debug("entering, sample", rank0_only=False)
                    sample_img_fp = self.sample(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                    log.debug("done, sample", rank0_only=False)
                if self.fix_batch is not None:
                    misc.to(self.fix_batch, "cpu")

                log.debug("waiting for all ranks to finish", rank0_only=False)
                dist.barrier()
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch)

        to_show = []
        for guidance in [3.0, 7.0, 9.0, 13.0]:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                is_negative_prompt=False,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())
        # visualize input video
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            for idx in range(0, hint.size(1), 3):
                x_rgb = hint[:, idx : idx + 3]
                to_show.append(x_rgb.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicaID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = output_batch["x0"].shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]

        if self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"{self.local_dir}/{base_fp_wo_ext}",
            )
        return None
