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

import torch
from einops import rearrange

from cosmos_transfer1.diffusion.inference.inference_utils import switch_config_for_inference
from cosmos_transfer1.diffusion.training.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_transfer1.utils import misc
from cosmos_transfer1.utils.parallel_state_helper import is_tp_cp_pp_rank0


class EveryNDrawSampleDistillation(EveryNDrawSample):
    def __init__(
        self,
        # Base arguments
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: str | None = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        is_x0: bool = True,
        is_sample: bool = True,
        save_verbose: bool = False,
        is_ema: bool = False,
        show_all_frames: bool = False,
        text_embedding_type: str = "t5_xxl",
        # Custom arguments
        output_key: str = "x0",
        include_teacher: bool = False,
        num_of_latent_overlap: int = 1,
    ):
        """
        Args:
            output_key (str): dictionary key to extract the generated samples out of the output data batch
            include_teacher (bool): generate and include video samples using the Teacher model
        """
        super().__init__(
            every_n,
            step_size,
            fix_batch_fp,
            n_x0_level,
            n_viz_sample,
            n_sample_to_save,
            is_x0,
            is_sample,
            save_verbose,
            is_ema,
            show_all_frames,
            text_embedding_type,
        )
        self.output_key = output_key
        self.include_teacher = include_teacher
        self.num_of_latent_overlap = num_of_latent_overlap

    def _generate_sample(self, model, data_batch, x0, use_teacher: bool, seed: int = 0) -> torch.Tensor:
        """Generate samples in inference mode using either Student or Teacher models"""
        with switch_config_for_inference(model):
            from cosmos_transfer1.diffusion.inference.inference_utils import generate_video_from_batch_with_loop

            sample_numpy, _, _ = generate_video_from_batch_with_loop(
                model=model,
                state_shape=x0.shape[1:],
                data_batch=data_batch,
                condition_latent=x0,
                is_negative_prompt=False,
                guidance=0.0,  # Unused
                num_of_loops=1,
                num_of_latent_overlap_list=[self.num_of_latent_overlap],
                num_steps=1,
                seed=seed,
                use_teacher=use_teacher,
            )
        sample = rearrange(torch.from_numpy(sample_numpy), "(1 t) h w c -> 1 c t h w") / 128.0 - 1.0
        return sample

    @misc.timer("EveryNDrawSampleDistill: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """Run inference on conditions provided by the training data batch"""
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, _ = model.get_data_and_condition(data_batch)

        # Generate sample with Student model
        sample_student = self._generate_sample(model, data_batch, x0, use_teacher=False)
        to_show = [sample_student.float().cpu()]

        if self.include_teacher:
            # Generate sample with Teacher model
            sample_teacher = self._generate_sample(model, data_batch, x0, use_teacher=True)
            to_show.append(sample_teacher.float().cpu())

        if "hint_key" in data_batch:
            # Include hint next
            hint = data_batch[data_batch["hint_key"]]
            for idx in range(0, hint.size(1), 3):
                x_rgb = hint[:, idx : idx + 3]
                to_show.append(x_rgb.float().cpu())

        # Include ground truth as bottom row
        raw_data = raw_data.float().cpu()
        to_show.append(raw_data)

        base_fp_wo_ext = f"{tag}_ReplicaID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = output_batch[self.output_key].shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path
        return None
