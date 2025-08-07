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

from cosmos_transfer1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_transfer1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_transfer1.distillation.callbacks.every_n_draw_sample import EveryNDrawSampleDistillation
from cosmos_transfer1.distillation.callbacks.grad_clip import GradClip as GradClipDistillation
from cosmos_transfer1.utils.callback import ProgressBarCallback
from cosmos_transfer1.utils.lazy_config import PLACEHOLDER
from cosmos_transfer1.utils.lazy_config import LazyCall as L

BASIC_CALLBACKS = dict(
    progress_bar=L(ProgressBarCallback)(),
    grad_clip=L(GradClipDistillation)(fsdp_enabled=True, model_key="net"),
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    # for the first 1000 iterations, log the iteration speed per iteration, after that, log every 200 iterations
    iter_speed=L(IterSpeed)(every_n=200, hit_thres=1000),
)

TRAIN_VIS_CALLBACK = dict(
    train_sample=L(EveryNDrawSampleDistillation)(
        output_key="gen_rand",
        is_x0=False,
        every_n=100,
        include_teacher=True,
        num_of_latent_overlap=1,
        n_sample_to_save=1,
    )
)
