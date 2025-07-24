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
from torch.nn import functional as F

from cosmos_transfer1.diffusion.diffusion.functional.batch_ops import batch_mul


def denoising_score_matching_loss(
    pred_type: str,
    perturbed_data: torch.Tensor,
    eps: torch.Tensor,
    pred_eps: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute the denoising diffusion objective.

    Args:
        pred_type (str): Prediction type, either 'eps' or 'x_0' supported.
        perturbed_data (torch.Tensor): The noised data x_t = f(x_0, t, eps).
        eps (torch.Tensor): The epsilon used to compute noised data.
        pred_eps (torch.Tensor): The output of the score model.
        sigma (torch.Tensor): The SDE sigma.

    Raises:
        NotImplementedError: If an unknown pred_type is used.

    Returns:
        loss (torch.Tensor): The denoising diffusion loss.
    """
    if pred_type == "eps":
        loss = F.mse_loss(eps, pred_eps, reduction="mean")
    elif pred_type == "x_0":
        x_data = perturbed_data - batch_mul(eps, sigma)
        x_pred = perturbed_data - batch_mul(pred_eps, sigma)
        loss = F.mse_loss(x_data, x_pred, reduction="mean")
    else:
        raise NotImplementedError(f"Unknown prediction type {pred_type}")
    return loss


def variational_score_distillation_loss(
    gen_data: torch.Tensor,
    teacher_eps: torch.Tensor,
    fake_score_eps: torch.Tensor,
    eps: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the variational score distillation loss.
    Args:
        gen_data (torch.Tensor): generated data
        teacher_eps (torch.Tensor): epsilon-prediction from the teacher
        fake_score_eps (torch.Tensor): epsilon-prediction from the fake score
        eps (torch.Tensor): The epsilon used to compute noised data.
        sigma (torch.Tensor): The SDE sigma.

    Returns:
        loss (torch.Tensor): The variational score distillation loss.
    """
    # Compute target for VSD gradient propagation
    teacher_score = -batch_mul(teacher_eps, 1.0 / sigma)
    fake_score = -batch_mul(fake_score_eps, 1.0 / sigma)
    # Compute the dims to reduce over
    dims = tuple(range(1, teacher_eps.ndim))
    # Note we include alpha_t here, from Eq 7 of the original DMD paper.
    w = batch_mul(1 / (eps - teacher_eps).abs().mean(dim=dims, keepdim=True), sigma)

    vsd_grad = fake_score - teacher_score
    vsd_target = (gen_data - vsd_grad * w).detach()

    vsd_loss = 0.5 * F.mse_loss(gen_data, vsd_target, reduction="mean")
    return vsd_loss


def gan_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the GAN loss for the generator
    Args:
        fake_logits (torch.Tensor): The logits for the fake data.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the generator.

    """

    gan_loss = F.softplus(-fake_logits).mean()
    return gan_loss


def gan_loss_discriminator(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the GAN loss for the discriminator
    Args:
        real_logits (torch.Tensor): The logits for the real data.
        fake_logits (torch.Tensor): The logits for the fake data.

    Returns:
        gan_loss (torch.Tensor): The GAN loss for the discriminator.
    """

    gan_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
    return gan_loss
