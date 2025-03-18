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

from cosmos_transfer1.diffusion.conditioner import BaseVideoCondition, CosmosCondition
from cosmos_transfer1.diffusion.diffusion.functional.batch_ops import batch_mul
from cosmos_transfer1.diffusion.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos_transfer1.diffusion.diffusion.modules.res_sampler import Sampler
from cosmos_transfer1.diffusion.diffusion.types import DenoisePrediction
from cosmos_transfer1.diffusion.module import parallel
from cosmos_transfer1.diffusion.module.blocks import FourierFeatures
from cosmos_transfer1.diffusion.module.pretrained_vae import BaseVAE
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate


class EDMSDE:
    def __init__(
        self,
        sigma_max: float,
        sigma_min: float,
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min


class DiffusionT2WModel(torch.nn.Module):
    """Text-to-world diffusion model that generates video frames from text descriptions.

    This model implements a diffusion-based approach for generating videos conditioned on text input.
    It handles the full pipeline including encoding/decoding through a VAE, diffusion sampling,
    and classifier-free guidance.
    """

    def __init__(self, config):
        """Initialize the diffusion model.

        Args:
            config: Configuration object containing model parameters and architecture settings
        """
        super().__init__()
        # Initialize trained_data_record with defaultdict, key: image, video, iteration
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.debug(f"DiffusionModel: precision {self.precision}")
        # Timer passed to network to detect slow ranks.
        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.state_shape = list(config.latent_shape)
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.sde = EDMSDE(sigma_max=80, sigma_min=0.0002)
        self.sampler = Sampler()
        self.scaling = EDMScaling(self.sigma_data)
        self.tokenizer = None
        self.model = None

    @property
    def net(self):
        return self.model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    @property
    def logvar(self):
        return self.model.logvar

    def set_up_tokenizer(self, tokenizer_dir: str):
        self.tokenizer: BaseVAE = lazy_instantiate(self.config.tokenizer)
        self.tokenizer.load_weights(tokenizer_dir)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self, memory_format: torch.memory_format = torch.preserve_format):
        """Initialize the core model components including network, conditioner and logvar."""
        self.model = self.build_model()
        self.model = self.model.to(memory_format=memory_format, **self.tensor_kwargs)

    def build_model(self) -> torch.nn.ModuleDict:
        """Construct the model's neural network components.

        Returns:
            ModuleDict containing the network, conditioner and logvar components
        """
        config = self.config
        net = lazy_instantiate(config.net)
        conditioner = lazy_instantiate(config.conditioner)
        logvar = torch.nn.Sequential(
            FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
        )

        return torch.nn.ModuleDict(
            {
                "net": net,
                "conditioner": conditioner,
                "logvar": logvar,
            }
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input state into latent representation using VAE.

        Args:
            state: Input tensor to encode

        Returns:
            Encoded latent representation scaled by sigma_data
        """
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to pixel space using VAE.

        Args:
            latent: Latent tensor to decode

        Returns:
            Decoded tensor in pixel space
        """
        return self.tokenizer.decode(latent / self.sigma_data)

    def setup_data_key(self) -> None:
        """Configure input data keys for video and image data."""
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (CosmosCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        # forward pass through the network
        net_output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )

        logvar = self.model.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        return DenoisePrediction(x0_pred, eps_pred, logvar)


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = parallel.broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition
