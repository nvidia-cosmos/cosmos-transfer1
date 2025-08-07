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

from enum import Enum

import torch
from einops import rearrange
from torch import Tensor

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

IS_PREPROCESSED_KEY = "is_preprocessed"


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"


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
        self.input_image_key = self.config.input_image_key

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


class DistillT2WModel(DiffusionT2WModel):
    """Base Video Distillation Model."""

    def __init__(self, config):
        super().__init__(config)

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
                ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = parallel.broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition
