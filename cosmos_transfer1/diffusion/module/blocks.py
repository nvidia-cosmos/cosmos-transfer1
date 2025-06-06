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
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from cosmos_transfer1.diffusion.module.attention import Attention, GPT2FeedForward
from cosmos_transfer1.utils import log


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        in_dype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(in_dype)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            emb_B_D = emb
            adaln_lora_B_3D = None

        return emb_B_D, adaln_lora_B_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias
            ),
        )
        self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)

        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


class VideoAttn(nn.Module):
    """
    Implements video attention with optional cross-attention capabilities.

    This module processes video features while maintaining their spatio-temporal structure. It can perform
    self-attention within the video features or cross-attention with external context features.

    Parameters:
        x_dim (int): Dimension of input feature vectors
        context_dim (Optional[int]): Dimension of context features for cross-attention. None for self-attention
        num_heads (int): Number of attention heads
        bias (bool): Whether to include bias in attention projections. Default: False
        qkv_norm_mode (str): Normalization mode for query/key/value projections. Must be "per_head". Default: "per_head"
        x_format (str): Format of input tensor. Must be "BTHWD". Default: "THWBD"
        n_views (int): Extra parameter used in multi-view diffusion model. It indicated total number of camera we model together.
    Input shape:
        - x: (T, H, W, B, D) video features
        - context (optional): (M, B, D) context features for cross-attention
        where:
            T: temporal dimension
            H: height
            W: width
            B: batch size
            D: feature dimension
            M: context sequence length
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        bias: bool = False,
        qkv_norm_mode: str = "per_head",
        x_format: str = "THWBD",
        n_views: int = 1,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.x_format = x_format

        self.attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_bias=bias,
            qkv_norm="RRI",
            out_bias=bias,
            qkv_norm_mode=qkv_norm_mode,
            qkv_format="sbhd",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for video attention with regional prompting support.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D) representing batches of video data.
            context (Tensor): Context tensor of shape (B, M, D) or (M, B, D),
            where M is the sequence length of the context.
            crossattn_mask (Optional[Tensor]): An optional mask for cross-attention mechanisms.
            rope_emb_L_1_1_D (Optional[Tensor]):
            Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training.

        Returns:
            Tensor: The output tensor with applied attention, maintaining the input shape.
        """
        if context is not None and self.n_views > 1:
            x_T_H_W_B_D = rearrange(x, "(v t) h w b d -> t h w (v b) d", v=self.n_views)
            context_M_B_D = rearrange(context, "(v m) b d -> m (v b) d", v=self.n_views)
        else:
            x_T_H_W_B_D = x
            context_M_B_D = context
        T, H, W, B, D = x_T_H_W_B_D.shape
        x_THW_B_D = rearrange(x_T_H_W_B_D, "t h w b d -> (t h w) b d")
        if regional_contexts is not None:
            regional_contexts = rearrange(regional_contexts, "r (v m) b d -> r m (v b) d", v=1)
        if region_masks is not None:
            r, t, h, w, b = region_masks.shape
            region_masks = rearrange(region_masks, "r (v t) h w b -> r t h w (v b)", v=1)
        x_THW_B_D = self.attn(
            x_THW_B_D,
            context_M_B_D,
            crossattn_mask,
            rope_emb=rope_emb_L_1_1_D,
            regional_contexts=regional_contexts,
            region_masks=region_masks,
        )
        x_T_H_W_B_D = rearrange(x_THW_B_D, "(t h w) b d -> t h w b d", h=H, w=W)
        if context is not None and self.n_views > 1:
            x_T_H_W_B_D = rearrange(x_T_H_W_B_D, "t h w (v b) d -> (v t) h w b d", v=self.n_views)
        return x_T_H_W_B_D


def adaln_norm_state(norm_state, x, scale, shift):
    normalized = norm_state(x)
    return normalized * (1 + scale) + shift


class DITBuildingBlock(nn.Module):
    """
    A building block for the DiT (Diffusion Transformer) architecture that supports different types of
    attention and MLP operations with adaptive layer normalization.

    Parameters:
        block_type (str): Type of block - one of:
            - "cross_attn"/"ca": Cross-attention
            - "full_attn"/"fa": Full self-attention
            - "mlp"/"ff": MLP/feedforward block
        x_dim (int): Dimension of input features
        context_dim (Optional[int]): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dimension multiplier. Default: 4.0
        bias (bool): Whether to use bias in layers. Default: False
        mlp_dropout (float): Dropout rate for MLP. Default: 0.0
        qkv_norm_mode (str): QKV normalization mode. Default: "per_head"
        x_format (str): Input tensor format. Default: "THWBD"
        use_adaln_lora (bool): Whether to use AdaLN-LoRA. Default: False
        adaln_lora_dim (int): Dimension for AdaLN-LoRA. Default: 256
        n_views (int): Extra parameter used in multi-view diffusion model. It indicated total number of camera we model together.
    """

    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        mlp_dropout: float = 0.0,
        qkv_norm_mode: str = "per_head",
        x_format: str = "THWBD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        n_views: int = 1,
    ) -> None:
        block_type = block_type.lower()

        super().__init__()
        self.x_format = x_format
        if block_type in ["cross_attn", "ca"]:
            self.block = VideoAttn(
                x_dim,
                context_dim,
                num_heads,
                bias=bias,
                qkv_norm_mode=qkv_norm_mode,
                x_format=self.x_format,
                n_views=n_views,
            )
        elif block_type in ["full_attn", "fa"]:
            self.block = VideoAttn(
                x_dim, None, num_heads, bias=bias, qkv_norm_mode=qkv_norm_mode, x_format=self.x_format
            )
        elif block_type in ["mlp", "ff"]:
            self.block = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora

        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False))

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for dynamically configured blocks with adaptive normalization.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D). Current only support (T, H, W, B, D).
            emb_B_D (Tensor): Embedding tensor for adaptive layer normalization modulation.
            crossattn_emb (Tensor): Tensor for cross-attention blocks.
            crossattn_mask (Optional[Tensor]): Optional mask for cross-attention.
            rope_emb_L_1_1_D (Optional[Tensor]): Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training.
            adaln_lora_B_3D (Optional[Tensor]): Additional embedding for adaptive layer norm.
            regional_contexts (Optional[List[Tensor]]): List of regional context tensors.
            region_masks (Optional[Tensor]): Region masks of shape (B, R, THW).

        Returns:
            Tensor: The output tensor after processing through the configured block and adaptive normalization.
        """
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_1_1_1_B_D, scale_1_1_1_B_D, gate_1_1_1_B_D = (
            shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        if self.block_type in ["mlp", "ff"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
            )
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=None,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            )
        elif self.block_type in ["cross_attn", "ca"]:
            normalized_x = adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D)
            x = x + gate_1_1_1_B_D * self.block(
                normalized_x,
                context=crossattn_emb,
                crossattn_mask=crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                regional_contexts=regional_contexts,
                region_masks=region_masks,
            )
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        return x


class GeneralDITTransformerBlock(nn.Module):
    """
    A wrapper module that manages a sequence of DITBuildingBlocks to form a complete transformer layer.
    Each block in the sequence is specified by a block configuration string.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention blocks
        num_heads (int): Number of attention heads
        block_config (str): String specifying block sequence (e.g. "ca-fa-mlp" for cross-attention,
                          full-attention, then MLP)
        mlp_ratio (float): MLP hidden dimension multiplier. Default: 4.0
        x_format (str): Input tensor format. Default: "THWBD"
        use_adaln_lora (bool): Whether to use AdaLN-LoRA. Default: False
        adaln_lora_dim (int): Dimension for AdaLN-LoRA. Default: 256

    The block_config string uses "-" to separate block types:
        - "ca"/"cross_attn": Cross-attention block
        - "fa"/"full_attn": Full self-attention block
        - "mlp"/"ff": MLP/feedforward block

    Example:
        block_config = "ca-fa-mlp" creates a sequence of:
        1. Cross-attention block
        2. Full self-attention block
        3. MLP block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str,
        mlp_ratio: float = 4.0,
        x_format: str = "THWBD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        n_views: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.x_format = x_format
        for block_type in block_config.split("-"):
            self.blocks.append(
                DITBuildingBlock(
                    block_type,
                    x_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    x_format=self.x_format,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    n_views=n_views,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                regional_contexts=regional_contexts,
                region_masks=region_masks,
            )
        return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
