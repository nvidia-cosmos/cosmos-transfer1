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
Discriminator prediction model built upon features extracted from GeneralDIT
"""
from typing import List, Set

import torch
from einops import rearrange
from torch import nn

from cosmos_transfer1.utils import log


class DiscriminatorPredHead3D(nn.Module):
    """A simple discriminator prediction head for video feature processing.

    This class should be used with the `Discriminator` class, that
    builds a discriminator acting on the features from a base video diffusion model.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=hidden_channels,
                stride=stride,
                padding=padding,
            ),  # Default: 2x16x16 -> 1x8x8
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.LeakyReLU(),
            nn.Conv3d(kernel_size=1, in_channels=hidden_channels, out_channels=1, stride=1, padding=0),
            # Default: 1x1x1 -> 1x1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


class DiscriminatorPredHead2D(nn.Module):
    """A discriminator prediction head using 2D convolutions for video feature processing."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Process each temporal frame: (B, C, T, H, W) -> (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Apply convolutions: (B*T, C, H, W) -> (B*T, 1, H', W')
        x = self.net(x)

        # Restore temporal dimension: (B*T, 1, H', W') -> (B, 1, T, H', W')
        _, CH, H, W = x.shape
        x = x.reshape(B, T, CH, H, W).permute(0, 2, 1, 3, 4)

        return x


class DiscriminatorPredHeadAttention(nn.Module):
    """Self-attention based discriminator head using PyTorch's MultiheadAttention.

    Args:
        in_channels (int): Input channels (corresponds to in_ch in Discriminator)
        hidden_channels (int): Number of hidden channels (corresponds to nc in Discriminator)
        num_heads (int, optional): Number of attention heads. Default: 8
        dropout (float, optional): Dropout rate. Default: 0.0
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Sequential(
            # Downsample spatial dimensions early
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.LeakyReLU(),
        )

        # Using PyTorch's MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

        # Final layers to get to desired output shape
        self.final = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(hidden_channels),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Spatial downsampling in input projection
        x = self.input_proj(x)  # Reduces spatial dimensions by 2x
        _, _, _, H, W = x.shape

        # Reshape for attention
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, T * H * W, -1)

        # Apply attention with normalized input
        x = self.norm(x)
        x, _ = self.attention(x, x, x)

        # Reshape back
        x = x.reshape(B, T, H, W, -1)
        x = x.permute(0, 4, 1, 2, 3)

        # Final processing
        x = self.final(x).contiguous()

        return x


class DiscriminatorPredHeadSpatialTemporal(nn.Module):
    """
    This is inspired by the SF-V paper (https://arxiv.org/pdf/2406.04324),
    where we employ separate spatial and temporal discriminator heads.

    Spatial head applies Conv2D to capture the spatial features,
    while temporal head applies Conv1D to capture the temporal features.

    The discriminator can be conditioned on additional information via projection to enhance performance (TBD).
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.net_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        self.net_temporal = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Process each temporal frame: (B, C, T, H, W) -> (B*T, C, H, W)
        x_spatial = rearrange(x, "b c t h w -> (b t) c h w")
        # Apply convolutions: (B*T, C, H, W) -> (B*T, 1, H', W') -> (B*T, 1, 1, 1) -> (B, 1, T, 1, 1)
        x_spatial = self.net_spatial(x_spatial).mean(dim=(-2, -1), keepdim=True)
        x_spatial = rearrange(x_spatial, "(b t) c h w -> b c t h w", b=B)

        # Process each temporal frame: (B, C, T, H, W) -> (B*H*W, C, T)
        x_temporal = rearrange(x, "b c t h w -> (b h w) c t")
        # Apply convolutions: (B*H*W, C, T) -> (B*H*W, 1, T') -> (B*H*W, 1, 1) -> (B, 1, 1, H, W)
        x_temporal = self.net_temporal(x_temporal).mean(dim=-1, keepdim=True)
        x_temporal = rearrange(x_temporal, "(b h w) c t -> b c t h w", b=B, h=H)

        return (x_spatial + x_temporal) * 0.5


class Discriminator(nn.Module):
    """
    A lightweight discriminator prediction head built upon features from DiT base model.

    Args:
        feature_indices (Set[int]): List of indices of features extracted from DiT base model.
        feature_t_min (int): Minimum temporal index of features to consider.
        in_ch (int): Input channels (corresponds to feature dimension in DiT base model)
        nc (int): Number of hidden channels (corresponds to feature dimension in Discriminator)
        block_type (str): Type of block. Options are
            'conv3d': DiscriminatorPredHead3D,
            'conv2d': DiscriminatorPredHead2D,
            'attention': DiscriminatorPredHeadAttention
            'spatial_temporal': DiscriminatorPredHeadSpatialTemporal
        num_blocks (int, optional): Number of blocks in the DiT base model. `feature_indices` should be less than it.
    """

    def __init__(
        self,
        feature_indices: Set[int] = None,
        feature_t_min: int = 0,
        in_ch: int = 192,
        nc: int = 320,
        block_type: str = "conv3d",
        num_blocks: int = 28,
        **head_kwargs,
    ):
        super().__init__()
        if feature_indices is None:
            feature_indices = {int(num_blocks // 2)}  # use the middle bottleneck feature
        self.feature_indices = {i for i in feature_indices if i < num_blocks}  # make sure feature indices are valid
        self.feature_t_min = feature_t_min
        self.num_features = len(self.feature_indices)
        self.block_type = block_type
        log.info(f"discriminator block_type: {block_type}")

        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            if block_type == "conv3d":
                cls_pred_head = DiscriminatorPredHead3D(in_ch, nc, **head_kwargs)
            elif block_type == "conv2d":
                cls_pred_head = DiscriminatorPredHead2D(in_ch, nc)
            elif block_type == "attention":
                cls_pred_head = DiscriminatorPredHeadAttention(in_ch, nc, **head_kwargs)
            elif block_type == "spatial_temporal":
                cls_pred_head = DiscriminatorPredHeadSpatialTemporal(in_ch, nc)
            else:
                raise ValueError(f"Unknown block_type: {block_type}. Must be one of: conv3d, conv2d, attention")
            self.cls_pred_heads.append(cls_pred_head)

    @property
    def fsdp_wrap_block_cls(self):
        if self.block_type == "conv3d":
            return DiscriminatorPredHead3D
        elif self.block_type == "conv2d":
            return DiscriminatorPredHead2D
        elif self.block_type == "attention":
            return DiscriminatorPredHeadAttention
        elif self.block_type == "spatial_temporal":
            return DiscriminatorPredHeadSpatialTemporal
        else:
            raise ValueError(f"Unknown block_type: {self.block_type}. Must be one of: conv3d, conv2d, attention")

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the discriminator logits from the teacher features.

        Takes in a list of teacher features, and returns a tensor containing discriminator
        logits for each teacher feature.

        Args:
            feats (List[torch.Tensor]): List of teacher features.

        Returns:
            torch.Tensor: The discriminator logits, with the output of
                each discriminator head concatenated along the first dimension.
        """
        assert isinstance(feats, list) and len(feats) == self.num_features
        all_logits = []

        for cls_pred_head, feat in zip(self.cls_pred_heads, feats):
            feat = rearrange(feat, "b t h w c -> b c t h w")[:, :, self.feature_t_min :]
            # we perform average pooling over temporal and spatial dimension if necessary
            logits = cls_pred_head(feat).mean(dim=(-3, -2, -1), keepdim=True)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=1)
        return all_logits
