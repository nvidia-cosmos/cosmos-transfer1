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

from typing import List, Optional

import numpy as np
import torch
import transformer_engine as te
from einops import rearrange
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb

# ---------------------- Feed Forward Network -----------------------


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        assert self.dropout.p == 0.0, "we skip dropout"

        x = self.layer1(x)

        def activation_layer2_forward(x):
            x = self.activation(x)
            x = self.layer2(x)
            return x

        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


# ---------------------- Normalization Layer -----------------------


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def get_normalization(name: str, channels: int):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return te.pytorch.RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()

class RegionalAttentionOp(BaseAttentionOp):
    def __init__(
        self,
        heads,
        dim_head,
        num_gqa_groups=None,
        attention_dropout=0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
        tp_size=1,
        tp_group=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.qkv_format = qkv_format
        self.tp_size = tp_size
        self.scale = dim_head**-0.5
        self.attention_dropout = attention_dropout
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        self.dot_product_attention = DotProductAttention(
            self.heads,
            self.dim_head,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
            tp_size=tp_size,
            tp_group=tp_group,
            sequence_parallel=sequence_parallel,
        )

    def forward(
        self,
        q,
        k,
        v,
        regional_k=None,
        regional_v=None,
        region_masks=None,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
    ):
        # Early return for non-regional case
        if regional_k is None or regional_v is None or region_masks is None:
            return self.dot_product_attention(
                q,
                k,
                v,
                attention_mask=None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
            )
        # Get dimensions
        is_bshd = self.qkv_format == "bshd"
        if is_bshd:
            batch_size, seq_len, num_heads, head_dim = q.shape
        else:
            seq_len, batch_size, num_heads, head_dim = q.shape

        # Process region masks
        processed_masks = []
        prompt_len = k.shape[1] if is_bshd else k.shape[0]
        num_regions = len(regional_k)

        def preprocess_mask(mask: Tensor) -> Tensor:
            mask = mask.permute(3, 0, 1, 2)
            B, T, H, W = mask.shape
            mask = mask.unsqueeze(1)  # dummy unsqueeze since trilinear interpolation expects 5D

            mask_i = [
                torch.nn.functional.interpolate(
                    mask[:, :, :1, :, :],
                    size=(1, 44, 80),
                    mode="trilinear",
                    align_corners=False,
                )
            ]
            for wi in range(1, T, 8):
                mask_i += [
                    torch.nn.functional.interpolate(
                        mask[:, :, wi : wi + 8, :, :],
                        size=(1, 44, 80),
                        mode="trilinear",
                        align_corners=False,
                    )
                ]
            assert len(mask_i) == 16
            mask = torch.cat(mask_i, dim=2)
            mask = mask.squeeze(1)
            return (mask > 0.5).float()

        for i in range(num_regions):
            mask = region_masks[i]
            mask = mask.to(q.device)
            if mask.shape[0] != seq_len:
                mask = preprocess_mask(mask)
                mask = rearrange(mask, "b t h w ->  b (t h w)")
            processed_masks.append(mask)

        hidden_seq_len = seq_len
        regional_attention_mask = torch.zeros(
            (batch_size, hidden_seq_len, (num_regions + 1) * prompt_len), device=q.device, dtype=torch.bool
        )
        start_idx = 0
        for i, mask in enumerate(processed_masks):
            regional_attention_mask[:, :, (i + 1) * prompt_len : (i + 2) * prompt_len] = mask.unsqueeze(-1).bool()

        regional_masks_tensor = torch.stack(processed_masks, dim=-1).bool()  # [B, S, R]
        global_mask = (regional_masks_tensor.sum(dim=-1) == 0).unsqueeze(-1).bool()  # [B, S, 1]
        regional_attention_mask[:, :, :prompt_len] = global_mask
        combined_k = torch.cat([k] + regional_k, dim=0)
        combined_v = torch.cat([v] + regional_v, dim=0)

        attn_bias = torch.zeros_like(regional_attention_mask, dtype=torch.float32)
        attn_bias = attn_bias.masked_fill(~regional_attention_mask, float("-inf"))
        attn_bias = attn_bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
        output = self.dot_product_attention(
            q,
            combined_k,
            combined_v,
            attention_mask=None,
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=attn_bias,
        )

        base_ratio = 0.5  # signifies the weight of the global prompt
        if base_ratio is not None:
            base_output = self.dot_product_attention(
                q,
                k,
                v,
                attention_mask=None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
            )
            output = output * (1 - base_ratio) + base_output * base_ratio

        if self.tp_size > 1 and not self.sequence_parallel:
            torch.distributed.all_reduce(output, group=self.tp_group)

        return output

class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None, self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for query, key, and value projections.
                                  Defaults to "SSI".
        qkv_norm_mode (str, optional): A string representing normalization mode for query, key, and value projections.
                                        Defaults to 'per_head'. Only support 'per_head'.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "sbhd",
    ) -> None:
        super().__init__()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        self.backend = backend

        self.to_q = nn.Sequential(
            nn.Linear(query_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[0], norm_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[1], norm_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(context_dim, inner_dim, bias=qkv_bias),
            get_normalization(qkv_norm[2], norm_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

        if attn_op:  # use what is given
            self.attn_op = attn_op
        elif self.backend == "transformer_engine":
            self.attn_op: BaseAttentionOp = DotProductAttention(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
                sequence_parallel=False,
            )
            self.regional_attn_op = RegionalAttentionOp(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="arbitrary"
            )
        else:
            raise ValueError(f"Backend {backend} not found")

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs

        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        We use a flag `self.qkv_norm_mode` to control the normalization behavior.
        The default value of `self.qkv_norm_mode` is "per_head", which means we normalize per head.
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            q, k, v = map(
                lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads, c=self.dim_head),
                (q, k, v),
            )
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        if self.backend == "transformer_engine":
            seq_dim = self.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
            return self.to_out(out)
        else:
            raise ValueError(f"Backend {self.backend} not found")

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        regional_contexts=None,
        region_masks=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            regional_contexts (Optional[Tensor]): Stacked regional context tensors [B, R, M, D] or [R, M, B, D] if THWBD format
            region_masks (Optional[Tensor]): Region masks [B, R, S] or [R, S, B] if THWBD format
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)

        # Early return if no regional contexts
        if regional_contexts is None or region_masks is None:
            return self.cal_attn(q, k, v, mask)

        # Process regional contexts
        regional_k = []
        regional_v = []

        # Determine format based on qkv_format
        is_bshd = self.qkv_format == "bshd"

        # Get number of regions
        num_regions = regional_contexts.shape[1] if is_bshd else regional_contexts.shape[0]

        # Process each region
        for i in range(num_regions):
            # Extract regional context
            reg_context = regional_contexts[:, i] if is_bshd else regional_contexts[i]

            # Ensure correct dtype
            if reg_context.dtype != context.dtype:
                reg_context = reg_context.to(dtype=context.dtype)

            _, k_regional, v_regional = self.cal_qkv(x, reg_context, mask, rope_emb=rope_emb, **kwargs)

            regional_k.append(k_regional)
            regional_v.append(v_regional)

        # Apply regional attention
        combined_attn = self.regional_attn_op(
            q,
            k,  # from global prompt
            v,  # from global prompt
            regional_k=regional_k,
            regional_v=regional_v,
            region_masks=region_masks,
            core_attention_bias_type="no_bias",
            core_attention_bias=None,
        )

        # Apply output projection
        return self.to_out(combined_attn)
