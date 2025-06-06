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

from contextlib import nullcontext
from typing import List, Optional, Union

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

import numpy as np
import torch
import torch.nn.functional as F
import transformer_engine as te
from einops import rearrange
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

from cosmos_transfer1.utils import log

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

        tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=False)
        sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
        if tp_group is None:
            tp_size = 1  # TP is not initialized.
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if tp_size == 1:
            self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
            self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        else:
            assert is_gated is False, "is_gated needs to be False to support Tensor Parallelism"
            assert dropout == 0.0, "dropout needs to be 0.0 to support Tensor Parallelism"
            self.layer1 = te.pytorch.Linear(
                d_model,
                d_ff,
                bias=bias,
                tp_size=tp_size,
                tp_group=tp_group,
                parallel_mode="column",
                sequence_parallel=sequence_parallel,
            )
            self.layer2 = te.pytorch.Linear(
                d_ff,
                d_model,
                bias=bias,
                tp_size=tp_size,
                tp_group=tp_group,
                parallel_mode="row",
                sequence_parallel=sequence_parallel,
            )

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

        assert self.dropout.p == 0.0, "skipping dropout to save memory"
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


# ---------------------- Attention Op -----------------------
# A list of attention ops
if version.parse(torch.__version__) >= version.parse("2.3.0"):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sdpa_context = sdpa_kernel
    USE_SDPA = True
elif version.parse(torch.__version__) >= version.parse("2.0.0"):
    from torch.backends.cuda import SDPBackend, sdp_kernel

    sdpa_context = sdp_kernel
    USE_SDPA = False
else:
    sdpa_context = nullcontext
    USE_SDPA = False
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


class FusedAttentionOp(BaseAttentionOp):
    def __init__(self):
        super().__init__()


class TorchAttentionOp(FusedAttentionOp):
    def __init__(self, backend: Optional[Union[List[SDPBackend], SDPBackend]] = SDPBackend.EFFICIENT_ATTENTION):
        super().__init__()
        self.backend = backend
        self.sdpa_context = sdpa_context if self.backend is not None else nullcontext
        if self.backend is not None:
            log.warning(
                "SDPA context manager is not working well with torch.compile, causing graph breaks and "
                "significant slowdowns. If you are using torch.compile you'll most likely want to turn off "
                "this context manager."
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the scaled dot-product attention over the input tensors using the specified backend.
        B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head

        check F.scaled_dot_product_attention
        Args:
            q (Tensor): The query tensor of shape [B, Mq, H, K] / [B, ..., H, K]
            k (Tensor): The key tensor of shape [B, Mk, H, V] / [B, ..., H, K]
            v (Tensor): The value tensor of shape [B, Mk, H, V] / [B, ..., H, V]

            mask (Optional[Tensor]): An optional mask tensor. Follow scaled_dot_product_attention API, mask should be a boolean tensor with shape [B, H, Mq, Mk]

        Returns:
            Tensor: [B, Mq, H, V] / [B, ..., H, V]
        """
        in_q_shape = q.shape
        in_k_shape = k.shape
        q = rearrange(q, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
        k = rearrange(k, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        v = rearrange(v, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        if mask is not None:
            assert mask.dtype == torch.bool, "Mask should be a boolean tensor"
        with self.sdpa_context(self.backend):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # scale is dim_head ** -0.5 per default
        return rearrange(out, "b h ... l -> b ... h l").view(*in_q_shape[:-1], in_k_shape[-1])


class Attention(nn.Module):
    """
    Generalized attention impl. **With TP support**.

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
        norm_args (dict, optional): Arguments to pass to the normalization function. Defaults to an empty dict.

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
        norm_args: dict = {},
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}. Norm options are {qkv_norm} and norm args are {norm_args}."
        )
        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_format = qkv_format
        norm_dim = dim_head
        self.backend = backend
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim
        tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=False) if USE_MEGATRON else None
        if tp_group is None:
            self.tp_size = 1  # TP is not initialized.
        else:
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if self.backend == "torch":
            assert (
                self.tp_size == 1
            ), f"Attention backend {self.backend} cannot use TP size > 1. Attempted: {self.tp_size}"

        assert self.heads % self.tp_size == 0, "the number of heads should be divisible by TP size"

        if self.tp_size == 1:
            self.to_q = nn.Sequential(
                nn.Linear(query_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[0], norm_dim, **norm_args),
            )
            self.to_k = nn.Sequential(
                nn.Linear(context_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[1], norm_dim, **norm_args),
            )
            self.to_v = nn.Sequential(
                nn.Linear(context_dim, inner_dim, bias=qkv_bias),
                get_normalization(qkv_norm[2], norm_dim, **norm_args),
            )

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            )
        else:  # TP enabled.
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if sequence_parallel:
                assert qkv_format == "sbhd", "sequence parallel only supports sbhd format"

            self.to_q = nn.Sequential(
                te.pytorch.Linear(
                    query_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[0], norm_dim, **norm_args),
            )
            self.to_k = nn.Sequential(
                te.pytorch.Linear(
                    context_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[1], norm_dim, **norm_args),
            )
            self.to_v = nn.Sequential(
                te.pytorch.Linear(
                    context_dim,
                    inner_dim,
                    bias=qkv_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                ),
                get_normalization(qkv_norm[2], norm_dim, **norm_args),
            )

            self.to_out = nn.Sequential(
                te.pytorch.Linear(
                    inner_dim,
                    query_dim,
                    bias=out_bias,
                    tp_size=self.tp_size,
                    tp_group=tp_group,
                    parallel_mode="row",
                    sequence_parallel=sequence_parallel,
                ),
                nn.Dropout(dropout),
            )

        if attn_op:  # use what is given
            self.attn_op = attn_op
        elif self.backend == "transformer_engine":
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False) if USE_MEGATRON else False
            self.attn_op: BaseAttentionOp = DotProductAttention(
                self.heads,
                self.dim_head,
                num_gqa_groups=self.heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
                tp_size=self.tp_size,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
            )
        elif self.backend == "torch":
            self.attn_op = TorchAttentionOp(None)
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
        """

        q = self.to_q[0](x)
        context = x if context is None else context
        k = self.to_k[0](context)
        v = self.to_v[0](context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads // self.tp_size, c=self.dim_head),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
            q = self.to_q[1](q)
            k = self.to_k[1](k)
            v = self.to_v[1](v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = checkpoint(apply_norm_and_rotary_pos_emb, q, k, v, rope_emb, use_reentrant=False)

        return q, k, v

    def cal_attn(self, q, k, v, mask=None):
        if self.backend == "transformer_engine":
            seq_dim = self.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
            return self.to_out(out)
        elif self.backend == "torch":
            out = self.attn_op(q, k, v, mask=mask)  # [B, Mq, H, V]
            return self.to_out(rearrange(out, " b ... n c -> b ... (n c)"))
        else:
            raise ValueError(f"Backend {self.backend} not found")

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        return self.cal_attn(q, k, v, mask)
