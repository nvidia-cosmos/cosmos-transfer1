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
ControlNet Encoder based on GeneralDIT
"""

from typing import List, Optional, Tuple

import torch
from einops import rearrange

# from megatron.core import parallel_state
from torch import nn
from torchvision import transforms

from cosmos_transfer1.diffusion.conditioner import DataType
from cosmos_transfer1.diffusion.module.blocks import PatchEmbed, zero_module
from cosmos_transfer1.diffusion.module.parallel import split_inputs_cp
from cosmos_transfer1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT as GeneralDIT


class GeneralDITEncoder(GeneralDIT):
    """
    ControlNet Encoder based on GeneralDIT. Heavily borrowed from GeneralDIT with minor modifications.
    """

    def __init__(self, *args, **kwargs):
        hint_channels = kwargs.pop("hint_channels", 16)
        self.dropout_ctrl_branch = kwargs.pop("dropout_ctrl_branch", 0.5)
        num_control_blocks = kwargs.pop("num_control_blocks", None)
        if num_control_blocks is not None:
            assert num_control_blocks > 0 and num_control_blocks <= kwargs["num_blocks"]
            kwargs["layer_mask"] = [False] * num_control_blocks + [True] * (kwargs["num_blocks"] - num_control_blocks)
        self.random_drop_control_blocks = kwargs.pop("random_drop_control_blocks", False)
        super().__init__(*args, **kwargs)
        num_blocks = self.num_blocks
        model_channels = self.model_channels
        layer_mask = kwargs.get("layer_mask", None)
        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        self.layer_mask = layer_mask
        self.hint_channels = hint_channels
        self.build_hint_patch_embed()
        hint_nf = [16, 16, 32, 32, 96, 96, 256]
        nonlinearity = nn.SiLU()
        input_hint_block = [nn.Linear(model_channels, hint_nf[0]), nonlinearity]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
        self.input_hint_block = nn.Sequential(*input_hint_block)
        # Initialize weights
        self.initialize_weights()
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            self.zero_blocks[f"block{idx}"] = zero_module(nn.Linear(model_channels, model_channels))
        self.input_hint_block.append(zero_module(nn.Linear(hint_nf[-1], model_channels)))

    def build_hint_patch_embed(self):
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.hint_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder2 = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

    def prepare_hint_embedded_sequence(
        self, x_B_C_T_H_W: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder2(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
    ) -> torch.Tensor:
        assert hint.size(1) <= self.hint_channels, f"Expected hint channels <= {self.hint_channels}, got {hint.size(1)}"
        if hint.size(1) < self.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.hint_channels - hint.size(1)
            hint = torch.cat([hint, torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device)], dim=1)
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        hint_B_T_H_W_D, _ = self.prepare_hint_embedded_sequence(hint, fps=fps, padding_mask=padding_mask)
        hint = rearrange(hint_B_T_H_W_D, "B T H W D -> T H W B D")

        guided_hint = self.input_hint_block(hint)
        return guided_hint

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        hint_key: Optional[str] = None,
        base_model: Optional[nn.Module] = None,
        control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        regional_contexts: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
            regional_contexts: Optional list of regional context tensors
            region_masks: Optional tensor of region masks
        """
        # record the input as they are replaced in this forward
        x_input = x
        crossattn_emb_input = crossattn_emb
        crossattn_mask_input = crossattn_mask
        condition_video_input_mask_input = condition_video_input_mask

        regional_contexts_input = regional_contexts
        region_masks_input = region_masks

        hint = kwargs.pop(hint_key)
        if hint is None:
            print("using none hint")
            return base_model.net.forward(
                x=x_input,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb_input,
                crossattn_mask=crossattn_mask_input,
                fps=fps,
                padding_mask=padding_mask,
                scalar_feature=scalar_feature,
                data_type=data_type,
                condition_video_input_mask=condition_video_input_mask_input,
                regional_contexts=regional_contexts_input,
                region_masks=region_masks_input,
                **kwargs,
            )
        if hasattr(self, "hint_encoders"):  # for multicontrol
            guided_hints = []
            for i in range(hint.shape[1]):
                self.input_hint_block = self.hint_encoders[i].input_hint_block
                self.pos_embedder = self.hint_encoders[i].pos_embedder
                self.x_embedder2 = self.hint_encoders[i].x_embedder2
                guided_hints += [self.encode_hint(hint[:, i], fps=fps, padding_mask=padding_mask, data_type=data_type)]
        else:
            guided_hints = self.encode_hint(hint, fps=fps, padding_mask=padding_mask, data_type=data_type)
            guided_hints = torch.chunk(guided_hints, hint.shape[0] // x.shape[0], dim=3)
            # Only support multi-control at inference time
            assert len(guided_hints) == 1 or not torch.is_grad_enabled()

        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        B, C, T, H, W = x.shape
        if data_type == DataType.VIDEO:
            if condition_video_input_mask is not None:
                if self.cp_group is not None:
                    condition_video_input_mask = split_inputs_cp(
                        condition_video_input_mask, seq_dim=2, cp_group=self.cp_group
                    )
                input_list = [x, condition_video_input_mask]
                x = torch.cat(input_list, dim=1)
        elif data_type == DataType.IMAGE:
            # For image, we dont have condition_video_input_mask, or condition_video_pose
            # We need to add the extra channel for video condition mask
            padding_channels = self.in_channels - x.shape[1]
            x = torch.cat([x, torch.zeros((B, padding_channels, T, H, W), dtype=x.dtype, device=x.device)], dim=1)
        else:
            assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"

        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
        if crossattn_mask:
            crossattn_mask = rearrange(crossattn_mask, "B M -> M B")
        if regional_contexts is not None:
            regional_contexts = rearrange(regional_contexts, "B R M D -> R M B D")
        if region_masks is not None:
            region_masks = rearrange(region_masks, "B R T H W -> R T H W B")

        outs = {}

        num_control_blocks = self.layer_mask.index(True)
        num_layers_to_use = num_control_blocks
        control_gate_per_layer = [i < num_layers_to_use for i in range(num_control_blocks)]

        if isinstance(control_weight, torch.Tensor):
            if control_weight.ndim == 0:  # Single scalar tensor
                control_weight = [float(control_weight)]
            elif control_weight.ndim == 1:  # List of scalar weights
                control_weight = [float(w) for w in control_weight]
            else:  # Spatial-temporal weight maps
                if self.cp_group is not None:
                    control_weight = split_inputs_cp(control_weight, seq_dim=3, cp_group=self.cp_group)
                control_weight = [w for w in control_weight]  # Keep as tensor
        else:
            control_weight = [control_weight] * len(guided_hints)

        x_before_blocks = x.clone()
        for i, guided_hint in enumerate(guided_hints):
            x = x_before_blocks
            if hasattr(self, "hint_encoders"):  # for multicontrol
                blocks = self.hint_encoders[i].blocks
                zero_blocks = self.hint_encoders[i].zero_blocks
                t_embedder = self.hint_encoders[i].t_embedder
                affline_norm = self.hint_encoders[i].affline_norm
                self.x_embedder = self.hint_encoders[i].x_embedder
                self.extra_pos_embedder = self.hint_encoders[i].extra_pos_embedder
            else:
                blocks = self.blocks
                zero_blocks = self.zero_blocks
                t_embedder = self.t_embedder
                affline_norm = self.affline_norm

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
                x, fps=fps, padding_mask=padding_mask
            )
            # logging affline scale information
            affline_scale_log_info = {}

            timesteps_B_D, adaln_lora_B_3D = t_embedder(timesteps.flatten())
            affline_emb_B_D = timesteps_B_D
            affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

            if scalar_feature is not None:
                raise NotImplementedError("Scalar feature is not implemented yet.")

            affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
            affline_emb_B_D = affline_norm(affline_emb_B_D)

            # for logging purpose
            self.affline_scale_log_info = affline_scale_log_info
            self.affline_emb = affline_emb_B_D

            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )

            for idx, (name, block) in enumerate(blocks.items()):
                assert (
                    blocks["block0"].x_format == block.x_format
                ), f"First block has x_format {blocks[0].x_format}, got {block.x_format}"
                x = block(
                    x,
                    affline_emb_B_D,
                    crossattn_emb,
                    crossattn_mask,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_3D=adaln_lora_B_3D,
                    extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                    regional_contexts=regional_contexts,
                    region_masks=region_masks,
                )
                if guided_hint is not None:
                    x = x + guided_hint
                    guided_hint = None

                gate = control_gate_per_layer[idx]
                if isinstance(control_weight[i], (float, int)) or control_weight[i].ndim < 2:
                    hint_val = zero_blocks[name](x) * control_weight[i] * gate
                else:  # Spatial-temporal weights [num_controls, B, 1, T, H, W]
                    control_feat = zero_blocks[name](x)
                    # Get current feature dimensions
                    weight_map = control_weight[i]  # [B, 1, T, H, W]
                    # Reshape to match THWBD format
                    weight_map = weight_map.permute(2, 3, 4, 0, 1)  # [T, H, W, B, 1]
                    hint_val = control_feat * weight_map * gate
                if name not in outs:
                    outs[name] = hint_val
                else:
                    outs[name] += hint_val

        output = base_model.net.forward(
            x=x_input,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb_input,
            crossattn_mask=crossattn_mask_input,
            fps=fps,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            x_ctrl=outs,
            condition_video_input_mask=condition_video_input_mask_input,
            regional_contexts=regional_contexts_input,
            region_masks=region_masks_input,
            **kwargs,
        )
        return output
