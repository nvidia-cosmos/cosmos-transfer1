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

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import torch

from cosmos_transfer1.utils import log


class RegionalPromptProcessor:
    """
    Processes regional prompts and creates corresponding masks for attention.
    """

    def __init__(self, max_img_h, max_img_w, max_frames):
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames

    def create_region_masks_from_boxes(
        self,
        bounding_boxes: List[List[float]],
        batch_size: int,
        time_dim: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create region masks from bounding boxes [x1, y1, x2, y2] in normalized coordinates (0-1).

        Returns:
            region_masks: Tensor of shape (B, R, T, H, W) with values between 0 and 1
        """
        num_regions = len(bounding_boxes)
        region_masks = torch.zeros(
            batch_size, num_regions, time_dim, height, width, device=device, dtype=torch.bfloat16
        )

        for r, box in enumerate(bounding_boxes):
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = box
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            # Create mask for this region
            region_masks[:, r, :, y1:y2, x1:x2] = 1.0

        return region_masks

    def create_region_masks_from_segmentation(
        self,
        segmentation_maps: List[torch.Tensor],
        batch_size: int,
        time_dim: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create masks from binary segmentation maps.

        Args:
            segmentation_maps: List of Tensors, each of shape (T, H, W) with binary values

        Returns:
            region_masks: Tensor of shape (B, R, T, H, W) with binary values
        """
        num_regions = len(segmentation_maps)
        region_masks = torch.zeros(
            batch_size, num_regions, time_dim, height, width, device=device, dtype=torch.bfloat16
        )

        for r, seg_map in enumerate(segmentation_maps):
            # Clip to 121 frames if longer
            if seg_map.shape[0] > time_dim:
                log.info(f"clipping segmentation map to {time_dim} frames")
                seg_map = seg_map[:time_dim]
            region_masks[:, r] = seg_map.float()

        return region_masks

    def visualize_region_masks(
        self, region_masks: torch.Tensor, save_path: str, time_dim: int, height: int, width: int
    ) -> None:
        """
        Visualize region masks for debugging purposes.

        Args:
            region_masks: Tensor of shape (B, R, T*H*W)
            save_path: Path to save the visualization
            time_dim: Number of frames
            height: Height in latent space
            width: Width in latent space
        """

        B, R, T, H, W = region_masks.shape
        reshaped_masks = region_masks

        # Create figure
        fig, axes = plt.subplots(R, 1, figsize=(10, 3 * R))
        if R == 1:
            axes = [axes]
        for r in range(R):
            axes[r].imshow(reshaped_masks[r, time_dim // 2].cpu().numpy(), cmap="gray")
            axes[r].set_title(f"Region {r+1} Mask (Middle Frame)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def compress_segmentation_map(segmentation_map, compression_factor):
    # Handle both [T,H,W] and [C,T,H,W] formats
    if len(segmentation_map.shape) == 4:  # [C,T,H,W] format
        C, T, H, W = segmentation_map.shape
        # Assuming first channel contains the main segmentation mask
        # Can be modified based on specific requirements
        segmentation_map = segmentation_map[0]  # Take first channel, now [T,H,W]

    # Add batch and channel dimensions [1, 1, T, H, W]
    expanded_map = segmentation_map.unsqueeze(0).unsqueeze(0)
    T, H, W = segmentation_map.shape
    new_H = H // compression_factor
    new_W = W // compression_factor

    compressed_map = torch.nn.functional.interpolate(
        expanded_map, size=(T, new_H, new_W), mode="trilinear", align_corners=False
    )

    return compressed_map.squeeze(0).squeeze(0)


def prepare_regional_prompts(
    model,
    global_prompt: Union[str, torch.Tensor],
    regional_prompts: torch.Tensor,
    region_definitions: List[Union[List[float], str]],
    batch_size: int,
    time_dim: int,
    height: int,
    width: int,
    device: torch.device,
    cache_dir: str = None,
    local_files_only: bool = False,
    visualize_masks: bool = False,
    visualization_path: str = None,
    compression_factor: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare regional prompts and masks for inference.

    Args:
        model: DiT model
        global_prompt: Global text prompt or pre-computed embedding
        regional_prompts: List of regional text prompts
        region_definitions: List of bounding boxes [x1, y1, x2, y2] or segmentation map
        batch_size: Batch size
        time_dim: Number of frames
        height: Height in latent space
        width: Width in latent space
        device: Device to create tensors on
        cache_dir: Cache directory for text encoder
        local_files_only: Whether to use only local files for text encoder
        visualize_masks: Whether to visualize the region masks for debugging
        visualization_path: Path to save the visualization

    Returns:
        global_context: Global prompt embedding
        regional_contexts: List of regional prompt embeddings
        region_masks: Region masks tensor with values between 0 and 1
    """
    processor = RegionalPromptProcessor(max_img_h=height, max_img_w=width, max_frames=time_dim)

    # Validate that we have matching number of prompts and region definitions
    if len(regional_prompts) != len(region_definitions):
        raise ValueError(
            f"Number of regional prompts ({len(regional_prompts)}) must match "
            f"total number of region definitions ({len(region_definitions)})"
        )

    # Track which prompts correspond to which region types while maintaining order
    box_prompts = []
    seg_prompts = []
    prompt_idx = 0

    segmentation_maps: List[torch.Tensor] = []
    region_definitions_list: List[List[float]] = []
    # Maintain correspondence between prompts and region definitions
    for region_definition in region_definitions:
        if isinstance(region_definition, str):
            segmentation_map = torch.load(region_definition, weights_only=False)
            # Validate segmentation map dimensions
            if len(segmentation_map.shape) not in [3, 4]:
                raise ValueError(f"Segmentation map should have shape [T,H,W] or [C,T,H,W], got shape {segmentation_map.shape}")

            segmentation_map = compress_segmentation_map(segmentation_map, compression_factor)
            log.info(f"segmentation_map shape: {segmentation_map.shape}")
            segmentation_maps.append(segmentation_map)
            seg_prompts.append(regional_prompts[prompt_idx])
        elif isinstance(region_definition, list):
            region_definitions_list.append(region_definition)
            box_prompts.append(regional_prompts[prompt_idx])
        else:
            raise ValueError(f"Region definition format not recognized: {type(region_definition)}")
        prompt_idx += 1

    # Update regional_prompts to maintain correct ordering
    regional_prompts = box_prompts + seg_prompts
    region_masks_boxes = processor.create_region_masks_from_boxes(
        region_definitions_list, batch_size, time_dim, height, width, device
    )
    region_masks_segmentation = processor.create_region_masks_from_segmentation(
        segmentation_maps, batch_size, time_dim, height, width, device
    )
    region_masks = torch.cat([region_masks_boxes, region_masks_segmentation], dim=1)

    if visualize_masks and visualization_path:
        processor.visualize_region_masks(region_masks, visualization_path, time_dim, height, width)

    if isinstance(global_prompt, str):
        pass
    elif isinstance(global_prompt, torch.Tensor):
        global_context = global_prompt.to(dtype=torch.bfloat16)
    else:
        raise ValueError("Global prompt format not recognized.")

    regional_contexts = []
    for regional_prompt in regional_prompts:
        if isinstance(regional_prompt, str):
            raise ValueError(f"Regional prompt should be converted to embedding: {type(regional_prompt)}")
        elif isinstance(regional_prompt, torch.Tensor):
            regional_context = regional_prompt.to(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Regional prompt format not recognized: {type(regional_prompt)}")

        regional_contexts.append(regional_context)

    regional_contexts = torch.stack(regional_contexts, dim=1)
    return global_context, regional_contexts, region_masks
