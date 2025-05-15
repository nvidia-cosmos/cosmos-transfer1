# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
from cosmos_transfer1.utils import log
import torch

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


def prepare_regional_prompts(
    model,
    global_prompt: Union[str, torch.Tensor],
    regional_prompts: torch.Tensor,
    region_definitions: Union[List[List[float]], torch.Tensor],
    batch_size: int,
    time_dim: int,
    height: int,
    width: int,
    device: torch.device,
    cache_dir: str = None,
    local_files_only: bool = False,
    visualize_masks: bool = False,
    visualization_path: str = None,
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
    if isinstance(region_definitions[0], list):
        region_masks = processor.create_region_masks_from_boxes(
            region_definitions, batch_size, time_dim, height, width, device
        )
    else:
        segmentation_maps = [torch.load(path, weights_only=False).to(device) for path in region_definitions]
        region_masks = processor.create_region_masks_from_segmentation(
            segmentation_maps, batch_size, time_dim, height, width, device
        )

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
