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

# This script processes segmentation results for each video frame saved as JSON files and generates a spatial-temporal weight matrix saved as a .pt file.
# The input JSON files contain segmentation information for each frame, and the output .pt file represents the spatial-temporal weight matrix for the video.

import argparse
import glob
import json
import logging
import os
import re

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Class to manage different weight settings
class WeightSettings:
    """Class to manage different weight settings for the features"""

    @staticmethod
    def get_settings(setting_name):
        """Get weight settings by name

        Args:
            setting_name (str): Name of the setting

        Returns:
            dict: Dictionary with weights for each feature
        """
        settings = {
            # Default setting: Emphasize robot in all features
            "fg_vis_edge_bg_seg": {
                "depth": {"foreground": 0.0, "background": 0.0},
                "vis": {"foreground": 1.0, "background": 0.0},
                "edge": {"foreground": 1.0, "background": 0.0},
                "seg": {"foreground": 0.0, "background": 1.0},
            },
            "fg_edge_bg_seg": {
                "depth": {"foreground": 0.0, "background": 0.0},
                "vis": {"foreground": 0.0, "background": 0.0},
                "edge": {"foreground": 1.0, "background": 0.0},
                "seg": {"foreground": 0.0, "background": 1.0},
            },
        }

        if setting_name not in settings:
            logger.warning(f"Setting '{setting_name}' not found. Using default.")
            return settings["fg_vis_edge_bg_seg"]

        return settings[setting_name]

    @staticmethod
    def list_settings():
        """List all available settings

        Returns:
            list: List of setting names
        """
        return ["fg_vis_edge_bg_seg", "fg_edge_bg_seg"]


def get_video_info(video_path):
    """Get video dimensions and frame count"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return width, height, frame_count, fps


def parse_color_key(color_key):
    """Parse a color key string into an RGB tuple

    Args:
        color_key (str): Color key string in the format "(r,g,b,a)" or similar

    Returns:
        tuple: RGB tuple (r, g, b)
    """
    # Extract numbers using regex to handle different formats
    numbers = re.findall(r"\d+", color_key)
    if len(numbers) >= 3:
        r, g, b = map(int, numbers[:3])
        return (r, g, b)
    else:
        raise ValueError(f"Invalid color key format: {color_key}")


def save_visualization(mask, frame_num, feature_name, viz_dir):
    """Save a visualization of the binary mask

    Args:
        mask (numpy.ndarray): The mask (values 0 or 255)
        frame_num (int): The frame number
        feature_name (str): The name of the feature (depth, vis, edge, seg)
        viz_dir (str): Directory to save visualizations
    """
    # Simply save the binary mask directly
    output_path = os.path.join(viz_dir, f"{feature_name}_frame_{frame_num:06d}.png")
    cv2.imwrite(output_path, mask)
    logger.info(f"Saved binary visualization to {output_path}")


def process_segmentation_files(
    segmentation_dir,
    output_dir,
    viz_dir,
    video_path=None,
    weights_dict=None,
    setting_name="fg_vis_edge_bg_seg",
    robot_keywords=None,
):
    """Process all segmentation JSON files and create weight matrices

    Args:
        segmentation_dir (str): Directory containing segmentation JSON files
        output_dir (str): Directory to save weight matrices
        viz_dir (str): Directory to save visualizations
        video_path (str, optional): Path to the video file. Defaults to None.
        weights_dict (dict, optional): Dictionary with weights for each feature.
            Format: {
                'depth': {'foreground': float, 'background': float},
                'vis': {'foreground': float, 'background': float},
                'edge': {'foreground': float, 'background': float},
                'seg': {'foreground': float, 'background': float}
            }
            Values should be in range 0-1. Defaults to None.
        setting_name (str, optional): Weight setting name. Defaults to 'fg_vis_edge_bg_seg (setting1)'.
        robot_keywords (list, optional): List of keywords to identify robot classes. Defaults to ["robot"].
    """

    # Set default robot keywords if not provided
    if robot_keywords is None:
        robot_keywords = ["robot"]

    # Get all JSON files
    json_files = sorted(glob.glob(os.path.join(segmentation_dir, "*.json")))
    logger.info(f"Found {len(json_files)} JSON files")

    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {segmentation_dir}")

    # For example directories, check for PNG files
    png_dir = os.path.join(os.path.dirname(segmentation_dir), "segmentation")
    png_files = []
    if os.path.exists(png_dir):
        png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        logger.info(f"Found {len(png_files)} PNG files in segmentation directory")

    # Step 1: Create a unified color-to-class mapping from all JSON files
    logger.info("Creating unified color-to-class mapping...")
    rgb_to_class = {}
    rgb_to_is_robot = {}

    for json_file in tqdm(json_files, desc="Processing JSON files for unified mapping"):
        with open(json_file, "r") as f:
            json_data = json.load(f)

        for color_key, data in json_data.items():
            color = parse_color_key(color_key)
            class_name = data["class"]

            # Store RGB color for matching
            rgb_to_class[color] = class_name
            rgb_to_is_robot[color] = any(keyword in class_name for keyword in robot_keywords)

    # Print statistics about the unified color mapping
    robot_colors = [color for color, is_robot in rgb_to_is_robot.items() if is_robot]
    logger.info(f"Unified mapping: Found {len(robot_colors)} robot colors out of {len(rgb_to_is_robot)} total colors")
    if robot_colors:
        logger.info(f"Robot classes: {[rgb_to_class[color] for color in robot_colors]}")

    # Convert color mapping to arrays for vectorized operations
    colors = list(rgb_to_is_robot.keys())
    color_array = np.array(colors)
    is_robot_array = np.array([rgb_to_is_robot[color] for color in colors], dtype=bool)

    # If we have PNG files, get dimensions from the first PNG
    if png_files:
        # Get dimensions from the first PNG file
        first_png = cv2.imread(png_files[0])
        if first_png is None:
            raise ValueError(f"Could not read PNG file: {png_files[0]}")

        height, width = first_png.shape[:2]
        frame_count = len(png_files)

        # Match frame numbers between JSON and PNG files to ensure correct correspondence
        json_frame_nums = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in json_files]
        png_frame_nums = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in png_files]

        # Find common frames between JSON and PNG files
        common_frames = sorted(set(json_frame_nums).intersection(set(png_frame_nums)))
        logger.info(f"Found {len(common_frames)} common frames between JSON and PNG files")

        if len(common_frames) == 0:
            raise ValueError("No matching frames found between JSON and PNG files")

        # Create maps to easily look up files by frame number
        json_map = {int(os.path.basename(f).split("_")[-1].split(".")[0]): f for f in json_files}
        png_map = {int(os.path.basename(f).split("_")[-1].split(".")[0]): f for f in png_files}

        # Create new lists with only matching files
        json_files = [json_map[frame] for frame in common_frames if frame in json_map]
        png_files = [png_map[frame] for frame in common_frames if frame in png_map]
        num_frames = len(json_files)

        logger.info(f"Using PNG dimensions: {width}x{height}, processing {num_frames} frames")
    else:
        # Get video information if no PNG files available
        try:
            width, height, frame_count, fps = get_video_info(video_path)
            logger.info(f"Video dimensions: {width}x{height}, {frame_count} frames, {fps} fps")
            num_frames = min(len(json_files), frame_count)
        except Exception as e:
            logger.warning(f"Warning: Could not get video information: {e}")
            # Use a default size if we can't get the video info
            width, height = 640, 480
            num_frames = len(json_files)
            logger.info(f"Using default dimensions: {width}x{height}, {num_frames} frames")

    # Initialize weight tensors
    depth_weights = torch.zeros((num_frames, height, width))
    vis_weights = torch.zeros((num_frames, height, width))
    edge_weights = torch.zeros((num_frames, height, width))
    seg_weights = torch.zeros((num_frames, height, width))

    # Process frames
    if png_files:
        # Process PNG files directly
        for i, (json_file, png_file) in enumerate(zip(json_files, png_files)):
            # Get frame number from filename
            frame_num = int(os.path.basename(json_file).split("_")[-1].split(".")[0])

            # Read the corresponding PNG file
            frame = cv2.imread(png_file)

            if frame is None:
                logger.warning(f"Warning: Could not read frame {i} from PNG. Using blank frame.")
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Calculate total pixels
            total_pixels = height * width

            # Vectorized approach for finding nearest colors
            # Convert frame_rgb to a 2D array of shape (height*width, 3)
            pixels = frame_rgb.reshape(-1, 3)

            # Calculate distances between each pixel and each color (vectorized)
            # This creates a matrix of shape (height*width, num_colors)
            distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - color_array[np.newaxis, :, :]) ** 2, axis=2))

            # Find the index of the nearest color for each pixel
            nearest_color_indices = np.argmin(distances, axis=1)

            # Get the is_robot value for each pixel based on its nearest color
            pixel_is_robot = is_robot_array[nearest_color_indices]

            # Reshape back to image dimensions
            pixel_is_robot_2d = pixel_is_robot.reshape(height, width)

            # Count robot and matched pixels
            robot_pixel_count = np.sum(pixel_is_robot)
            matched_pixel_count = pixels.shape[0]  # All pixels are matched now

            # Create masks based on the is_robot classification
            depth_mask = np.where(
                pixel_is_robot_2d, weights_dict["depth"]["foreground"], weights_dict["depth"]["background"]
            )

            vis_mask = np.where(pixel_is_robot_2d, weights_dict["vis"]["foreground"], weights_dict["vis"]["background"])

            edge_mask = np.where(
                pixel_is_robot_2d, weights_dict["edge"]["foreground"], weights_dict["edge"]["background"]
            )

            seg_mask = np.where(pixel_is_robot_2d, weights_dict["seg"]["foreground"], weights_dict["seg"]["background"])

            # Create visualization mask
            visualization_mask = np.zeros((height, width), dtype=np.uint8)
            visualization_mask[pixel_is_robot_2d] = 255

            # Log statistics
            robot_percentage = (robot_pixel_count / total_pixels) * 100
            matched_percentage = (matched_pixel_count / total_pixels) * 100
            logger.info(f"Frame {frame_num}: {robot_pixel_count} robot pixels ({robot_percentage:.2f}%)")
            logger.info(f"Frame {frame_num}: {matched_pixel_count} matched pixels ({matched_percentage:.2f}%)")

            # Save visualizations for this frame
            save_visualization(visualization_mask, frame_num, "segmentation", viz_dir)

            # Store the masks in the weight tensors
            depth_weights[i] = torch.from_numpy(depth_mask)
            vis_weights[i] = torch.from_numpy(vis_mask)
            edge_weights[i] = torch.from_numpy(edge_mask)
            seg_weights[i] = torch.from_numpy(seg_mask)
    else:
        # Use video frames if available
        try:
            # Open the segmentation video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Process each frame using the unified color mapping
            for i, json_file in enumerate(tqdm(json_files[:num_frames], desc="Processing frames")):
                # Get frame number from filename
                frame_num = int(os.path.basename(json_file).split("_")[-1].split(".")[0])

                # Read the corresponding frame from the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Warning: Could not read frame {i} from video. Using blank frame.")
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate total pixels
                total_pixels = height * width

                # Vectorized approach for finding nearest colors
                pixels = frame_rgb.reshape(-1, 3)
                distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - color_array[np.newaxis, :, :]) ** 2, axis=2))
                nearest_color_indices = np.argmin(distances, axis=1)
                pixel_is_robot = is_robot_array[nearest_color_indices]
                pixel_is_robot_2d = pixel_is_robot.reshape(height, width)

                # Count robot and matched pixels
                robot_pixel_count = np.sum(pixel_is_robot)
                matched_pixel_count = pixels.shape[0]

                # Create masks based on the is_robot classification
                depth_mask = np.where(
                    pixel_is_robot_2d, weights_dict["depth"]["foreground"], weights_dict["depth"]["background"]
                )
                vis_mask = np.where(
                    pixel_is_robot_2d, weights_dict["vis"]["foreground"], weights_dict["vis"]["background"]
                )
                edge_mask = np.where(
                    pixel_is_robot_2d, weights_dict["edge"]["foreground"], weights_dict["edge"]["background"]
                )
                seg_mask = np.where(
                    pixel_is_robot_2d, weights_dict["seg"]["foreground"], weights_dict["seg"]["background"]
                )

                # Create visualization mask
                visualization_mask = np.zeros((height, width), dtype=np.uint8)
                visualization_mask[pixel_is_robot_2d] = 255

                # Log statistics
                robot_percentage = (robot_pixel_count / total_pixels) * 100
                matched_percentage = (matched_pixel_count / total_pixels) * 100
                logger.info(f"Frame {frame_num}: {robot_pixel_count} robot pixels ({robot_percentage:.2f}%)")
                logger.info(f"Frame {frame_num}: {matched_pixel_count} matched pixels ({matched_percentage:.2f}%)")

                # Save visualizations for this frame
                save_visualization(visualization_mask, frame_num, "segmentation", viz_dir)

                # Store the masks in the weight tensors
                depth_weights[i] = torch.from_numpy(depth_mask)
                vis_weights[i] = torch.from_numpy(vis_mask)
                edge_weights[i] = torch.from_numpy(edge_mask)
                seg_weights[i] = torch.from_numpy(seg_mask)

            # Close the video capture
            cap.release()
        except Exception as e:
            logger.warning(f"Warning: Error processing video: {e}")
            logger.warning("Cannot process this example without proper frame data.")
            raise ValueError(f"Cannot process example without frame data: {e}")

    # Save weight tensors
    # Convert weights to half precision (float16) to reduce file size
    depth_weights_half = depth_weights.to(torch.float16)
    vis_weights_half = vis_weights.to(torch.float16)
    edge_weights_half = edge_weights.to(torch.float16)
    seg_weights_half = seg_weights.to(torch.float16)

    # Save the half precision tensors
    torch.save(depth_weights_half, os.path.join(output_dir, "depth_weights.pt"))
    torch.save(vis_weights_half, os.path.join(output_dir, "vis_weights.pt"))
    torch.save(edge_weights_half, os.path.join(output_dir, "edge_weights.pt"))
    torch.save(seg_weights_half, os.path.join(output_dir, "seg_weights.pt"))

    logger.info(f"Saved weight matrices to {output_dir}")
    logger.info(f"Weight matrix shape: {depth_weights_half.shape}, dtype: {depth_weights_half.dtype}")
    logger.info(f"Saved visualizations to {viz_dir}")

    return output_dir, viz_dir


def process_all_examples(input_dir, output_dir, setting_name="fg_vis_edge_bg_seg", robot_keywords=None):
    """Process all example directories in the provided input directory

    Args:
        input_dir (str): Input directory containing example folders
        output_dir (str): Output directory for weight matrices
        setting_name (str, optional): Weight setting name. Defaults to 'fg_vis_edge_bg_seg'.
        robot_keywords (list, optional): List of keywords to identify robot classes. Defaults to None.
    """
    # Find all example directories
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return []

    # List example directories
    examples = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    examples = sorted(examples)

    if not examples:
        logger.warning("No example directories found.")
        return []

    # Print found examples
    logger.info(f"Found {len(examples)} example directories:")
    for example in examples:
        logger.info(f"  - {example}")

    # Store processing results
    results = []

    # Process each example
    for example in examples:
        try:
            logger.info(f"\nProcessing {example}...")

            # Process this example with custom directories
            out_dir, viz_dir = process_example_with_dirs(example, input_dir, output_dir, setting_name, robot_keywords)
            results.append((example, out_dir, viz_dir))

            logger.info(f"Results for {example} saved to:")
            logger.info(f"  Weight matrices: {out_dir}")
            logger.info(f"  Visualizations: {viz_dir}")

        except Exception as e:
            logger.error(f"Error processing {example}: {e}")

    logger.info("\nAll examples processed.")
    return results


# Process a specific example with custom input and output directories
def process_example_with_dirs(
    example_name, input_dir, output_dir, setting_name="fg_vis_edge_bg_seg", robot_keywords=None
):
    """Process a specific example with custom input and output directories

    Args:
        example_name (str): Name of the example directory
        input_dir (str): Path to input directory containing example folders
        output_dir (str): Path to output directory for weight matrices
        setting_name (str, optional): Weight setting name. Defaults to 'fg_vis_edge_bg_seg'.
        robot_keywords (list, optional): List of keywords to identify robot classes. Defaults to None.
    """
    # Create paths for this example
    example_dir = os.path.join(input_dir, example_name)
    segmentation_dir = os.path.join(example_dir, "segmentation_label")
    video_path = os.path.join(example_dir, "segmentation.mp4")

    # Create output directories
    example_output_dir = os.path.join(output_dir, example_name)
    viz_dir = os.path.join(example_output_dir, "visualizations")

    # Check if weight files already exist
    depth_weights_path = os.path.join(example_output_dir, "depth_weights.pt")
    if os.path.exists(depth_weights_path):
        logger.info(f"Weight files already exist for {example_name}, skipping processing")
        return example_output_dir, viz_dir

    # Create output directories if they don't exist
    os.makedirs(example_output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Get weight settings
    weights_dict = WeightSettings.get_settings(setting_name)

    # Process this example directly with paths
    return process_segmentation_files(
        segmentation_dir=segmentation_dir,
        output_dir=example_output_dir,
        viz_dir=viz_dir,
        video_path=video_path,
        weights_dict=weights_dict,
        setting_name=setting_name,
        robot_keywords=robot_keywords,
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process segmentation files to generate spatial-temporal weight matrices"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="fg_vis_edge_bg_seg",
        choices=WeightSettings.list_settings(),
        help="Weight setting to use (default: fg_vis_edge_bg_seg (setting1), fg_edge_bg_seg (setting2))",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="assets/robot_augmentation_example",
        help="Input directory containing example folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/robot_augmentation_example",
        help="Output directory for weight matrices",
    )
    parser.add_argument(
        "--robot-keywords",
        type=str,
        nargs="+",
        default=["world_robot", "gripper", "robot"],
        help="Keywords used to identify robot classes (default: world_robot gripper robot)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set logging level from command line argument
    logger.setLevel(getattr(logging, args.log_level))

    # Get directories from arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    setting_name = args.setting
    robot_keywords = args.robot_keywords

    logger.info(f"Using input directory: {input_dir}")
    logger.info(f"Using output directory: {output_dir}")
    logger.info(f"Using weight setting: {setting_name}")
    logger.info(f"Using robot keywords: {robot_keywords}")

    # Process all examples with the provided input and output directories
    process_all_examples(input_dir, output_dir, setting_name, robot_keywords)
