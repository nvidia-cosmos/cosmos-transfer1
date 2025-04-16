# Input: segmentation results for each video saved as json files for each frame.
# Output: .pt file to save a spatial-temporal weight matrix.

import json
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import glob
import re
from collections import defaultdict
import argparse

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
            'setting1': {
                'depth': {'foreground': 0.0, 'background': 0.0},
                'blur': {'foreground': 1.0, 'background': 0.0},
                'canny': {'foreground': 1.0, 'background': 0.0},
                'segmentation': {'foreground': 0.0, 'background': 1.0}
            },

            'setting2': {
                'depth': {'foreground': 0.0, 'background': 0.0},
                'blur': {'foreground': 0.0, 'background': 0.0},
                'canny': {'foreground': 1.0, 'background': 0.0},
                'segmentation': {'foreground': 0.0, 'background': 1.0}
            }      
        }
        
        if setting_name not in settings:
            print(f"Warning: Setting '{setting_name}' not found. Using default.")
            return settings['default']
        
        return settings[setting_name]
    
    @staticmethod
    def list_settings():
        """List all available settings
        
        Returns:
            list: List of setting names
        """
        return [
            'setting1', 'setting2'
        ]

# Function to get paths based on scene number
def get_paths(scene_num, setting_name='default'):
    """Get paths based on scene number and setting name
    
    Args:
        scene_num (int): Scene number
        setting_name (str, optional): Weight setting name. Defaults to 'default'.
        
    Returns:
        tuple: (segmentation_dir, video_path, output_dir, viz_dir)
    """
    base_dir = "/mnt/andy/data/robot/robotlab_20250303"
    scene_dir = f"scene{scene_num}"
    
    segmentation_dir = os.path.join(base_dir, scene_dir, "semantic_segmentation")
    video_path = os.path.join(base_dir, scene_dir, "semantic_segmentation_5sec.mp4")
    
    # Create setting-specific output directories
    output_dir = os.path.join(base_dir, scene_dir, "weight_matrices_union", setting_name)
    viz_dir = os.path.join(base_dir, scene_dir, "visualizations_union", setting_name)
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    return segmentation_dir, video_path, output_dir, viz_dir

# Paths will be set based on command-line arguments
SEGMENTATION_DIR = None
VIDEO_PATH = None
OUTPUT_DIR = None
VIZ_DIR = None

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
    numbers = re.findall(r'\d+', color_key)
    if len(numbers) >= 3:
        r, g, b = map(int, numbers[:3])
        return (r, g, b)
    else:
        raise ValueError(f"Invalid color key format: {color_key}")

def find_nearest_color(pixel_rgb, color_mapping):
    """Find the nearest color in the mapping based on Euclidean distance in RGB space
    
    Args:
        pixel_rgb (tuple): RGB tuple of the pixel
        color_mapping (dict): Mapping of RGB tuples to values
        
    Returns:
        tuple: The nearest RGB color in the mapping
    """
    if not color_mapping:
        return None
    
    # If the exact color is in the mapping, return it
    if pixel_rgb in color_mapping:
        return pixel_rgb
    
    # Find the nearest color based on Euclidean distance
    min_distance = float('inf')
    nearest_color = None
    
    for color in color_mapping:
        # Calculate Euclidean distance in RGB space
        distance = sum((a - b) ** 2 for a, b in zip(pixel_rgb, color))
        
        if distance < min_distance:
            min_distance = distance
            nearest_color = color
    
    return nearest_color

def save_visualization(frame, mask, frame_num, feature_name):
    """Save a visualization of the binary mask
    
    Args:
        frame (numpy.ndarray): The original frame (not used)
        mask (numpy.ndarray): The mask (values 0 or 255)
        frame_num (int): The frame number
        feature_name (str): The name of the feature (depth, blur, canny, segmentation)
    """
    # Simply save the binary mask directly
    output_path = os.path.join(VIZ_DIR, f"{feature_name}_frame_{frame_num:06d}.png")
    cv2.imwrite(output_path, mask)
    print(f"Saved binary visualization to {output_path}")

def process_segmentation_files(weights_dict=None, scene_num=1, setting_name='default'):
    """Process all segmentation JSON files and create weight matrices
    
    Args:
        weights_dict (dict, optional): Dictionary with weights for each feature.
            Format: {
                'depth': {'foreground': float, 'background': float},
                'blur': {'foreground': float, 'background': float},
                'canny': {'foreground': float, 'background': float},
                'segmentation': {'foreground': float, 'background': float}
            }
            Values should be in range 0-1. Defaults to None.
        scene_num (int, optional): Scene number to process. Defaults to 1.
        setting_name (str, optional): Weight setting name. Defaults to 'default'.
    """
    # Set paths based on scene number and setting name
    global SEGMENTATION_DIR, VIDEO_PATH, OUTPUT_DIR, VIZ_DIR
    SEGMENTATION_DIR, VIDEO_PATH, OUTPUT_DIR, VIZ_DIR = get_paths(scene_num, setting_name)
    
    # Default weights if not provided
    if weights_dict is None:
        weights_dict = {
            'depth': {'foreground': 1.0, 'background': 0.0},
            'blur': {'foreground': 1.0, 'background': 0.0},
            'canny': {'foreground': 1.0, 'background': 0.0},
            'segmentation': {'foreground': 1.0, 'background': 0.0}
        }
    
    # Get video information
    width, height, frame_count, fps = get_video_info(VIDEO_PATH)
    print(f"Video dimensions: {width}x{height}, {frame_count} frames, {fps} fps")
    
    # Get all JSON files
    json_files = sorted(glob.glob(os.path.join(SEGMENTATION_DIR, "*.json")))
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {SEGMENTATION_DIR}")
    
    # Step 1: Create a unified color-to-class mapping from all JSON files
    print("Creating unified color-to-class mapping...")
    rgb_to_class = {}
    rgb_to_is_robot = {}
    
    for json_file in tqdm(json_files, desc="Processing JSON files for unified mapping"):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        for color_key, data in json_data.items():
            color = parse_color_key(color_key)
            class_name = data["class"]
            
            # Store RGB color for matching
            rgb_to_class[color] = class_name
            rgb_to_is_robot[color] = class_name.startswith("world_robot")
    
    # Print statistics about the unified color mapping
    robot_colors = [color for color, is_robot in rgb_to_is_robot.items() if is_robot]
    print(f"Unified mapping: Found {len(robot_colors)} robot colors out of {len(rgb_to_is_robot)} total colors")
    
    # Convert color mapping to arrays for vectorized operations
    colors = list(rgb_to_is_robot.keys())
    color_array = np.array(colors)
    is_robot_array = np.array([rgb_to_is_robot[color] for color in colors], dtype=bool)
    
    # Initialize weight tensors
    num_frames = min(len(json_files), frame_count)
    depth_weights = torch.zeros((num_frames, height, width))
    blur_weights = torch.zeros((num_frames, height, width))
    canny_weights = torch.zeros((num_frames, height, width))
    segmentation_weights = torch.zeros((num_frames, height, width))
    
    # Open the segmentation video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {VIDEO_PATH}")
    
    # Process each frame using the unified color mapping
    for i, json_file in enumerate(tqdm(json_files[:num_frames], desc="Processing frames")):
        # Get frame number from filename
        frame_num = int(os.path.basename(json_file).split('_')[-1].split('.')[0])
        
        # Read the corresponding frame from the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {i} from video. Using blank frame.")
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
        depth_mask = np.where(pixel_is_robot_2d, 
                            weights_dict['depth']['foreground'], 
                            weights_dict['depth']['background'])
        
        blur_mask = np.where(pixel_is_robot_2d, 
                           weights_dict['blur']['foreground'], 
                           weights_dict['blur']['background'])
        
        canny_mask = np.where(pixel_is_robot_2d, 
                            weights_dict['canny']['foreground'], 
                            weights_dict['canny']['background'])
        
        segmentation_mask = np.where(pixel_is_robot_2d, 
                                   weights_dict['segmentation']['foreground'], 
                                   weights_dict['segmentation']['background'])
        
        # Create visualization mask
        visualization_mask = np.zeros((height, width), dtype=np.uint8)
        visualization_mask[pixel_is_robot_2d] = 255
        
        # Log statistics
        robot_percentage = (robot_pixel_count / total_pixels) * 100
        matched_percentage = (matched_pixel_count / total_pixels) * 100
        print(f"Frame {frame_num}: {robot_pixel_count} robot pixels ({robot_percentage:.2f}%)")
        print(f"Frame {frame_num}: {matched_pixel_count} matched pixels ({matched_percentage:.2f}%)")
        
        # Save visualizations for this frame
        save_visualization(frame, visualization_mask, frame_num, "segmentation")
        
        # Store the masks in the weight tensors
        depth_weights[i] = torch.from_numpy(depth_mask)
        blur_weights[i] = torch.from_numpy(blur_mask)
        canny_weights[i] = torch.from_numpy(canny_mask)
        segmentation_weights[i] = torch.from_numpy(segmentation_mask)
    
    # Close the video capture
    cap.release()
    
    # Save weight tensors
    torch.save(depth_weights, os.path.join(OUTPUT_DIR, "depth_weights.pt"))
    torch.save(blur_weights, os.path.join(OUTPUT_DIR, "blur_weights.pt"))
    torch.save(canny_weights, os.path.join(OUTPUT_DIR, "canny_weights.pt"))
    torch.save(segmentation_weights, os.path.join(OUTPUT_DIR, "segmentation_weights.pt"))
    
    print(f"Saved weight matrices to {OUTPUT_DIR}")
    print(f"Weight matrix shape: {depth_weights.shape}")
    print(f"Saved visualizations to {VIZ_DIR}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process segmentation files for a specific scene')
    parser.add_argument('--scene', type=int, default=1, help='Scene number to process (default: 1)')
    parser.add_argument('--setting', type=str, default='default', 
                        choices=WeightSettings.list_settings(),
                        help='Weight setting to use (default: default)')
    args = parser.parse_args()
    
    # Get weight settings based on command-line argument
    weights_dict = WeightSettings.get_settings(args.setting)
    
    # Process segmentation files with selected weights for the specified scene
    process_segmentation_files(weights_dict, args.scene, args.setting)
    print(f"Processed scene {args.scene} with weight setting '{args.setting}'")
    print(f"Results saved to:")
    print(f"  Weight matrices: {os.path.join('/mnt/andy/data/robot/robotlab_20250303', f'scene{args.scene}', 'weight_matrices', args.setting)}")
    print(f"  Visualizations: {os.path.join('/mnt/andy/data/robot/robotlab_20250303', f'scene{args.scene}', 'visualizations', args.setting)}")


