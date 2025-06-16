import os
import torch
import numpy as np
import cv2
import magic
from typing import Tuple
from cosmos_transfer1.utils import log

# Supported video extensions and corresponding MIME types
SUPPORTED_VIDEO_TYPES = {
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.webm': 'video/webm',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
}

def video_to_tensor(video_path: str, output_path: str, normalize: bool = True) -> Tuple[torch.Tensor, float]:
    """Convert an MP4 video file to a tensor and save it as a .pt file.
    Args:
        video_path (str): Path to input MP4 video file
        output_path (str): Path to save output .pt tensor file
        normalize (bool): Whether to normalize pixel values to [-1,1] range (default: True)

    Returns:
        Tuple[torch.Tensor, float]: Tuple containing:
            - Video tensor in shape [C,T,H,W] 
            - Video FPS
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read frames from video: {video_path}")

    height, width = frame.shape[:2]

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize tensor to store frames
    frames = []

    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    
    

    log.info(f"frames: {len(frames)}")
    # Convert frames to tensor
    video_tensor = torch.from_numpy(np.array(frames))
    log.info(f"video_tensor shape: {video_tensor.shape}")
    # Reshape from [T,H,W,C] to [C,T,H,W]
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    # Normalize if requested
    if normalize:
        video_tensor = video_tensor.float() / 127.5 - 1.0

    # Save tensor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(video_tensor, output_path)

    return video_tensor, fps


def is_valid_video(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    expected_mime = SUPPORTED_VIDEO_TYPES.get(ext)

    if not expected_mime:
        return False  # Extension not supported

    # Detect MIME type from actual file content
    detected_mime = magic.from_file(file_path, mime=True)

    return detected_mime == expected_mime
