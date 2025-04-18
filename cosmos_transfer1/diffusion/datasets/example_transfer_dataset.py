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
Run this command to interactively debug:
PYTHONPATH=. python cosmos_transfer1/diffusion/datasets/example_transfer_dataset.py
"""

import os
import warnings
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import pickle

from cosmos_transfer1.diffusion.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import VIDEO_RES_SIZE_INFO
from cosmos_transfer1.diffusion.inference.inference_utils import detect_aspect_ratio
from cosmos_transfer1.utils.lazy_config import instantiate



# mappings between control types and corresponding sub-folders names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "seg": {"folder": "seg", "format": "pickle", "data_dict_key": "segmentation"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly
    "vis": {"folder": None},   # Blur, computed on-the-fly
    "upscale": {"folder": None} # Computed on-the-fly
}


class ExampleTransferDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        resolution,
        hint_key="control_input_vis",
        is_train=True
    ):
        """Dataset class for loading video-text-to-video generation data with control inputs.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of consecutive frames to load per sequence
            resolution (str): resolution of the target video size
            hint_key (str): The hint key for loading the correct control input data modality
            is_train (bool): Whether this is for training

        NOTE: in our example dataset we do not have a validation dataset. The is_train flag is kept here for customized configuration.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = resolution
        assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        # Control input setup with file formats
        self.ctrl_type = hint_key.lstrip("control_input_")
        self.ctrl_data_pth_config = CTRL_TYPE_INFO[self.ctrl_type]

        # Set up directories - only collect paths
        video_dir = os.path.join(self.dataset_dir, "videos")
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")
        print(f"Finish initializing dataset with {len(self.video_paths)} videos in total.")

        # Set up preprocessing and augmentation
        augmentor_name = f"video_ctrlnet_augmentor_{hint_key}"
        augmentor_cfg = AUGMENTOR_OPTIONS[augmentor_name](resolution=resolution)
        self.augmentor = {k: instantiate(v) for k, v in augmentor_cfg.items()}

    def _sample_frames(self, video_path):
        """Sample frames from video and get metadata"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)
        
        # Calculate valid start frame range
        max_start_idx = n_frames - self.sequence_length
        if max_start_idx < 0:  # Video is too short
            return None, None, None
            
        # Sample start frame
        start_frame = np.random.randint(0, max_start_idx + 1)
        frame_ids = list(range(start_frame, start_frame + self.sequence_length))
        
        # Load frames
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = frames.astype(np.uint8)
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
            
        return frames, frame_ids, fps

    def _load_control_data(self, sample):
        """Load control data for the video clip."""
        data_dict = {}
        frame_ids = sample["frame_ids"]
        ctrl_path = sample["ctrl_path"]
        try:
            if self.ctrl_type == "seg":
                with open(ctrl_path, 'rb') as f:
                    ctrl_data = pickle.load(f)
                # key should match line 982 at cosmos_transfer1/diffusion/datasets/augmentors/control_input.py
                data_dict["segmentation"] = ctrl_data
            elif self.ctrl_type == "keypoint":
                with open(ctrl_path, 'rb') as f:
                    ctrl_data = pickle.load(f)
                data_dict["keypoint"] = ctrl_data
            elif self.ctrl_type == "depth":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                # Ensure the depth video has the same number of frames
                assert len(vr) >= frame_ids[-1] + 1, \
                    f"Depth video {ctrl_path} has fewer frames than main video"

                # Load the corresponding frames
                depth_frames = vr.get_batch(frame_ids).asnumpy() # [T,H,W,C]
                depth_frames = torch.from_numpy(depth_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                data_dict["depth"] = {
                    "video": depth_frames,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1],
                }

        except Exception as e:
            warnings.warn(f"Failed to load control data from {ctrl_path}: {str(e)}")
            return None

        return data_dict

    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:
                video_path = self.video_paths[index]
                video_name = os.path.basename(video_path).replace(".mp4", "")

                # Sample frames
                frames, frame_ids, fps = self._sample_frames(video_path)
                if frames is None:  # Invalid video or too short
                    index = np.random.randint(len(self.video_paths))
                    continue

                data = dict()

                # Process video frames
                video = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [T,H,W,C] -> [C,T,H,W]
                aspect_ratio = detect_aspect_ratio((video.shape[3], video.shape[2]))  # expects (W, H)
                
                # Basic data
                data["video"] = video
                data["aspect_ratio"] = aspect_ratio
                data["video_name"] = {
                    "video_path": video_path,
                    "t5_embedding_path": os.path.join(self.t5_dir, f"{video_name}.pickle"),
                    "start_frame_id": str(frame_ids[0]),
                }

                # Load T5 embeddings
                with open(data["video_name"]["t5_embedding_path"], "rb") as f:
                    t5_embedding = pickle.load(f)[0]
                data["t5_text_embeddings"] = torch.from_numpy(t5_embedding) #.cuda()
                data["t5_text_mask"] = torch.ones(512, dtype=torch.int64) #.cuda()

                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([704, 1280, 704, 1280]) #.cuda()
                data["padding_mask"] = torch.zeros(1, 704, 1280) #.cuda()
                
                if self.ctrl_type:
                    ctrl_data = self._load_control_data({
                        "ctrl_path": os.path.join(
                            self.dataset_dir,
                            self.ctrl_data_pth_config["folder"],
                            f"{video_name}.{self.ctrl_data_pth_config['format']}"
                        ) if self.ctrl_data_pth_config["folder"] is not None else None,
                        "frame_ids": frame_ids
                    })
                    if ctrl_data is None:  # Control data loading failed
                        index = np.random.randint(len(self.video_paths))
                        continue
                    data.update(ctrl_data) 

                    # The ctrl_data above is the 'raw' data loaded (e.g. a loaded segmentation pkl).
                    # Next, we process it into the control input "video" tensor that the model expects.
                    # This is done in the augmentor.
                    for _, aug_fn in self.augmentor.items():
                        data = aug_fn(data)

                return data

            except Exception:
                warnings.warn(
                    f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                    f"(by randomly sampling another sample in the same dataset)."
                )
                warnings.warn("FULL TRACEBACK:")
                warnings.warn(traceback.format_exc())
                if _ == max_retries - 1:
                    raise RuntimeError(f"Failed to load data after {max_retries} attempts")
                index = np.random.randint(len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"


if __name__ == "__main__":
    '''
    Sanity check for the dataset.
    '''
    control_input_key = "control_input_keypoint"
    visualize_control_input = True

    dataset = ExampleTransferDataset(
        dataset_dir="datasets/hdvila/",
        hint_key=control_input_key,
        num_frames=121,
        resolution="720",
        is_train=True
    )
    print("finished init dataset")
    indices = [0, 12, 100, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['frame_start']=}\n"
                f"{data['frame_end']=}\n"
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data[control_input_key].shape=}\n" # should match the video shape
                f"{data['video_name']=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                "---"
            )
        )
        if visualize_control_input:
            import imageio
            control_input_tensor = data[control_input_key].permute(1, 2, 3, 0).cpu().numpy()
            video_name = f"{control_input_key}.mp4"
            imageio.mimsave(video_name, control_input_tensor, fps=24)
