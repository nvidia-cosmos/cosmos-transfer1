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
import pickle
import traceback
import warnings

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from cosmos_transfer1.diffusion.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer1.diffusion.datasets.augmentors.control_input import VIDEO_RES_SIZE_INFO
from cosmos_transfer1.diffusion.inference.inference_utils import detect_aspect_ratio
from cosmos_transfer1.utils.lazy_config import instantiate

# mappings between control types and corresponding sub-folders names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "lidar": {"folder": "lidar", "format": "mp4", "data_dict_key": "lidar"},
    "hdmap": {"folder": "hdmap", "format": "mp4", "data_dict_key": "hdmap"},
    "seg": {"folder": "seg", "format": "pickle", "data_dict_key": "segmentation"},
    "edge": {"folder": None},  # Canny edge, computed on-the-fly
    "vis": {"folder": None},  # Blur, computed on-the-fly
    "upscale": {"folder": None},  # Computed on-the-fly
}


class ExampleTransferDataset(Dataset):
    def __init__(self, dataset_dir, num_frames, resolution, hint_key="control_input_vis", is_train=True):
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
        assert (
            resolution in VIDEO_RES_SIZE_INFO.keys()
        ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        # Control input setup with file formats
        self.ctrl_type = hint_key.replace("control_input_", "")
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
                with open(ctrl_path, "rb") as f:
                    ctrl_data = pickle.load(f)
                # key should match line 982 at cosmos_transfer1/diffusion/datasets/augmentors/control_input.py
                data_dict["segmentation"] = ctrl_data
            elif self.ctrl_type == "keypoint":
                with open(ctrl_path, "rb") as f:
                    ctrl_data = pickle.load(f)
                data_dict["keypoint"] = ctrl_data
            elif self.ctrl_type == "depth":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                # Ensure the depth video has the same number of frames
                assert len(vr) >= frame_ids[-1] + 1, f"Depth video {ctrl_path} has fewer frames than main video"

                # Load the corresponding frames
                depth_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                depth_frames = torch.from_numpy(depth_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                data_dict["depth"] = {
                    "video": depth_frames,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1],
                }
            elif self.ctrl_type == "lidar":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                # Ensure the lidar depth video has the same number of frames
                assert len(vr) >= frame_ids[-1] + 1, f"Lidar video {ctrl_path} has fewer frames than main video"
                # Load the corresponding frames
                lidar_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                lidar_frames = torch.from_numpy(lidar_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                data_dict["lidar"] = {
                    "video": lidar_frames,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1],
                }
            elif self.ctrl_type == "hdmap":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                # Ensure the hdmap video has the same number of frames
                assert len(vr) >= frame_ids[-1] + 1, f"Hdmap video {ctrl_path} has fewer frames than main video"
                # Load the corresponding frames
                hdmap_frames = vr.get_batch(frame_ids).asnumpy()  # [T,H,W,C]
                hdmap_frames = torch.from_numpy(hdmap_frames).permute(3, 0, 1, 2)  # [C,T,H,W], same as rgb video
                data_dict["hdmap"] = {
                    "video": hdmap_frames,
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

                # Load T5 embeddings
                if self.ctrl_type in ["hdmap", "lidar"]:
                    # AV data load captions differently
                    data["video_name"] = {
                        "video_path": video_path,
                        "t5_embedding_path": os.path.join(self.t5_dir, f"{video_name}.pkl"),
                        "start_frame_id": str(frame_ids[0]),
                    }
                    with open(data["video_name"]["t5_embedding_path"], "rb") as f:
                        t5_embedding = pickle.load(f)["pickle"]["ground_truth"]["embeddings"]["t5_xxl"]
                    # Ensure t5_embedding is a numpy array
                    if isinstance(t5_embedding, list):
                        t5_embedding = np.array(t5_embedding[0] if len(t5_embedding) > 0 else t5_embedding)
                    data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)  # .cuda()
                    data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)  # .cuda()
                else:
                    data["video_name"] = {
                        "video_path": video_path,
                        "t5_embedding_path": os.path.join(self.t5_dir, f"{video_name}.pickle"),
                        "start_frame_id": str(frame_ids[0]),
                    }
                    with open(data["video_name"]["t5_embedding_path"], "rb") as f:
                        t5_embedding = pickle.load(f)
                    # Ensure t5_embedding is a numpy array
                    if isinstance(t5_embedding, list):
                        t5_embedding = np.array(t5_embedding[0] if len(t5_embedding) > 0 else t5_embedding)
                    data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)  # .cuda()
                    data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)  # .cuda()

                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([704, 1280, 704, 1280])  # .cuda()
                data["padding_mask"] = torch.zeros(1, 704, 1280)  # .cuda()

                if self.ctrl_type:
                    ctrl_data = self._load_control_data(
                        {
                            "ctrl_path": os.path.join(
                                self.dataset_dir,
                                self.ctrl_data_pth_config["folder"],
                                f"{video_name}.{self.ctrl_data_pth_config['format']}",
                            )
                            if self.ctrl_data_pth_config["folder"] is not None
                            else None,
                            "frame_ids": frame_ids,
                        }
                    )
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
        return

    def __len__(self):
        return len(self.video_paths)

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"


class AVTransferDataset(ExampleTransferDataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        resolution,
        view_keys,
        hint_key="control_input_hdmap",
        sample_n_views=-1,
        caption_view_idx_map=None,
        is_train=True,
        load_mv_emb=False,
    ):
        """Dataset class for loading video-text-to-video generation data with control inputs.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of consecutive frames to load per sequence
            resolution (str): resolution of the target video size
            hint_key (str): The hint key for loading the correct control input data modality
            view_keys (list[str]): list of view names that the dataloader should load
            sample_n_views (int): Number of views to sample
            caption_view_idx_map (dict): Optional dictionary mapping index in view_keys to index in model.view_embeddings
            is_train (bool): Whether this is for training
            load_mv_emb (bool): Whether to load t5 embeddings for all views, or only  for front view
        NOTE: in our example dataset we do not have a validation dataset. The is_train flag is kept here for customized configuration.
        """
        super(ExampleTransferDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = resolution
        self.view_keys = view_keys
        self.load_mv_emb = load_mv_emb
        assert (
            resolution in VIDEO_RES_SIZE_INFO.keys()
        ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        # Control input setup with file formats
        self.ctrl_type = hint_key.replace("control_input_", "")
        self.ctrl_data_pth_config = CTRL_TYPE_INFO[self.ctrl_type]

        # Set up directories - only collect paths
        video_dir = os.path.join(self.dataset_dir, "videos", "pinhole_front")
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        cache_dir = os.path.join(self.dataset_dir, "cache")
        self.prefix_t5_embeddings = {}
        for view_key in view_keys:
            with open(os.path.join(cache_dir, f"prefix_{view_key}.pkl"), "rb") as f:
                self.prefix_t5_embeddings[view_key] = pickle.load(f)
        if caption_view_idx_map is None:
            self.caption_view_idx_map = dict([(i, i) for i in range(len(self.view_keys))])
        else:
            self.caption_view_idx_map = caption_view_idx_map
        self.sample_n_views = sample_n_views

        print(f"Finish initializing dataset with {len(self.video_paths)} videos in total.")

        # Set up preprocessing and augmentation
        augmentor_name = f"video_ctrlnet_augmentor_{hint_key}"
        augmentor_cfg = AUGMENTOR_OPTIONS[augmentor_name](resolution=resolution)
        self.augmentor = {k: instantiate(v) for k, v in augmentor_cfg.items()}

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:
                video_path = self.video_paths[index]
                video_name = os.path.basename(video_path).replace(".mp4", "")

                data = dict()
                ctrl_videos = []
                videos = []
                t5_embeddings = []
                t5_masks = []
                view_indices = [i for i in range(len(self.view_keys))]
                view_indices_conditioning = []
                if self.sample_n_views > 1:
                    sampled_idx = np.random.choice(
                        np.arange(1, len(view_indices)),
                        size=min(self.sample_n_views - 1, len(view_indices) - 1),
                        replace=False,
                    )
                    sampled_idx = np.concatenate(
                        [
                            [
                                0,
                            ],
                            sampled_idx,
                        ]
                    )
                    sampled_idx.sort()
                    view_indices = sampled_idx.tolist()

                frame_ids = None
                fps = None
                for view_index in view_indices:
                    view_key = self.view_keys[view_index]
                    if frame_ids is None:
                        frames, frame_ids, fps = self._sample_frames(video_path)
                        if frames is None:  # Invalid video or too short
                            raise Exception(f"Failed to load frames {video_path}")

                    else:
                        frames, fps = self._load_video(
                            os.path.join(self.dataset_dir, "videos", view_key, os.path.basename(video_path)), frame_ids
                        )
                    # Process video frames
                    video = torch.from_numpy(frames)

                    video = video.permute(3, 0, 1, 2)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                    aspect_ratio = detect_aspect_ratio((video.shape[3], video.shape[2]))  # expects (W, H)
                    videos.append(video)

                    if video_name[-2] == "_" and video_name[-1].isdigit():
                        video_name_emb = video_name[:-2]
                    else:
                        video_name_emb = video_name

                    if self.load_mv_emb or view_key == "pinhole_front":
                        t5_embedding_path = os.path.join(self.dataset_dir, "t5_xxl", view_key, f"{video_name_emb}.pkl")
                        with open(t5_embedding_path, "rb") as f:
                            t5_embedding = pickle.load(f)[0]
                        if self.load_mv_emb:
                            t5_embedding = np.concatenate([self.prefix_t5_embeddings[view_key], t5_embedding], axis=0)
                    else:
                        # use camera prompt
                        t5_embedding = self.prefix_t5_embeddings[view_key]

                    t5_embedding = torch.from_numpy(t5_embedding)
                    t5_mask = torch.ones(t5_embedding.shape[0], dtype=torch.int64)
                    if t5_embedding.shape[0] < 512:
                        t5_embedding = torch.cat([t5_embedding, torch.zeros(512 - t5_embedding.shape[0], 1024)], dim=0)
                        t5_mask = torch.cat([t5_mask, torch.zeros(512 - t5_mask.shape[0])], dim=0)
                    else:
                        t5_embedding = t5_embedding[:512]
                        t5_mask = t5_mask[:512]
                    t5_embeddings.append(t5_embedding)
                    t5_masks.append(t5_mask)
                    caption_viewid = self.caption_view_idx_map[view_index]
                    view_indices_conditioning.append(torch.ones(video.shape[1]) * caption_viewid)

                    if self.ctrl_type:
                        v_ctrl_data = self._load_control_data(
                            {
                                "ctrl_path": os.path.join(
                                    self.dataset_dir,
                                    self.ctrl_data_pth_config["folder"],
                                    view_key,
                                    f"{video_name}.{self.ctrl_data_pth_config['format']}",
                                )
                                if self.ctrl_data_pth_config["folder"] is not None
                                else None,
                                "frame_ids": frame_ids,
                            }
                        )
                        if v_ctrl_data is None:  # Control data loading failed
                            raise Exception("Failed to load v_ctrl_data")
                        ctrl_videos.append(v_ctrl_data[self.ctrl_type]["video"])

                video = torch.cat(videos, dim=1)
                ctrl_videos = torch.cat(ctrl_videos, dim=1)
                t5_embedding = torch.cat(t5_embeddings, dim=0)
                view_indices_conditioning = torch.cat(view_indices_conditioning, dim=0)

                # Basic data
                data["video"] = video
                data["video_name"] = video_name
                data["aspect_ratio"] = aspect_ratio
                data["t5_text_embeddings"] = t5_embedding
                data["t5_text_mask"] = torch.cat(t5_masks)
                data["view_indices"] = view_indices_conditioning.contiguous()
                data["frame_repeat"] = torch.zeros(len(view_indices))
                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([704, 1280, 704, 1280])
                data["padding_mask"] = torch.zeros(1, 704, 1280)
                data[self.ctrl_type] = dict()
                data[self.ctrl_type]["video"] = ctrl_videos

                # The ctrl_data above is the 'raw' data loaded (e.g. a loaded lidar pkl).
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
        return


if __name__ == "__main__":
    """
    Sanity check for the dataset.
    """
    control_input_key = "control_input_lidar"
    visualize_control_input = True

    dataset = AVTransferDataset(
        dataset_dir="datasets/waymo_transfer1",
        view_keys=["pinhole_front"],
        hint_key=control_input_key,
        num_frames=121,
        resolution="720",
        is_train=True,
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
                f"{data[control_input_key].shape=}\n"  # should match the video shape
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
