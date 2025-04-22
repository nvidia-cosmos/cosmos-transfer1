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

from typing import Optional

import omegaconf
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_transfer1.diffusion.datasets.augmentors.control_input import Augmentor
from cosmos_transfer1.diffusion.datasets.dataset_utils import obtain_augmentation_size, obtain_image_size


class ReflectionPadding(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs reflection padding. This function also returns a padding mask.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """

        assert self.args is not None, "Please specify args in augmentation"
        if self.output_keys is None:
            self.output_keys = self.input_keys

        # Obtain image and augmentation sizes
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        target_size = obtain_augmentation_size(data_dict, self.args)

        assert isinstance(target_size, (tuple, omegaconf.listconfig.ListConfig)), "Please specify target size as tuple"
        target_w, target_h = target_size

        target_w = int(target_w)
        target_h = int(target_h)

        # Calculate padding vals
        padding_left = int((target_w - orig_w) / 2)
        padding_right = target_w - orig_w - padding_left
        padding_top = int((target_h - orig_h) / 2)
        padding_bottom = target_h - orig_h - padding_top
        padding_vals = [padding_left, padding_top, padding_right, padding_bottom]

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            if max(padding_vals[0], padding_vals[2]) >= orig_w or max(padding_vals[1], padding_vals[3]) >= orig_h:
                # In this case, we can't perform reflection padding. This is because padding values
                # are larger than the image size. So, perform edge padding instead.
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="edge")
            else:
                # Perform reflection padding
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="reflect")

            if out_key != inp_key:
                del data_dict[inp_key]

        # Return padding_mask when padding is performed.
        # Padding mask denotes which pixels are padded.
        padding_mask = torch.ones((1, target_h, target_w))
        padding_mask[:, padding_top : (padding_top + orig_h), padding_left : (padding_left + orig_w)] = 0
        data_dict["padding_mask"] = padding_mask
        data_dict["image_size"] = torch.tensor([target_h, target_w, orig_h, orig_w], dtype=torch.float)

        return data_dict


class ResizeSmallestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to smaller side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=out_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to larger side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

            scaling_ratio = min(out_size / orig_w, out_size / orig_h)
            target_size = [int(scaling_ratio * orig_h), int(scaling_ratio * orig_w)]

            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(
            img_size, (tuple, omegaconf.listconfig.ListConfig)
        ), f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] >= img_h and target_size[1] >= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the larger ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the larger of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(
            img_size, (tuple, omegaconf.listconfig.ListConfig)
        ), f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] <= img_h and target_size[1] <= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict
