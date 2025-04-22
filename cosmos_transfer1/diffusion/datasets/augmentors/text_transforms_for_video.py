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

import random
from typing import Optional

import numpy as np
import torch

from cosmos_transfer1.diffusion.datasets.augmentors.control_input import Augmentor
from cosmos_transfer1.utils import log


def pad_and_resize(
    arr_np: np.ndarray, ntokens: int, is_mask_all_ones: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Function for padding and resizing a numpy array.
    Args:
        arr (np.ndarray): Input array
        ntokens (int): Number of output tokens after padding
        is_mask_all_ones (bool): if true, set mask to ones
    Returns:
        arr_padded (torch.Tensor): Padded output tensor
        mask (torch.Tensor): Padding mask
    """

    if isinstance(arr_np, np.ndarray):
        arr = torch.from_numpy(arr_np)
    elif isinstance(arr_np, torch.Tensor):
        arr = arr_np.clone().detach()
    else:
        raise TypeError("`arr_np` should be a numpy array or torch tensor.")
    embed_dim = arr.shape[1]

    arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

    # If the input text is larger than num_text_tokens, clip it.
    if arr.shape[0] > ntokens:
        arr = arr[0:ntokens]

    mask = torch.LongTensor(ntokens).zero_()
    if len(arr.shape) > 1:
        mask[0 : arr.shape[0]] = 1

    if len(arr.shape) > 1:
        arr_padded[0 : arr.shape[0]] = arr

    if is_mask_all_ones:
        mask.fill_(1)

    return arr_padded, mask


class TextTransformForVideo(Augmentor):
    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        data_source = data_dict["__url__"].meta.source
        input_keys_by_source = self.input_keys[data_source]

        if "chunk_index" not in data_dict:
            log.warning(
                "Chunk_index is not in data_dict, set chunk_index to be 0. This should only happen for sampling."
            )
            data_dict["chunk_index"] = 0  # this is for sampling only, whereas decoder is not loaded
        try:
            windows = data_dict[input_keys_by_source["ai_caption"]]["windows"]
            n_windows = len(windows)
            chunk_index = data_dict["chunk_index"]

            if chunk_index == n_windows:
                # This will only happen when the number of captions does not match number of chunks due to re-transcoding the videos.
                log.info(
                    f"Found {data_dict['orig_num_frames']} in video but captioning is done with videos of {windows[-1]['end_frame']} frames. This mismatch is due to video re-transcoding.",
                    rank0_only=False,
                )
                chunk_index -= 1

            selected_caption_window = windows[chunk_index]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}, {data_dict['chunk_index']}\n error {e}",
                rank0_only=False,
            )
            return None

        try:
            if "vila_caption" in selected_caption_window:
                caption_type = "vila_caption"
            else:
                caption_type = random.choices(["long_caption", "short_caption"], weights=[0.95, 0.05], k=1)[0]
            data_dict["ai_caption"] = selected_caption_window[caption_type]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}, {selected_caption_window}\n error {e}",
                rank0_only=False,
            )
            return None

        if data_dict["ai_caption"] is None:
            data_dict["ai_caption"] = ""
        del data_dict[input_keys_by_source["ai_caption"]]

        ai_caption_embedding_data = data_dict[input_keys_by_source["ai_caption_embedding"]]
        try:
            if caption_type in ["vila_caption"]:
                t5_embedding = ai_caption_embedding_data[data_dict["chunk_index"]]
            else:
                t5_embedding = ai_caption_embedding_data[data_dict["chunk_index"]][
                    caption_type.replace("_caption", "")
                ]  # t5_embedding is saved in {"short": array, "long": array} format
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}, {data_dict['chunk_index']}, {len(ai_caption_embedding_data)} \n error {e}",
                rank0_only=False,
            )
            return None
        out_t5, out_t5_mask = pad_and_resize(
            t5_embedding,
            self.args["t5_tokens"]["num"],
            is_mask_all_ones=self.args["is_mask_all_ones"],
        )
        data_dict["t5_text_embeddings"] = out_t5
        data_dict["t5_text_mask"] = out_t5_mask
        del data_dict[input_keys_by_source["ai_caption_embedding"]]

        return data_dict
