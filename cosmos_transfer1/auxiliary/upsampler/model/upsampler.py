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

import gc
import os
from typing import Union

import torch
from vllm import LLM, SamplingParams

from cosmos_transfer1.checkpoints import COSMOS_UPSAMPLER_CHECKPOINT
from cosmos_transfer1.utils.misc import extract_video_frames, image_to_base64


class PixtralPromptUpsampler:
    def __init__(self, checkpoint_dir: str, offload_prompt_upsampler: bool = False):
        """
        Initializes the Upsampler model.
        Args:
            checkpoint_dir (str): The directory where model checkpoints are stored.
            offload_prompt_upsampler (bool, optional): If True, the upsampler model will not be loaded during initialization. Defaults to False.
        """

        self.offload_prompt_upsampler = offload_prompt_upsampler
        self.checkpoint_dir = checkpoint_dir
        if not self.offload_prompt_upsampler:
            self._load_upsampler_model()

    def _load_upsampler_model(self):
        """
        Loads the upsampler model.
        Sets:
            self.upsampler_model: An instance of VLM initialized with the specified model configuration.
            self.sampling_params: An instance of SamplingParams with predefined parameters.
        """
        model_path = os.path.join(self.checkpoint_dir, COSMOS_UPSAMPLER_CHECKPOINT)

        self.upsampler_model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            tokenizer_mode="mistral",
            gpu_memory_utilization=0.98,
            max_model_len=4096,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=300,
        )

    def _prompt_upsample_with_offload(self, prompt: str, video_path: Union[list, str]):
        """
        Upsamples the given prompt using the upsampler model, with optional model offloading.
        This method first checks if the upsampler model should be offloaded. If so, it loads the model,
        performs the upsampling, and then offloads the model again if necessary.
        Args:
            prompt (str): The prompt to be upsampled.
            image_paths (list): A list of paths to the images associated with the prompt.
        Returns:
            upsampled_prompt: The upsampled version of the input prompt.
        """

        if self.offload_prompt_upsampler:
            self._load_upsampler_model()

        upsampled_prompt = self._prompt_upsample(prompt, video_path)

        if self.offload_prompt_upsampler:
            self._offload_upsampler_model()
        return upsampled_prompt

    def _prompt_upsample(self, prompt: str, video_path: Union[list, str]):
        """
        Generates an upsampled image based on the provided prompt and image paths.
        Args:
            prompt (str): The textual prompt to guide the upsampling process.
            image_paths (list of str): List of file paths to the images to be upsampled.
        Returns:
            str: The text output from the language model after processing the prompt and images.
        """
        prompt = prompt if prompt else "describe the following images"
        image_paths = video_path
        if isinstance(video_path, str):
            image_paths = extract_video_frames(video_path)

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"}}
                    for image_path in image_paths
                ]
                + [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        outputs = self.upsampler_model.chat(
            [message],
            sampling_params=self.sampling_params,
        )
        output = outputs[0].outputs[0].text if outputs[0].outputs[0].text else ""
        return str(output).strip()

    def _offload_upsampler_model(self):
        """
        Offloads the upsampler model from memory.
        This method deletes the `upsampler_model` attribute if it exists, sets it to None,
        triggers garbage collection, and clears the CUDA cache to free up GPU memory.
        """
        if self.upsampler_model:
            del self.upsampler_model
            self.upsampler_model = None
            gc.collect()
            torch.cuda.empty_cache()
