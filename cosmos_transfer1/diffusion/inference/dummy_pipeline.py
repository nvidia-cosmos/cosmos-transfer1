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


import os

from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferValidator
from cosmos_transfer1.utils import log
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names


class TransferPipeline:
    def __init__(
        self, num_gpus: int = 1, checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1", output_dir: str = "outputs/"
    ):

        self.pipeline = None
        self.preprocessors = None
        self.device_rank = 0
        self.process_group = None

        self.output_dir = output_dir
        self.video_save_name = "output"
        self.control_inputs = {
            "vis": {
                "ckpt_path": os.path.join(checkpoint_dir, default_model_names["vis"]),
                "control_weight": 0.5,
            },
        }

    def infer(self, args: dict):
        return self.generate(**args)

    def generate(
        self,
        controlnet_specs,
        input_video=None,
        prompt="test prompt",
        negative_prompt="test negative prompt",
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
    ):

        prompt_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.txt")

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved prompt to {prompt_save_path}")


def create_dummy_pipeline(cfg):
    log.info("Creating dummy pipeline for testing")
    return TransferPipeline(num_gpus=1, output_dir=cfg.output_dir), TransferValidator()
