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


import argparse
import json
import os

from cosmos_transfer1.utils import log
import copy
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
        self.update_controlnet_spec(
            checkpoint_dir="/mnt/pvc/cosmos-transfer1",
            controlnet_specs=controlnet_specs,
        )

        prompt_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.txt")

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved prompt to {prompt_save_path}")

    valid_hint_keys = {"vis", "seg", "edge", "depth", "keypoint", "upscale", "hdmap", "lidar"}

    def update_controlnet_spec(
        self,
        checkpoint_dir: str,
        controlnet_specs: dict = {},
    ):
        """
        Create the controlnet specification defines which control netwworks are active.
        Note that controlnets are active even if the weights are set to 0."""

        config_changed = False

        for hint_key in self.valid_hint_keys:
            if hint_key in controlnet_specs:
                if hint_key not in self.control_inputs:
                    config_changed = True

                # overwrite old parameters
                self.control_inputs[hint_key] = copy.deepcopy(controlnet_specs[hint_key])
                self.control_inputs[hint_key]["ckpt_path"] = os.path.join(checkpoint_dir, default_model_names[hint_key])
            elif hint_key in self.control_inputs:
                # remove old parameters
                del self.control_inputs[hint_key]
                config_changed = True

        log.info(f"{config_changed=}, control_inputs: {json.dumps(self.control_inputs, indent=4)}")

        return config_changed

    @staticmethod
    def validate_params(
        controlnet_specs,
        input_video=None,
        prompt="test prompt",
        negative_prompt=None,
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
    ):
        """
        advanced parameter check
        """
        args = argparse.Namespace()

        if sigma_max < 80 and not input_video:
            raise ValueError("Must have 'input_video' specified if sigma_max < 80")

        # Video and prompt settings
        args_dict = {}
        if input_video:
            args_dict["input_video"] = input_video
        if prompt:
            args_dict["prompt"] = prompt
        if negative_prompt:
            args_dict["negative_prompt"] = negative_prompt

        # Generation parameters
        args_dict["guidance"] = guidance
        args_dict["num_steps"] = num_steps
        args_dict["seed"] = seed
        args_dict["sigma_max"] = sigma_max
        args_dict["blur_strength"] = blur_strength
        args_dict["canny_threshold"] = canny_threshold

        valid_hint_keys = {"vis", "seg", "edge", "depth", "keypoint", "upscale", "hdmap", "lidar"}

        controlnet_specs_clean = {}

        for hint_key, val in controlnet_specs.items():
            if hint_key in valid_hint_keys:
                controlnet_specs_clean[hint_key] = val
            else:
                if type(val) == dict:
                    raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")
                else:
                    log.warning(
                        f"parameter '{hint_key}' in controlnet_spec will be ignored. parameter already set by UI."
                    )
                    # args_dict[hint_key] = val

        for hint_key, config in controlnet_specs_clean.items():
            if "control_weight" not in config:
                log.warning(f"No control weight specified for {hint_key}. Setting to 0.5.")
                config["control_weight"] = "0.5"
            else:
                # Check if control weight is a path or a scalar
                weight = config["control_weight"]
                if not isinstance(weight, str) or not weight.endswith(".pt"):
                    try:
                        # Try converting to float
                        scalar_value = float(weight)
                        if scalar_value < 0:
                            raise ValueError(f"Control weight for {hint_key} must be non-negative.")
                    except ValueError:
                        raise ValueError(
                            f"Control weight for {hint_key} must be a valid non-negative float or a path to a .pt file."
                        )

        args_dict["controlnet_specs"] = controlnet_specs_clean
        log.info(f"Model parameters: {json.dumps(args_dict, indent=4)}")

        # TODO
        # if edge_weight > 0 and not input_video:
        #     raise ValueError("Edge controlnet must have 'input_video' specified if no 'input_control' video specified.")

        # if seg_weight > 0 and not input_video:
        #     raise ValueError(
        #         "Segment controlnet must have 'input_video' specified if no 'input_control' video specified."
        #     )
        # Regardless whether "control_weight_prompt" is provided (i.e. whether we automatically
        # generate spatiotemporal control weight binary masks), control_weight is needed to.

        return args_dict


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


if __name__ == "__main__":
    controlnet_specs = get_spec("assets/inference_cosmos_transfer1_uniform_weights_auto.json")

    pipeline = TransferPipeline(num_gpus=int(os.environ.get("NUM_GPU", 1)))
    model_params = TransferPipeline.validate_params(
        input_video="assets/example1_input_video.mp4",
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_single_control_edge.json"),
    )
    pipeline.infer(model_params)

    model_params = TransferPipeline.validate_params(
        input_video="assets/example1_input_video.mp4",
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_multi_control.json"),
    )
    pipeline.infer(model_params)
