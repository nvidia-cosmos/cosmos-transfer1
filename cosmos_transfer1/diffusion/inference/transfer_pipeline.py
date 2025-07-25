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
import copy
import json
import os


from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import default_model_names
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline
from cosmos_transfer1.utils import log
from cosmos_transfer1.utils.io import save_video

"""
pipeline class similar to demo function.
we can't use the demo function as the function discards the underlying pipeline after each inference.

This class creates the pipeline during initialization and keeps it alive for multiple inferences.
The pipeline is configured for to NOT offload the models in favor of speed.

For now we assume a fixed configuration of the controlnets for the lifetime of the pipeline.
This makes the config.json file for the controlnets a init parameter of the pipeline.

TODO regional prompt support

TODO av support

"""

# todo "keypoint" is causing dependency issue
valid_hint_keys = {"vis", "seg", "edge", "depth", "upscale", "hdmap", "lidar"}
default_prompt = "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution."
default_negative_prompt = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."


class TransferPipeline:
    def __init__(
        self,
        num_gpus: int = 1,
        checkpoint_dir: str = "/mnt/pvc/cosmos-transfer1",
        output_dir: str = "outputs/",
    ):
        self.device_rank = 0
        self.process_group = None

        self.preprocessors = Preprocessors()

        if num_gpus > 1:
            from megatron.core import parallel_state
            from cosmos_transfer1.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
            self.process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(self.process_group)

        self.control_inputs = {
            "vis": {
                "ckpt_path": os.path.join(checkpoint_dir, default_model_names["vis"]),
                "control_weight": 0.5,
            },
        }

        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.video_save_name = "output"

        self.pipeline = DiffusionControl2WorldGenerationPipeline(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=BASE_7B_CHECKPOINT_PATH,
            control_inputs=self.control_inputs,
            process_group=self.process_group,
            offload_network=False,
            offload_text_encoder_model=False,
            offload_guardrail_models=False,
            offload_prompt_upsampler=False,
            upsample_prompt=False,
            fps=24,
            num_input_frames=24,
            disable_guardrail=True,
        )

    def update_controlnet_spec(
        self,
        checkpoint_dir: str,
        controlnet_specs: dict,
    ):
        """
        Create the controlnet specification defines which control netwworks are active.
        Note that controlnets are active even if the weights are set to 0."""

        config_changed = False

        for hint_key in valid_hint_keys:
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

    def infer(self, args: dict):
        return self.generate(**args)

    def generate(
        self,
        controlnet_specs,
        input_video=None,
        prompt="",
        negative_prompt="",
        guidance=5,
        num_steps=35,
        seed=1,
        sigma_max=70.0,
        blur_strength="medium",
        canny_threshold="medium",
    ):

        config_changed = self.update_controlnet_spec(
            checkpoint_dir=self.checkpoint_dir,
            controlnet_specs=controlnet_specs,
        )

        if config_changed:
            self.pipeline.reload_model(self.control_inputs)

        # original code is creating deepcopy. are values touched?
        # TODO add control weights as inference parameter
        current_control_inputs = copy.deepcopy(self.control_inputs)
        log.info(f"current_control_inputs: {json.dumps(current_control_inputs, indent=4)}")

        log.info("Running preprocessor")
        self.preprocessors(
            input_video,
            prompt,
            current_control_inputs,
            self.output_dir,
        )

        # TODO: add support for regional prompts and region definitions
        if hasattr(self.pipeline, "regional_prompts"):
            self.pipeline.regional_prompts = []
        if hasattr(self.pipeline, "region_definitions"):
            self.pipeline.region_definitions = []

        # WAR these inference parameters are for unknown reasons not part of the generate function
        self.pipeline.guidance = guidance
        self.pipeline.num_steps = num_steps
        self.pipeline.seed = seed
        self.pipeline.sigma_max = sigma_max
        self.pipeline.blur_strength = blur_strength
        self.pipeline.canny_threshold = canny_threshold

        # self.pipeline.control_inputs = current_control_inputs

        batch_outputs = self.pipeline.generate(
            prompt=[prompt],
            video_path=[input_video],
            negative_prompt=negative_prompt,
            control_inputs=[current_control_inputs],
            save_folder=self.output_dir,
            batch_size=1,
        )
        if batch_outputs is None:
            log.critical("Guardrail blocked generation for entire batch.")
        elif self.device_rank == 0:
            videos, final_prompts = batch_outputs
            for i, (video, prompt) in enumerate(zip(videos, final_prompts)):

                video_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.mp4")
                prompt_save_path = os.path.join(self.output_dir, f"{self.video_save_name}.txt")
                os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

                save_video(
                    video=video,
                    fps=self.pipeline.fps,
                    H=video.shape[1],
                    W=video.shape[2],
                    video_save_quality=5,
                    video_save_path=video_save_path,
                )

                # Save prompt to text file alongside video
                with open(prompt_save_path, "wb") as f:
                    f.write(prompt.encode("utf-8"))

                log.info(f"Saved video to {video_save_path}")
                log.info(f"Saved prompt to {prompt_save_path}")

    @staticmethod
    def validate_params(
        controlnet_specs,
        input_video_path=None,
        prompt=default_prompt,
        negative_prompt=default_negative_prompt,
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

        if sigma_max < 80 and not input_video_path:
            raise ValueError("Must have 'input_video' specified if sigma_max < 80")

        # Video and prompt settings
        args_dict = {}
        if input_video_path:
            args_dict["input_video"] = input_video_path
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

    def cleanup(self, cfg):
        """Clean up resources"""
        if cfg.num_gpus > 1:
            from megatron.core import parallel_state
            import torch.distributed as dist

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


if __name__ == "__main__":
    pipeline = TransferPipeline(num_gpus=int(os.environ.get("NUM_GPU", 1)))
    model_params = TransferPipeline.validate_params(
        input_video="assets/example1_input_video.mp4",
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_single_control_depth.json"),
    )
    pipeline.infer(model_params)

    model_params = TransferPipeline.validate_params(
        input_video="assets/example1_input_video.mp4",
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_single_control_edge.json"),
    )
    pipeline.infer(model_params)

    log.info("Inference complete****************************************")

    model_params = TransferPipeline.validate_params(
        input_video="assets/example1_input_video.mp4",
        controlnet_specs=get_spec("assets/inference_cosmos_transfer1_multi_control.json"),
    )
    pipeline.infer(model_params)
