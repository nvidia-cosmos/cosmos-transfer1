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

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Workaround to suppress MP warning

import sys
from io import BytesIO

import torch

from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_AV_SAMPLE_PATH, BASE_7B_CHECKPOINT_PATH
from cosmos_transfer1.diffusion.inference.inference_utils import load_controlnet_specs, validate_controlnet_specs
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import DiffusionControl2WorldGenerationPipeline
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.io import read_prompts_from_file, save_video

torch.enable_grad(False)
torch.serialization.add_safe_globals([BytesIO])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control to world generation demo script", conflict_handler="resolve")

    # Add transfer specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        help="prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        help="negative prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="",
        help="Optional input RGB video path",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of conditional frames for long video generation",
        choices=[1],
    )
    parser.add_argument("--sigma_max", type=float, default=70.0, help="sigma_max for partial denoising")
    parser.add_argument(
        "--blur_strength",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength.",
    )
    parser.add_argument(
        "--canny_threshold",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="blur strength of canny threshold applied to input. Lower means less blur or more detected edges, which means higher fidelity to input.",
    )
    parser.add_argument(
        "--controlnet_specs",
        type=str,
        help="Path to JSON file specifying multicontrolnet configurations",
        required=True,
    )
    parser.add_argument(
        "--is_av_sample", action="store_true", help="Whether the model is an driving post-training model"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="Cosmos-Tokenize1-CV8x8x8-720p",
        help="Tokenizer weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--video_save_name",
        type=str,
        default="output",
        help="Output filename for generating a single video",
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default="outputs/",
        help="Output folder for generating a batch of videos",
    )
    parser.add_argument(
        "--batch_input_path",
        type=str,
        help="Path to a JSONL file of input prompts for generating a batch of videos",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--guidance", type=float, default=5, help="Classifier-free guidance scale value")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs used to run inference in parallel.")
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload DiT after inference",
    )
    parser.add_argument(
        "--offload_text_encoder_model",
        action="store_true",
        help="Offload text encoder model after inference",
    )
    parser.add_argument(
        "--offload_guardrail_models",
        action="store_true",
        help="Offload guardrail models after inference",
    )
    parser.add_argument(
        "--upsample_prompt",
        action="store_true",
        help="Upsample prompt using Pixtral upsampler model",
    )
    parser.add_argument(
        "--offload_prompt_upsampler",
        action="store_true",
        help="Offload prompt upsampler model after inference",
    )

    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)

    log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")
    log.info(f"args in json: {json.dumps(json_args, indent=4)}")
    # if parameters not set on command line, use the ones from the controlnet_specs
    # if both not set use command line defaults
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    log.info(f"final args: {json.dumps(vars(cmd_args), indent=4)}")

    return cmd_args, control_inputs


def demo(cfg, control_inputs):
    """Run control-to-world generation demo.

    This function handles the main control-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """

    control_inputs = validate_controlnet_specs(cfg, control_inputs)
    misc.set_random_seed(cfg.seed)

    device_rank = 0
    process_group = None
    if cfg.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_transfer1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=cfg.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

        device_rank = distributed.get_rank(process_group)

    preprocessors = Preprocessors()

    checkpoint = BASE_7B_CHECKPOINT_AV_SAMPLE_PATH if cfg.is_av_sample else BASE_7B_CHECKPOINT_PATH

    # Initialize transfer generation model pipeline
    pipeline = DiffusionControl2WorldGenerationPipeline(
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=checkpoint,
        offload_network=cfg.offload_diffusion_transformer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        fps=cfg.fps,
        seed=cfg.seed,
        num_input_frames=cfg.num_input_frames,
        control_inputs=control_inputs,
        sigma_max=cfg.sigma_max,
        blur_strength=cfg.blur_strength,
        canny_threshold=cfg.canny_threshold,
        upsample_prompt=cfg.upsample_prompt,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        process_group=process_group,
    )

    if cfg.batch_input_path:
        log.info(f"Reading batch inputs from path: {cfg.batch_input_path}")
        prompts = read_prompts_from_file(cfg.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": cfg.prompt, "visual_input": cfg.input_video_path}]

    batch_size = cfg.batch_size if hasattr(cfg, "batch_size") else 1
    os.makedirs(cfg.video_save_folder, exist_ok=True)
    for batch_start in range(0, len(prompts), batch_size):
        # Get current batch
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        actual_batch_size = len(batch_prompts)
        # Extract batch data
        batch_prompt_texts = [p.get("prompt", None) for p in batch_prompts]
        batch_video_paths = [p.get("visual_input", None) for p in batch_prompts]

        batch_control_inputs = []
        for i, input_dict in enumerate(batch_prompts):
            current_prompt = input_dict.get("prompt", None)
            current_video_path = input_dict.get("visual_input", None)

            if cfg.batch_input_path:
                video_save_subfolder = os.path.join(cfg.video_save_folder, f"video_{batch_start+i}")
                os.makedirs(video_save_subfolder, exist_ok=True)
            else:
                video_save_subfolder = cfg.video_save_folder

            current_control_inputs = copy.deepcopy(control_inputs)
            if "control_overrides" in input_dict:
                for hint_key, override in input_dict["control_overrides"].items():
                    if hint_key in current_control_inputs:
                        current_control_inputs[hint_key].update(override)
                    else:
                        log.warning(f"Ignoring unknown control key in override: {hint_key}")

            # if control inputs are not provided, run respective preprocessor (for seg and depth)
            log.info("running preprocessor")
            preprocessors(current_video_path, current_prompt, current_control_inputs, video_save_subfolder, cfg.regional_prompts if hasattr(cfg, "regional_prompts") else None)
            batch_control_inputs.append(current_control_inputs)

        regional_prompts = []
        region_definitions = []
        if hasattr(cfg, "regional_prompts") and cfg.regional_prompts:
            log.info(f"regional_prompts: {cfg.regional_prompts}")
            for regional_prompt in cfg.regional_prompts:
                regional_prompts.append(regional_prompt["prompt"])
                if "region_definitions_path" in regional_prompt:
                    log.info(f"region_definitions_path: {regional_prompt['region_definitions_path']}")
                    region_definition_path = regional_prompt["region_definitions_path"]
                    if isinstance(region_definition_path, str) and region_definition_path.endswith(".json"):
                        with open(region_definition_path, "r") as f:
                            region_definitions_json = json.load(f)
                        region_definitions.extend(region_definitions_json)
                    else:
                        region_definitions.append(region_definition_path)

        if hasattr(pipeline, "regional_prompts"):
            pipeline.regional_prompts = regional_prompts
        if hasattr(pipeline, "region_definitions"):
            pipeline.region_definitions = region_definitions

        # Generate videos in batch
        batch_outputs = pipeline.generate(
            prompt=batch_prompt_texts,
            video_path=batch_video_paths,
            negative_prompt=cfg.negative_prompt,
            control_inputs=batch_control_inputs,
            save_folder=video_save_subfolder,
            batch_size=actual_batch_size,
        )
        if batch_outputs is None:
            log.critical("Guardrail blocked generation for entire batch.")
            continue

        videos, final_prompts = batch_outputs
        for i, (video, prompt) in enumerate(zip(videos, final_prompts)):
            if cfg.batch_input_path:
                video_save_subfolder = os.path.join(cfg.video_save_folder, f"video_{batch_start+i}")
                video_save_path = os.path.join(video_save_subfolder, "output.mp4")
                prompt_save_path = os.path.join(video_save_subfolder, "prompt.txt")
            else:
                video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
                prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")
            # Save video and prompt
            if device_rank == 0:
                os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
                save_video(
                    video=video,
                    fps=cfg.fps,
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

    # clean up properly
    if cfg.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args, control_inputs = parse_arguments()
    demo(args, control_inputs)
