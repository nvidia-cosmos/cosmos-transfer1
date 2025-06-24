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

from cosmos_transfer1.checkpoints import (
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
)
from cosmos_transfer1.diffusion.inference.inference_utils import (
    default_model_names,
    load_controlnet_specs,
    valid_hint_keys,
)
from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from cosmos_transfer1.diffusion.inference.world_generation_pipeline import (
    DiffusionControl2WorldMultiviewGenerationPipeline,
)
from cosmos_transfer1.utils import log, misc
from cosmos_transfer1.utils.io import save_video

torch.enable_grad(False)

from cosmos_transfer1.checkpoints import (
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_7B_CHECKPOINT_PATH,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH,
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH,
    SV2MV_v2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH,
)
from cosmos_transfer1.diffusion.model.model_ctrl import VideoDiffusionModelWithCtrl, VideoDiffusionT2VModelWithCtrl
from cosmos_transfer1.diffusion.model.model_multi_camera_ctrl import MultiVideoDiffusionModelWithCtrl

MODEL_CLASS_DICT = {
    BASE_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionModelWithCtrl,
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: VideoDiffusionT2VModelWithCtrl,
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: VideoDiffusionT2VModelWithCtrl,
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: MultiVideoDiffusionModelWithCtrl,
}

MODEL_NAME_DICT = {
    BASE_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    EDGE2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
    VIS2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_vis_block3",
    DEPTH2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3",
    KEYPOINT2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_keypoint_block3",
    SEG2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_seg_block3",
    UPSCALER_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_lvg_tp_121frames_control_input_upscale_block3",
    BASE_7B_CHECKPOINT_AV_SAMPLE_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_hdmap_block3",
    LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_t2v_121frames_control_input_lidar_block3",
    BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_block3",
    BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH: "CTRL_7Bv1pt3_sv2mv_v2w_57frames_control_input_hdmap_block3",
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_block3",
    SV2MV_t2w_LIDAR2WORLD_CONTROLNET_7B_CHECKPOINT_PATH: "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_lidar_block3",
    SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: "CTRL_7Bv1pt3_sv2mv_t2w_57frames_control_input_hdmap_waymo_block3",
    SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH: "CTRL_7Bv1pt3_sv2mv_v2w_57frames_control_input_hdmap_waymo_block3",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control to world generation demo script", conflict_handler="resolve")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        help="prompt which the sampled video condition on",
    )
    parser.add_argument(
        "--prompt_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the left. ",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the right.",
        help="Text prompt for generating right camera view video",
    )

    parser.add_argument(
        "--prompt_back",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing backwards.",
        help="Text prompt for generating rear camera view video",
    )
    parser.add_argument(
        "--prompt_back_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_back_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
        help="Text prompt for generating right camera view video",
    )
    parser.add_argument(
        "--view_condition_video",
        type=str,
        default="",
        help="We require that only a single condition view is specified and this video is treated as conditioning for that view. "
        "This video/videos should have the same duration as control videos",
    )
    parser.add_argument(
        "--initial_condition_video",
        type=str,
        default="",
        help="Can be either a path to a mp4 or a directory. If it is a mp4, we assume"
        "that it is a video temporally concatenated with the same number of views as the model. "
        "If it is a directory, we assume that the file names evaluate to integers that correspond to a view index,"
        " e.g. '000.mp4', '003.mp4', '004.mp4'."
        "This video/videos should have at least num_input_frames number of frames for each view. Frames will be taken from the back"
        "of the video(s) if the duration of the video in each view exceed num_input_frames",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of conditional frames for long video generation, not used in t2w",
        choices=[1, 9],
    )
    parser.add_argument(
        "--controlnet_specs",
        type=str,
        help="Path to JSON file specifying multicontrolnet configurations",
        required=True,
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
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--guidance", type=float, default=5, help="Classifier-free guidance scale value")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n_clip_max", type=int, default=-1, help="Maximum number of video extension loop")
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
    parser.add_argument(
        "--waymo_example",
        type=bool,
        default=False,
        help="Set to true when using post-trained checkpoint from the Waymo post-training example",
    )

    cmd_args = parser.parse_args()

    # Load and parse JSON input
    control_inputs, json_args = load_controlnet_specs(cmd_args)
    control_inputs.update(json_args)
    log.info(f"control_inputs: {json.dumps(control_inputs, indent=4)}")
    log.info(f"args in json: {json.dumps(json_args, indent=4)}")

    # if parameters not set on command line, use the ones from the controlnet_specs
    # if both not set use command line defaults
    for key in json_args:
        if f"--{key}" not in sys.argv:
            setattr(cmd_args, key, json_args[key])

    log.info(f"final args: {json.dumps(vars(cmd_args), indent=4)}")

    return cmd_args, control_inputs


def validate_controlnet_specs(cfg, controlnet_specs):
    """
    Load and validate controlnet specifications from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing controlnet specs.
        checkpoint_dir (str): Base directory for checkpoint files.

    Returns:
        Dict[str, Any]: Validated and processed controlnet specifications.
    """
    checkpoint_dir = cfg.checkpoint_dir

    for hint_key, config in controlnet_specs.items():
        if hint_key not in list(valid_hint_keys) + ["prompts", "view_condition_video"]:
            raise ValueError(f"Invalid hint_key: {hint_key}. Must be one of {valid_hint_keys}")
        if hint_key in valid_hint_keys:
            if "ckpt_path" not in config:
                log.info(f"No checkpoint path specified for {hint_key}. Using default.")
                config["ckpt_path"] = os.path.join(checkpoint_dir, default_model_names[hint_key])

            # Regardless whether "control_weight_prompt" is provided (i.e. whether we automatically
            # generate spatiotemporal control weight binary masks), control_weight is needed to.
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

    return controlnet_specs


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

    if cfg.waymo_example:
        prompts = [
            cfg.prompt,
            cfg.prompt_left,
            cfg.prompt_right,
            cfg.prompt_back_left,
            cfg.prompt_back_right,
        ]
        if cfg.initial_condition_video:
            cfg.is_lvg_model = True
            checkpoint = SV2MV_v2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH
        else:
            cfg.is_lvg_model = False
            cfg.num_input_frames = 0
            checkpoint = SV2MV_t2w_HDMAP2WORLD_CONTROLNET_7B_WAYMO_CHECKPOINT_PATH

    else:
        prompts = [
            cfg.prompt,
            cfg.prompt_left,
            cfg.prompt_right,
            cfg.prompt_back,
            cfg.prompt_back_left,
            cfg.prompt_back_right,
        ]

        if cfg.initial_condition_video:
            cfg.is_lvg_model = True
            checkpoint = BASE_v2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH
        else:
            cfg.is_lvg_model = False
            cfg.num_input_frames = 0
            checkpoint = BASE_t2w_7B_SV2MV_CHECKPOINT_AV_SAMPLE_PATH

    # Initialize transfer generation model pipeline
    pipeline = DiffusionControl2WorldMultiviewGenerationPipeline(
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
        sigma_max=80.0,
        num_video_frames=57,
        process_group=process_group,
        height=576,
        width=1024,
        is_lvg_model=cfg.is_lvg_model,
        n_clip_max=cfg.n_clip_max,
        waymo_example=cfg.waymo_example,
    )

    os.makedirs(cfg.video_save_folder, exist_ok=True)

    current_prompt = prompts
    current_video_path = ""
    video_save_subfolder = os.path.join(cfg.video_save_folder, "video_0")
    os.makedirs(video_save_subfolder, exist_ok=True)
    current_control_inputs = copy.deepcopy(control_inputs)

    # if control inputs are not provided, run respective preprocessor (for seg and depth)
    preprocessors(current_video_path, current_prompt, current_control_inputs, video_save_subfolder)

    # Generate video
    generated_output = pipeline.generate(
        prompts=current_prompt,
        view_condition_video=cfg.view_condition_video,
        initial_condition_video=cfg.initial_condition_video,
        control_inputs=current_control_inputs,
        save_folder=video_save_subfolder,
    )
    if generated_output is None:
        log.critical("Guardrail blocked generation.")
    video, prompt = generated_output

    video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
    prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")

    if device_rank == 0:
        # Save video
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(
            video=video,
            fps=cfg.fps,
            H=video.shape[1],
            W=video.shape[2],
            video_save_quality=7,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(";".join(prompt).encode("utf-8"))

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
