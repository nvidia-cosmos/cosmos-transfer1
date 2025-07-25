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
import torch
import gc
import json
from cosmos_transfer1.utils import log
from cosmos_transfer1.diffusion.inference import transfer_pipeline

from cosmos_transfer1.diffusion.inference import dummy_pipeline
from server.model_server import ModelServer
from server.deploy_config import Config
import gradio as gr
from server import gradio_file_server


model = None


def create_dummy_pipeline():
    log.info("Creating dummy pipeline for testing")
    return dummy_pipeline.TransferPipeline(num_gpus=1, output_dir=os.path.join(Config.output_dir, "dummy"))


def create_pipeline():
    log.info(f"Initializing model using factory function {Config.factory_module}.{Config.factory_function}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    pipeline = transfer_pipeline.TransferPipeline(
        num_gpus=world_size,
        checkpoint_dir=Config.checkpoint_dir,
        output_dir=Config.output_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return pipeline


def get_spec(spec_file):
    with open(spec_file, "r") as f:
        controlnet_specs = json.load(f)
    return controlnet_specs


# Event handler
def infer_wrapper(
    request_text,
):
    try:
        # Parse the request as JSON
        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError as e:
            return None, f"Error parsing request JSON: {e}\nPlease ensure your request is valid JSON."

        # Extract parameters from request
        input_video = request_data.get("input_video")
        prompt = request_data.get("prompt", "")
        negative_prompt = request_data.get("negative_prompt", "")
        guidance_scale = request_data.get("guidance_scale", 7.0)
        num_steps = request_data.get("num_steps", 35)
        seed = request_data.get("seed", 1)
        sigma_max = request_data.get("sigma_max", 70.0)
        blur_strength = request_data.get("blur_strength", "medium")
        canny_threshold = request_data.get("canny_threshold", "medium")
        controlnet_specs = {}
        for key in transfer_pipeline.valid_hint_keys:
            controlnet_specs[key] = request_data.get(key, {})

        if not input_video:
            return None, "Error: 'input_video' is required in the request"

        log.info(f"Using provided controlnet_specs from request: {controlnet_specs}")

        args_dict = transfer_pipeline.TransferPipeline.validate_params(
            controlnet_specs=controlnet_specs,
            input_video=input_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            sigma_max=sigma_max,
            blur_strength=blur_strength,
            canny_threshold=canny_threshold,
        )
    except ValueError as e:
        return None, f"Error validating parameters: {e}"

    model.infer(args_dict)

    # Check if output was generated
    output_path = os.path.join(Config.output_dir, "output.mp4")
    if os.path.exists(output_path):
        # Read the generated prompt
        prompt_path = os.path.join(Config.output_dir, "output.txt")
        final_prompt = prompt
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                final_prompt = f.read().strip()

        return (
            output_path,
            f"Video generated successfully!\nOutput saved to: {Config.output_dir}\nFinal prompt: {final_prompt}",
        )
    else:
        return None, f"Generation failed - no output video was created\nCheck folder: {Config.output_dir}"


def create_gradio_interface():

    with gr.Blocks(title="Cosmos-Transfer1 Video Generation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Cosmos-Transfer1: World Generation with Adaptive Multimodal Control")
        gr.Markdown("Upload a video and configure controls to generate a new video with the Cosmos-Transfer1 model.")

        with gr.Row():
            gradio_file_server.file_server_components(Config.uploads_dir, open=False)

        gr.Markdown("---")
        gr.Markdown(f"**Output Directory**: {Config.output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                # Single request input field
                request_input = gr.Textbox(
                    label="Request (JSON)",
                    value=json.dumps(
                        {
                            "input_video": "/path/to/your/video.mp4",
                            "prompt": "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
                            "negative_prompt": "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
                            "guidance_scale": 7.0,
                            "num_steps": 35,
                            "seed": 1,
                            "sigma_max": 70.0,
                            "blur_strength": "medium",
                            "canny_threshold": "medium",
                            **{key: {"control_weight": 0.0} for key in sorted(transfer_pipeline.valid_hint_keys)},
                        },
                        indent=2,
                    ),
                    lines=20,
                    interactive=True,
                )

                # Help section
                with gr.Accordion("Request Format Help", open=False):
                    gr.Markdown(
                        """
                    ### Required Fields:
                    - `input_video` (string): Path to the input video file
                    
                    ### Optional Fields:
                    - `prompt` (string): Text prompt describing the desired output
                    - `negative_prompt` (string): What to avoid in the output
                    - `guidance_scale` (float): Guidance scale (1-15, default: 7.0)
                    - `num_steps` (int): Number of inference steps (10-50, default: 35)
                    - `seed` (int): Random seed (default: 1)
                    - `sigma_max` (float): Maximum noise level (0-80, default: 70.0)
                    - `blur_strength` (string): One of ["very_low", "low", "medium", "high", "very_high"] (default: "medium")
                    - `canny_threshold` (string): One of ["very_low", "low", "medium", "high", "very_high"] (default: "medium")
                    - `vis` (object): Vis controlnet (default: {"control_weight": 0.0})
                    - `seg` (object): Segmentation controlnet (default: {"control_weight": 0.0})
                    - `edge` (object): Edge controlnet (default: {"control_weight": 0.0})
                    - `depth` (object): Depth controlnet (default: {"control_weight": 0.0})
                    - `keypoint` (object): Keypoint controlnet (default: {"control_weight": 0.0})
                    - `upscale` (object): Upscale controlnet (default: {"control_weight": 0.0})
                    - `hdmap` (object): HDMap controlnet (default: {"control_weight": 0.0})
                    - `lidar` (object): Lidar controlnet (default: {"control_weight": 0.0})
                    
                    ### Example:
                    ```json
                    {
                        "input_video": "/mnt/pvc/gradio_outdir/upload_20240115_120000/my_video.mp4",
                        "prompt": "A beautiful landscape video",
                        "guidance_scale": 8.5,
                        "num_steps": 40
                    }
                    ```
                    """
                    )
                with gr.Accordion("Tips", open=False):
                    gr.Markdown(
                        """
                    - **Use the file browser above** to upload your video and copy its path for the `input_video` field
                    - **Describe a single, captivating scene**: Focus on one scene to prevent unnecessary shot changes
                    - **Use detailed prompts**: Rich descriptions lead to better quality outputs  
                    - **Experiment with control weights**: Different combinations can yield different artistic effects
                    - **Adjust sigma_max**: Lower values preserve more of the input video structure
                    """
                    )

            with gr.Column(scale=1):
                # Output
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        generate_btn.click(
            fn=infer_wrapper,
            inputs=[request_input],
            outputs=[output_video, status_text],
        )

    return interface


if __name__ == "__main__":

    # Check if checkpoints exist
    if not os.path.exists(Config.checkpoint_dir):
        print(f"Error: checkpoints directory {Config.checkpoint_dir} not found.")
        exit(1)

    if Config.num_gpus == 0:
        model = create_dummy_pipeline()
    elif Config.num_gpus == 1:
        model = create_pipeline()
    else:
        model = ModelServer(num_workers=Config.num_gpus)

    interface = create_gradio_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        # Configure file upload limits
        max_file_size="500MB",  # Adjust as needed
        allowed_paths=[Config.output_dir, Config.uploads_dir],
    )
