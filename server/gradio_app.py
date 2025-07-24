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
from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferPipeline

# from cosmos_transfer1.diffusion.inference.dummy_pipeline import TransferPipeline as DummyTransferPipeline
from server.model_server import ModelServer
from server.deploy_config import Config
import gradio as gr


model = None


def create_dummy_pipeline():
    log.info("Creating dummy pipeline for testing")
    # return DummyTransferPipeline(
    #     num_gpus=1,
    #     output_dir=Config.output_dir,
    # )


def create_pipeline():
    log.info(f"Initializing model using factory function {Config.factory_module}.{Config.factory_function}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    pipeline = TransferPipeline(
        num_gpus=world_size,
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
    input_video,
    prompt,
    negative_prompt,
    guidance_scale,
    num_steps,
    seed,
    sigma_max,
    blur_strength,
    canny_threshold,
    json_data,
):
    try:
        # Use uploaded JSON data if available, otherwise fall back to default
        if json_data and isinstance(json_data, dict) and json_data:
            controlnet_specs = json_data
            log.info("Using uploaded JSON configuration for inference")
        else:
            # Fallback to default JSON file
            controlnet_specs = get_spec("assets/inference_cosmos_transfer1_single_control_edge.json")
            log.info("Using default JSON configuration for inference")

        args_dict = TransferPipeline.validate_params(
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
        gr.Markdown(f"**Output Directory**: {Config.output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_video = gr.Video(
                    label="Input Video",
                    height=300,
                    # Configure file upload settings
                    interactive=True,
                )

                json_file = gr.File(label="Select JSON File", file_types=[".json"], type="filepath")

                json_content = gr.Textbox(label="JSON Content", lines=10, interactive=False, visible=True)

                json_status = gr.Textbox(
                    label="JSON Status", value="No JSON file loaded - using default config", interactive=False, lines=1
                )

                # State to store the parsed JSON data
                json_data_state = gr.State(value={})

                def load_json_file(file_path):
                    if file_path is None:
                        return "", {}, "No JSON file loaded - using default config", False
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Parse and format JSON for better readability
                        import json

                        json_data = json.loads(content)
                        formatted_content = json.dumps(json_data, indent=2, ensure_ascii=False)

                        # Extract filename for status
                        import os

                        filename = os.path.basename(file_path)
                        status_msg = f"✅ JSON loaded successfully: {filename} - Config will be used for inference"

                        return formatted_content, json_data, status_msg
                    except json.JSONDecodeError as e:
                        error_msg = f"❌ Invalid JSON format - {str(e)}"
                        return error_msg, {}, error_msg
                    except Exception as e:
                        error_msg = f"❌ Error loading file - {str(e)}"
                        return error_msg, {}, error_msg

                json_file.change(
                    fn=load_json_file,
                    inputs=[json_file],
                    outputs=[json_content, json_data_state, json_status],
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    value="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
                    lines=4,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
                    lines=3,
                )

                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance Scale")
                    num_steps = gr.Slider(10, 50, value=35, step=1, label="Number of Steps")
                    seed = gr.Number(value=1, label="Seed", precision=0)
                    sigma_max = gr.Slider(0, 80, value=70.0, step=1, label="Sigma Max")

                    blur_strength = gr.Dropdown(
                        choices=["very_low", "low", "medium", "high", "very_high"],
                        value="medium",
                        label="Blur Strength",
                    )

                    canny_threshold = gr.Dropdown(
                        choices=["very_low", "low", "medium", "high", "very_high"],
                        value="medium",
                        label="Canny Threshold",
                    )

            with gr.Column(scale=1):
                # Output
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        generate_btn.click(
            fn=infer_wrapper,
            inputs=[
                input_video,
                prompt,
                negative_prompt,
                guidance_scale,
                num_steps,
                seed,
                sigma_max,
                blur_strength,
                canny_threshold,
                json_data_state,
            ],
            outputs=[output_video, status_text],
        )

        # Examples section
        gr.Markdown("## Tips for better results:")
        gr.Markdown(
            """
        - **Describe a single, captivating scene**: Focus on one scene to prevent unnecessary shot changes
        - **Use detailed prompts**: Rich descriptions lead to better quality outputs  
        - **Experiment with control weights**: Different combinations can yield different artistic effects
        - **Adjust sigma_max**: Lower values preserve more of the input video structure
        """
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
        allowed_paths=["/mnt/pvc/gradio_outdir"],  # Allow access to output directory
    )
