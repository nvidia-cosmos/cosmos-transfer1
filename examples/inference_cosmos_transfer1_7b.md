# Cosmos-Transfer1: World Generation with Adaptive Multimodal Control

## Install Cosmos-Transfer1

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e):

```bash
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

Note that this will require about 300GB of free storage. Not all these checkpoints will be used in every generation.

5. The downloaded files should be in the following structure:

```
checkpoints/
├── nvidia
│   │
│   ├── Cosmos-Guardrail1
│   │   ├── README.md
│   │   ├── blocklist/...
│   │   ├── face_blur_filter/...
│   │   └── video_content_safety_filter/...
│   │
│   ├── Cosmos-Transfer1-7B
│   │   ├── base_model.pt
│   │   ├── vis_control.pt
│   │   ├── edge_control.pt
│   │   ├── edge_control_distilled.pt
│   │   ├── seg_control.pt
│   │   ├── depth_control.pt
│   │   ├── 4kupscaler_control.pt
│   │   └── config.json
│   │
│   ├── Cosmos-Transfer1-7B-Sample-AV/
│   │   ├── base_model.pt
│   │   ├── hdmap_control.pt
│   │   └── lidar_control.pt
│   │
│   │── Cosmos-Tokenize1-CV8x8x8-720p
│   │   ├── decoder.jit
│   │   ├── encoder.jit
│   │   ├── autoencoder.jit
│   │   └── mean_std.pt
│   │
│   └── Cosmos-UpsamplePrompt1-12B-Transfer
│       ├── depth
│       │   ├── consolidated.safetensors
│       │   ├── params.json
│       │   └── tekken.json
│       ├── README.md
│       ├── segmentation
│       │   ├── consolidated.safetensors
│       │   ├── params.json
│       │   └── tekken.json
│       ├── seg_upsampler_example.png
│       └── viscontrol
│           ├── consolidated.safetensors
│           ├── params.json
│           └── tekken.json
│
├── depth-anything/...
├── facebook/...
├── google-t5/...
├── IDEA-Research/...
└── meta-llama/...
```

## Sample Commands

Here's an example command:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir checkpoints \
    --input_video_path path/to/input_video.mp4 \
    --video_save_name output_video \
    --controlnet_specs spec.json \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

Cosmos-Transfer1 supports a variety of configurations. You can pass your configuration in a JSON file via the argument `--controlnet_specs`. Let's go over a few examples:

### Example 1: single control (Edge)

The following `controlnet_specs` only activates the edge controlnet.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

You can also choose to run the inference on multiple GPUs as follows:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=4}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

This launches `transfer.py` and configures the controlnets for inference according to `assets/inference_cosmos_transfer1_single_control_edge.json`. For the distilled edge model, a `control_weight` of `1.0` is recommended:
```json
{
    "prompt": "The video is set in a modern, well-lit office environment with a sleek, minimalist design. ...",
    "input_video_path" : "assets/example1_input_video.mp4",
    "edge": {
        "control_weight": 1.0
    }
}
```

The input video is a low-resolution 640 × 480 video.

<video src="https://github.com/user-attachments/assets/e63b9e9c-fee1-4105-a480-bb525bde1115">
  Your browser does not support the video tag.
</video>

We generate a 960 x 704 video.

<video src="https://github.com/user-attachments/assets/83ab7724-335a-46cf-b96e-34860ab735d1">
  Your browser does not support the video tag.
</video>

You can use our prompt upsampler to convert your short prompt into a longer, more detailed prompt for video generation by using the `--upsample_prompt` argument.


```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0  cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge_upsampled_prompt \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge_short_prompt.json \
    --offload_text_encoder_model \
    --upsample_prompt \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```


Input prompt: *Robotic arms hand over a coffee cup to a woman in a modern office.*

Upsampled prompt: *The video opens with a close-up of a robotic arm holding a coffee cup with a lid, positioned next to a coffee machine. The arm is metallic with a black wrist, and the coffee cup is white with a brown lid. The background shows a modern office environment with a woman in a blue top and black pants standing in the distance. As the video progresses, the robotic arm moves the coffee cup towards the woman, who approaches to receive it. The woman has long hair and is wearing a blue top and black pants. The office has a contemporary design with glass partitions, potted plants, and other office furniture.*

Here is the generated video using the upsampled prompt.

<video src="https://github.com/user-attachments/assets/bf20ac09-7a8e-448c-8961-db2adfe11e69">
  Your browser does not support the video tag.
</video>

### Example 2: distilled single control (Edge)

A lightweight, distilled variant of the edge controlnet can be enabled through the `--use_distilled` flag. This model allows for single-step inference, reducing the compute resources used:
```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_edge_distilled \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU \
    --use_distilled
```

### Example 3: multimodal control

The following `controlnet_specs` activates vis, edge, depth, seg controls at the same time and apply uniform spatial weights.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example2_uniform_weights \
    --controlnet_specs assets/inference_cosmos_transfer1_uniform_weights.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

This launches `transfer.py` and configures the controlnets for inference according to `assets/inference_cosmos_transfer1_uniform_weights.json`:

```json
{
    "prompt": "The video is set in a modern, well-lit office environment with a sleek, minimalist design. ...",
    "input_video_path" : "assets/example1_input_video.mp4",
    "vis": {
        "control_weight": 0.25
    },
    "edge": {
        "control_weight": 0.25
    },
    "depth": {
        "input_control": "assets/example1_depth.mp4",
        "control_weight": 0.25
    },
    "seg": {
        "input_control": "assets/example1_seg.mp4",
        "control_weight": 0.25
    }
}
```

The output video can be found at `assets/example1_uniform_weights.mp4`.
<video src="https://github.com/user-attachments/assets/d1c59d92-fa8c-4248-8960-13bd20444bb5">
  Your browser does not support the video tag.
</video>

#### Explanation of the controlnet spec
* `prompt` specifies the global prompt that all underlying networks will receive.
* `input_video_path` specifies the input video
* `sigma_max` specifies the level of noise that should be added to the input video before feeding through the base model branch
* The dictionaries `vis`, `edge`, `depth`, and `seg` activate the corresponding controlnet branches.
* The `control_weight` parameter is a number within the range [0, 1] that controls how strongly the controlnet branch should affect the output of the model. The larger the value (closer to 1.0), the more strongly the generated video will adhere to the controlnet input. However, this rididity may come at a cost of quality. Lower (closer to 0) values would give more creative liberty to the model at the cost of reduced adherance. Usually a middleground value, say 0.5, yields optinal results.
* The inputs to each controlnet branch is automatically computed according to the branch:
  * `vis` applies bilateral blurring on the input video to compute the `input_control` to that branch
  * `edge` uses [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) to compute the Canny edge `input_control` from the `input_control`
  * `depth` uses [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)
  * `seg` uses [Segment Anything Model 2](https://ai.meta.com/sam2/) for generating the segmentation map as `input_control` from the input video.

#### Additional Information
- At each spatiotemporal site, if the sum of the control maps across different modalities is greater than one, we apply normalization to the modality weights so that they sum up to one.
- For `depth` and `seg`, if the `input_control` is not provided, we will run DepthAnything2 and GroundingDino+SAM2 on `input_video_path` to generate the corresponding `input_control`. Please see `assets/inference_cosmos_transfer1_uniform_weights_auto.json` as an example.
- For `seg`, `input_control_prompt` can be provided to customize the prompt sent to GroundingDino. We can use ` . ` to separate objects in the `input_control_prompt`, e.g. `robotic arms . woman . cup`, as suggested by [GroundingDino](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#arrow_forward-demo). If `input_control_prompt` is not provided, `prompt` will be used by default. Please see `assets/inference_cosmos_transfer1_uniform_weights_auto.json` as an example.

### Example 4: multimodal control with spatiotemporal control map

The following `controlnet_specs` activates vis, edge, depth, seg controls at the same time and apply spatiotemporal weights.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example3_spatiotemporal_weights \
    --controlnet_specs assets/inference_cosmos_transfer1_spatiotemporal_weights_auto.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

This launches `transfer.py` and configures the controlnets for inference according to `assets/inference_cosmos_transfer1_spatiotemporal_weights_auto.json`:

```json
{
    "prompt": "The video is set in a modern, well-lit office environment with a sleek, minimalist design...",
    "input_video_path" : "assets/example1_input_video.mp4",
    "vis": {
        "control_weight": 0.5,
        "control_weight_prompt": "robotic arms . gloves"
    },
    "edge": {
        "control_weight": 0.5,
        "control_weight_prompt": "robotic arms . gloves"
    },
    "depth": {
        "control_weight": 0.5
    },
    "seg": {
        "control_weight": 0.5
    }
}
```

The output video can be found at `assets/example1_spatiotemporal_weights.mp4` and is shown below.

<video src="https://github.com/user-attachments/assets/2cc57c4c-b59f-4b25-8911-66d4bd0646aa">
  Your browser does not support the video tag.
</video>

The spatiotemporal mask extracted by the prompt `robotic arms . gloves` is show below.

<video src="https://github.com/user-attachments/assets/98a2e4a1-cd92-469d-8e57-ba376aca4e7e">
  Your browser does not support the video tag.
</video>

#### Explanation of the controlnet spec

The controlnet spec is similar to Example 3 above, with the following exceptions:
* Additional `control_weight_prompt` for the vis and edge modalities. This will trigger the GroundingDINO+SAM2 pipeline to run video segmentation of the input video using `control_weight_prompt` (e.g. `robotic arms . gloves`) for `vis` and `edge` and extract a binarized spatiotemporal mask in which the positive pixels will have a `control_weight` of 0.5 (and negative pixels will have 0.0).
* Change the prompt section of the woman's clothing into a cream-colored and brown shirt. Since this area of the video will be conditioned only by `depth` and `seg`, there will be no conflict to the color information from `vis` modality.

In effect, for the configuration given in `assets/inference_cosmos_transfer1_spatiotemporal_weights_auto.json`, `seg` and `depth` modalities will be applied everywhere uniformly, and `vis` and `edge` will be applied exclusively in the spatiotemporal mask given by the union of `robotic arms` and `gloves` mask detections. In those areas, the weight of each modality will be normalized to one, therefore `vis`, `edge`, `seg` and `depth` will be applied evenly there.


#### Example 5: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file). This enables running multiple generations with different prompts (and per-video control input customization) based on the same controlnet configurations.
Each line in the JSONL file must contain a `visual_input` field equivalent to the `--input_video_path` argument in the case of single control generation. It can also contain the `prompt` field. The batch system supports automatic control input generation, manual override of specific controls per video, and mixed usage of automatic and manual controls in the same batch. By default, the `input_control` specified within the controlnet spec json will be used for all samples in the batch, and are overridden if explicitly specified in the batch input json file (either with another `input_control` path or with `null` to indicate automatic generation based on the visual input).
Here is an example of the Batch Input JSONL Format
```json
{"visual_input": "path/to/video0.mp4", "prompt": "A detailed description..."}
{"visual_input": "path/to/video1.mp4", "prompt": "A detailed description...",   "control_overrides": {"seg": {"input_control": "path/to/video1_seg.mp4"}, "depth": {"input_control": null}}}
{"visual_input": "path/to/video2.mp4", "prompt": "A detailed description...",   "control_overrides": {"seg": {"input_control": "path/to/video2_seg.mp4"}, "depth": {"input_control": "path/to/video2_depth.mp4"}}}
```
Inference command (with 9 input frames):
```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example2_uniform_weights \
    --controlnet_specs assets/inference_cosmos_transfer1_uniform_weights.json \
    --offload_text_encoder_model \
    --batch_input_path path/to/batch_input_path.json \
    --num_gpus $NUM_GPU
```


## Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--controlnet_specs` | A JSON describing the Multi-ControlNet config | JSON |
| `--checkpoint_dir` | Directory containing model weights | "checkpoints" |
| `--tokenizer_dir` | Directory containing tokenizer weights | "Cosmos-Tokenize1-CV8x8x8-720p" |
| `--input_video_path` | The path to the input video | None |
| `--video_save_name` | Output video filename for single video generation | "output" |
| `--video_save_folder` | Output directory for batch video generation | "outputs/" |
| `--prompt` | Text prompt for video generation. | "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution." |
| `--negative_prompt` | Negative prompt for improved quality | "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all." |
| `--num_steps` | Number of diffusion sampling steps | 35 |
| `--guidance` | CFG guidance scale | 7.0 |
| `--sigma_max` | The level of partial noise added to the input video in the range [0, 80.0]. Any value equal or higher than 80.0 will result in not using the input video and providing the model with pure noise. | 70.0 |
| `--blur_strength` | The strength of blurring when preparing the control input for the vis controlnet. Valid values are 'very_low', 'low', 'medium', 'high', and 'very_high'. | 'medium' |
| `--canny_threshold` | The threshold for canny edge detection when preparing the control input for the edge controlnet. Lower threshold means more edges detected. Valid values are 'very_low', 'low', 'medium', 'high', and 'very_high'. | 'medium' |
| `--fps` | Output frames-per-second | 24 |
| `--seed` | Random seed | 1 |
| `--offload_text_encoder_model` | Offload text encoder after inference, used for low-memory GPUs | False |
| `--offload_guardrail_models` | Offload guardrail models after inference, used for low-memory GPUs | False |
| `--upsample_prompt` | Upsample prompt using prompt upsampler model | False |
| `--offload_prompt_upsampler` |  Offload prompt upsampler models after inference, used for low-memory GPUs | False |

Note: in order to run Cosmos on low-memory GPUs, you can use model offloading. This is accomplished by offloading the model from GPU memory after it has served its purpose to open space for the next model execution.

Note: we support various aspect ratios, including 1:1 (960x960 for height and width), 4:3 (960x704), 3:4 (704x960), 16:9 (1280x704), and 9:16 (704x1280). If the input is not one of these five resolutions, it is first resized to one of them according to the nearest aspect ratio.



### Examples 5: Robotics Augmentation Workflow: Mapping one robotics synthetic example to multiple realistic examples

[Robotics Augmentation Workflow](/cosmos_transfer1/auxiliary/robot_augmentation/README.md)

<video src="https://github.com/user-attachments/assets/6dee15f5-9d8b-469a-a92a-3419cb466d44">
  Your browser does not support the video tag.
</video>


## Safety Features

The model uses a built-in safety guardrail system that cannot be disabled. Generating human faces is not allowed and will be blurred by the guardrail.

## Prompting Instructions

The input prompt is the most important parameter under the user's control when interacting with the model. Providing rich and descriptive prompts can positively impact the output quality of the model, whereas short and poorly detailed prompts can lead to subpar video generation. Here are some recommendations to keep in mind when crafting text prompts for the model:

1. **Describe a single, captivating scene**: Focus on a single scene to prevent the model from generating videos with unnecessary shot changes.
2. **Limit camera control instructions**: The model doesn't handle prompts involving camera control well, as this feature is still under development.
