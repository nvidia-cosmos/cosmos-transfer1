# Sample-AV Transfer

## Install Cosmos-Transfer1

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e):

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

Note that this will require about 300GB of free storage. Not all these checkpoints will be used in every generation.

5. The downloaded files should be in the following structure:

```
checkpoints/
├── nvidia
│   │
│   ├── Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── models--nvidia--Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0/...
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
│   └── Cosmos-Tokenize1-CV8x8x8-720p
│       ├── decoder.jit
│       ├── encoder.jit
│       ├── autoencoder.jit
│       └── mean_std.pt
│
├── depth-anything/...
├── facebook/...
├── google-t5/...
├── IDEA-Research/...
└── meta-llama/...
```

## Run Example

For a general overview of how to use the model see [this guide](/examples/inference_cosmos_transfer1_7b.md).

This is an example of post-training Cosmos-Transfer1 using autonomous vehicle (AV) data. Here we provide two controlnets, `hdmap` and `lidar`, that allow transfering from those domains to the real world.

Ensure you are at the root of the repository before executing the following:

```bash
#!/bin/bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive."
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_name output_video \
    --video_save_folder outputs/sample_av_multi_control \
    --prompt "$PROMPT" \
    --sigma_max 80 \
    --offload_text_encoder_model --is_av_sample \
    --controlnet_specs assets/sample_av_multi_control_spec.json
```

You can also choose to run the inference on multiple GPUs as follows:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=4}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_name output_video \
    --video_save_folder outputs/sample_av_multi_control \
    --prompt "$PROMPT" \
    --sigma_max 80 \
    --offload_text_encoder_model --is_av_sample \
    --controlnet_specs assets/sample_av_multi_control_spec.json \
    --num_gpus $NUM_GPU
```

This launches `transfer.py` and configures the controlnets for inference according to `assets/sample_av_multi_control_spec.json`:

```json
{
    "hdmap": {
        "control_weight": 0.3,
        "input_control": "assets/sample_av_multi_control_input_hdmap.mp4"
    },
    "lidar": {
        "control_weight": 0.7,
        "input_control": "assets/sample_av_multi_control_input_lidar.mp4"
    }
}
```

Note that unlike other examples, here we chose to provide the input prompt and some other parameters through the command line arguments, as opposed to through the spec file. This flexibility allows abstracting out the fixed parameters in the spec file and vary the dynamic parameters through the command line.

### Additional Toolkits
We provide the `cosmos-av-sample-toolkits` at https://github.com/nv-tlabs/cosmos-av-sample-toolkits.

This toolkit includes:

- 10 additional raw data samples (e.g., HDMap and LiDAR), along with scripts to preprocess and render them into model-compatible inputs.
- Rendering scripts for converting other datasets, such as the Waymo Open Dataset, into inputs compatible with Cosmos-Transfer1.

### The input and output videos

HDMap input control:

<video src="https://github.com/user-attachments/assets/5518273f-5dd6-42a2-99b1-9af683da6c9d">
  Your browser does not support the video tag.
</video>


LiDAR input control:

<video src="https://github.com/user-attachments/assets/2a9c1bf7-f239-4ac0-adde-5521311785b8">
  Your browser does not support the video tag.
</video>


Output video using HDMap and LiDAR:

<video src="https://github.com/user-attachments/assets/36292685-044f-4d04-98e9-bb3187a615e5">
  Your browser does not support the video tag.
</video>

Feel free to experiment with more specs. For example, the command below only uses HDMap:

```bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive."
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_name output_video \
    --video_save_folder outputs/sample_av_hdmap_spec \
    --prompt "$PROMPT" \
    --offload_text_encoder_model --is_av_sample \
    --controlnet_specs assets/sample_av_hdmap_spec.json
```
