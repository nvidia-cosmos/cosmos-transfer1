# Cosmos-Transfer1: Inference featuring keypoint control

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
│   │   ├── keypoint_control.ptg
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

## Run Example

For a general overview of how to use the model see [this guide](inference_cosmos_transfer1_7b.md).

Ensure you are at the root of the repository before executing the following:

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/inference_keypoint \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_keypoint.json \
    --offload_text_encoder_model
```

You can also choose to run the inference on multiple GPUs as follows:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=4}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/inference_keypoint \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_keypoint.json \
    --offload_text_encoder_model \
    --num_gpus $NUM_GPU
```

This launches `transfer.py` and configures the controlnets for inference according to `assets/inference_keypoint_input_video.json`:

```json
{
    "prompt": "The video takes place in a kitchen setting ...",
    "input_video_path": "assets/inference_keypoint_input_video.mp4",
    "keypoint": {
        "control_weight": 1.0
    }
}
```

### The input and output videos

The input video looks like this:

<video src="https://github.com/user-attachments/assets/b28096ca-ce47-4fb8-8d41-7a7e14104cf0">
  Your browser does not support the video tag.
</video>


Here's what the model outputs:

<video src="https://github.com/user-attachments/assets/493eae48-8fbb-4692-9700-16d78ffddad1">
  Your browser does not support the video tag.
</video>

Note that the faces in the generated video have been blurred by the guardrail.
