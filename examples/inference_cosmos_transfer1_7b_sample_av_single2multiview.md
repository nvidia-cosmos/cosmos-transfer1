# Transfer1 Sample-AV Single2Multiview Inference Example

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
│   ├── Cosmos-Transfer1-7B-Sample-AV-Single2Multiview/
│   │   ├── t2w_base_model.pt
│   │   ├── t2w_hdmap_control.pt
│   │   ├── t2w_lidar_control.pt
│   │   ├── v2w_base_model.pt
│   │   ├── v2w_hdmap_control.pt
│   │   └── v2w_lidar_control.pt
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

This is an example of running Cosmos-Transfer1 Single2Multiview using autonomous vehicle (AV) data. Here we provide two controlnets, `hdmap` and `lidar`, that allow transfering from those domains to the real world.

Ensure you are at the root of the repository before executing the following to launch `transfer_multiview.py` and configures the controlnets for inference according to `assets/sample_av_hdmap_multiview_spec.json`:

:

```bash
#!/bin/bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun’s rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
 --checkpoint_dir checkpoints \
 --video_save_name output_video\
 --video_save_folder outputs/sample_av_multiview \
 --offload_text_encoder_model \
 --guidance 3 \
 --controlnet_specs assets/sample_av_hdmap_multiview_spec.json --num_gpus ${NUM_GPUS} --num_steps 35 \
 --input_video_path assets/sample_av_mv_input_rgb.mp4 \
 --prompt "$PROMPT"
```

