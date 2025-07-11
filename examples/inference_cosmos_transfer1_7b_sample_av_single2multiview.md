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
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/ --model 7b_av
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
│   ├── Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/
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

This is an example of running Cosmos-Transfer1-Sample-AV-Single2Multiview using autonomous vehicle (AV) data. Here we provide multiview `hdmap` as conditioning, transferring virtual worlds demarcated by map elements to the real world.

Ensure you are at the root of the repository before executing the following to launch `transfer_multiview.py` and configures the controlnets for inference according to `assets/sample_av_hdmap_multiview_spec.json`:

```bash
#!/bin/bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video captures a driving scene on a multi-lane highway during the day. The sky is clear and blue, indicating good weather conditions. The road is relatively busy with several cars and trucks in motion. A red sedan is driving in the left lane, followed by a black pickup truck in the right lane. The vehicles are maintaining a safe distance from each other. On the right side of the road, there are speed limit signs indicating a limit of 65 mph. The surrounding area includes a mix of greenery and industrial buildings, with hills visible in the distance. The overall environment appears to be a typical day on a highway with moderate traffic. The golden light of the late afternoon bathes the highway, casting long shadows and creating a warm, serene atmosphere. The sky is a mix of orange and blue, with the sun low on the horizon. The red sedan in the left lane reflects the golden hues, while the black pickup truck in the right lane casts a distinct shadow on the pavement. The speed limit signs stand out clearly under the fading sunlight. The surrounding greenery glows with a rich, warm tone, and the industrial buildings take on a softened appearance in the sunset."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1
PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name output_video_1_30_0 \
--video_save_folder outputs/sample_av_multiview \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/sample_av_hdmap_multiview_spec.json --num_gpus ${NUM_GPUS} --num_steps 30 \
--view_condition_video assets/sample_av_mv_input_rgb.mp4 \
--prompt "$PROMPT"
```

We can further extend the video we've just generated with the Cosmos-Transfer1-Sample-AV-Single2Multiview-Video2World model using this command:

```bash
#!/bin/bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video captures a driving scene on a multi-lane highway during the day. The sky is clear and blue, indicating good weather conditions. The road is relatively busy with several cars and trucks in motion. A red sedan is driving in the left lane, followed by a black pickup truck in the right lane. The vehicles are maintaining a safe distance from each other. On the right side of the road, there are speed limit signs indicating a limit of 65 mph. The surrounding area includes a mix of greenery and industrial buildings, with hills visible in the distance. The overall environment appears to be a typical day on a highway with moderate traffic. The golden light of the late afternoon bathes the highway, casting long shadows and creating a warm, serene atmosphere. The sky is a mix of orange and blue, with the sun low on the horizon. The red sedan in the left lane reflects the golden hues, while the black pickup truck in the right lane casts a distinct shadow on the pavement. The speed limit signs stand out clearly under the fading sunlight. The surrounding greenery glows with a rich, warm tone, and the industrial buildings take on a softened appearance in the sunset."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1
PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name output_video_extension \
--video_save_folder outputs/sample_av_multiview \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/sample_av_hdmap_multiview_lvg_spec.json --num_gpus ${NUM_GPUS} --num_steps 30 \
--view_condition_video assets/sample_av_mv_input_rgb.mp4 \
--prompt "$PROMPT" \
--n_clip_max 3 --num_input_frames 9 --initial_condition_video outputs/sample_av_multiview/output_video.mp4
```
Video extension is achieved by looping the Cosmos-Transfer1-Sample-AV-Single2Multiview model to generate multiple 57-frame clips. Three additional arguments are provided to enable video extension:
1. `--n_clip_max` control the number of clips. it does not control the number of frames generated per clip. The model cannot generate more frames than that is present in `--view_contion_video`.
2. `--num_input_frames` controls the number of overlapping frames between each clip, creating smooth transition between clips. This can be set to either `1` or `9`.
3. `--initial_condition_video` is the video generated in the first example using the `t2w` model.

We also provide `lidar` controled examples that can be tested by modifying the `--controlnet_specs` to `assets/sample_av_lidar_multiview_spec.json` in the above commands.

## Run Post-trained Example
If you follow the post-training example in [Training README](./training_cosmos_transfer_7B_sample_AV.md), you will eventually end up with a waymo-style post-trained ckpt where there are 5 input and output views. The inference scirpt is a little bit different than the pre-trained 6 view model. We provided an example of running Cosmos-Transfer1-Sample-AV-Single2Multiview post-trained with waymo data. Here we provide multiview `hdmap` as conditioning, transferring virtual worlds demarcated by map elements to the real world.

Ensure you are at the root of the repository before executing the following to launch `transfer_multiview.py` and configures the controlnets for inference according to `assets/sample_av_hdmap_multiview_waymo_spec.json`, the ckpt_path need to match your own post-trained ckpt :

```bash
#!/bin/bash
export PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. The video captures a driving scene on a multi-lane highway during the day. The sky is clear and blue, indicating good weather conditions. The road is relatively busy with several cars and trucks in motion. A red sedan is driving in the left lane, followed by a black pickup truck in the right lane. The vehicles are maintaining a safe distance from each other. On the right side of the road, there are speed limit signs indicating a limit of 65 mph. The surrounding area includes a mix of greenery and industrial buildings, with hills visible in the distance. The overall environment appears to be a typical day on a highway with moderate traffic. The golden light of the late afternoon bathes the highway, casting long shadows and creating a warm, serene atmosphere. The sky is a mix of orange and blue, with the sun low on the horizon. The red sedan in the left lane reflects the golden hues, while the black pickup truck in the right lane casts a distinct shadow on the pavement. The speed limit signs stand out clearly under the fading sunlight. The surrounding greenery glows with a rich, warm tone, and the industrial buildings take on a softened appearance in the sunset."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1
PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name output_video \
--video_save_folder outputs/sample_av_multiview_waymo \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/sample_av_hdmap_multiview_spec.json --num_gpus ${NUM_GPUS} --num_steps 30 \
--view_condition_video assets/sample_av_mv_input_rgb.mp4 \
--prompt "$PROMPT"
--waymo_example True
```
