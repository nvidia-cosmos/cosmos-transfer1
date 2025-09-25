# Robot Data Augmentation with Cosmos-Transfer1

This pipeline provides a two-step process to augment robotic videos using **Cosmos-Transfer1-7B**. It leverages **spatial-temporal control** to modify backgrounds while preserving the shape and/or appearance of the robot foreground.

## Overview of Settings

We propose two augmentation settings:

### Setting 1 (fg_vis_edge_bg_seg): Preserve Shape and Appearance of the Robot (foreground)
- **Foreground Controls**: `Edge`, `Vis`
- **Background Controls**: `Segmentation`
- **Weights**:
  - `w_edge(FG) = 1`
  - `w_vis(FG) = 1`
  - `w_seg(BG) = 1`
  - All other weights = 0

### Setting 2 (fg_edge_bg_seg): Preserve Only Shape of the Robot (foreground)
- **Foreground Controls**: `Edge`
- **Background Controls**: `Segmentation`
- **Weights**:
  - `w_edge(FG) = 1`
  - `w_seg(BG) = 1`
  - All other weights = 0

## Step-by-Step Instructions

### Step 1: Generate Spatial-Temporal Weights

This script extracts foreground (robot) and background information from semantic segmentation data. It processes per-frame segmentation masks and color-to-class mappings to generate spatial-temporal weight matrices for each control modality based on the selected setting.

#### Input Requirements:
- A `segmentation` folder containing per-frame segmentation masks in PNG format
- A `segmentation_label` folder containing color-to-class mapping JSON files for each frame, for example:
  ```json
  {
      "(29, 0, 0, 255)": {
          "class": "gripper0_right_r_palm_vis"
      },
      "(31, 0, 0, 255)": {
          "class": "gripper0_right_R_thumb_proximal_base_link_vis"
      },
      "(33, 0, 0, 255)": {
          "class": "gripper0_right_R_thumb_proximal_link_vis"
      }
  }
  ```
- An input video file

Here is an example input format:
[Example input directory](https://github.com/nvidia-cosmos/cosmos-transfer1/tree/main/assets/robot_augmentation_example/example1)

#### Usage

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/auxiliary/robot_augmentation/spatial_temporal_weight.py \
    --setting setting1 \
    --robot-keywords world_robot gripper robot \
    --input-dir assets/robot_augmentation_example \
    --output-dir outputs/robot_augmentation_example
```

#### Parameters:

* `--setting`: Weight setting to use (choices: 'setting1', 'setting2', default: 'setting1')
  * setting1: Emphasizes robot in visual and edge features (vis: 1.0 foreground, edge: 1.0 foreground, seg: 1.0 background)
  * setting2: Emphasizes robot only in edge features (edge: 1.0 foreground, seg: 1.0 background)

* `--input-dir`: Input directory containing example folders
  * Default: 'assets/robot_augmentation_example'

* `--output-dir`: Output directory for weight matrices
  * Default: 'outputs/robot_augmentation_example'

* `--robot-keywords`: Keywords used to identify robot classes
  * Default: ["world_robot", "gripper", "robot"]
  * Any semantic class containing these keywords will be treated as robot foreground

### Step 2: Run Cosmos-Transfer1 Inference

Use the generated spatial-temporal weight matrices to perform video augmentation with the proper controls.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"

PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 \
cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/robot_example_spatial_temporal_setting1 \
    --controlnet_specs assets/robot_augmentation_example/example1/inference_cosmos_transfer1_robot_spatiotemporal_weights.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```

- Augmented videos are saved in `outputs/robot_example_spatial_temporal_setting1/`

## Input Outputs Example

Input video:

<video src="https://github.com/user-attachments/assets/9c2df99d-7d0c-4dcf-af87-4ec9f65328ed">
  Your browser does not support the video tag.
</video>

You can run multiple times with different prompts (e.g., `assets/robot_augmentation_example/example1/example1_prompts.json`), and you can get different augmentation results:

<video src="https://github.com/user-attachments/assets/6dee15f5-9d8b-469a-a92a-3419cb466d44">
  Your browser does not support the video tag.
</video>
