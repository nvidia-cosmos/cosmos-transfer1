# Training Cosmos-Transfer1-Sample-AV Models
In this document, we provide examples and steps to:
- Build your own Cosmos-Transfer1-Sample-AV models, training from scratch; or
- Post-train Cosmos-Transfer1-Sample-AV models from our checkpoint using your data.

The model is trained separately for each control input type.


## Model Support Matrix
We support the following Cosmos-Transfer1-Sample-AV models for pre-training and post-training. Review the available models and their compute requirements for training to determine the best model for your use case. We use Tensor Parallel of size 8 for training.

| Model Name                                                        | Model Status | Compute Requirements for Post-Training |
|-------------------------------------------------------------------|--------------|----------------------------------------|
| Cosmos-Transfer1-7B-Sample-AV [Lidar]                             | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B-Sample-AV [HDMap]                             | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_model [Lidar] | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_model [HDMap] | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/v2w_model [Lidar] | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/v2w_model [HDMap] | **Supported**| 8 NVIDIA GPUs*                         |

**\*** 80GB GPU memory required for training. `H100-80GB` or `A100-80GB` GPUs are recommended.

## Environment setup

Please refer to the training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

## Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e). Note that this will require about 300GB of free storage.

```bash
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

5. The downloaded files should be in the following structure.

```
checkpoints/
├── nvidia
│   ├── Cosmos-Transfer1-7B
│   │   ├── base_model.pt
│   │   ├── vis_control.pt
│   │   ├── edge_control.pt
│   │   ├── seg_control.pt
│   │   ├── depth_control.pt
│   │   ├── keypoint_control.pt
│   │   ├── 4kupscaler_control.pt
│   │   ├── config.json
│   │   └── guardrail
│   │       ├── aegis/
│   │       ├── blocklist/
│   │       ├── face_blur_filter/
│   │       └── video_content_safety_filter/
│   │
│   ├── Cosmos-Transfer1-7B-Sample-AV/
│   │   ├── base_model.pt
│   │   ├── hdmap_control.pt
│   │   └── lidar_control.pt
│   │
│   ├── Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/
│   │   ├── v2w_base_model.pt
│   │   ├── v2w_hdmap_control.pt
│   │   ├── v2w_lidar_control.pt
│   │   ├── t2w_base_model.pt
│   │   ├── t2w_hdmap_control.pt
│   │   └── t2w_lidar_control.pt
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
└── IDEA-Research/
```

Checkpoint Requirements:
- Base model (`base_model.pt` for single view, `t2w_base_model.pt`, `v2w_base_model.pt` for multiview) and tokenizer models (under `Cosmos-Tokenize1-CV8x8x8-720p`): Required for all training.
- Control modality-specific model checkpoint (e.g., `hdmap_control.pt`): Only needed for post-training that specific control. Not needed if training from scratch.
- Other folders such as `depth-anything`, `facebook/sam2-hiera-large` etc.: optional. These are helper modules to process the video data into the respective control modalities such as depth and segmentation.

## Examples
There are 3 steps to train a Cosmos-Transfer1-Sample-AV model: preparing a dataset, prepare checkpoints, and launch training.

### 1. Dataset Download and Preprocessing
In the example below, we use a subset of [Waymo Open Dataset](https://waymo.com/open/) dataset to demonstrate the steps for preparing the data and launching training.
Please follow the [instructions](https://github.com/nv-tlabs/cosmos-av-sample-toolkits/blob/main/docs/processing_waymo_for_transfer1.md) in [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) to download and convert the Waymo Open Dataset.


### 2. Splitting the Checkpoints to TensorParallel Checkpoints
Due to the large model size, we leverage TensorParallel (TP) to split the model weights across multiple GPUs. We use 8 for the TP size.

```bash
# Will split the Base model checkpoint into 8 TP checkpoints
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV/t2w_base_model.pt
# Example: for LidarControl checkpoint splitting for post-train.
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV/t2w_lidar_control.pt

# Example: for Single2MultiView, the base model checkpoint is different
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_base_model.pt
# Example: for Single2MultiView HDMapControl
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_hdmap_control.pt
```
This will generate the TP checkpoints under `checkpoints/checkpoints_tp/*_mp_*.pt`, which we load in the training below.

### 3. (Optional): Dry-run a Training Job
As a sanity check, run the following command to dry-run an example training job with the above data. The command will generated a full configuration of the experiment.

```bash
export OUTPUT_ROOT=checkpoints # default value

# Training from scratch
torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain

# Post-train from our provided checkpoint (need to first split checkpoint into TP checkpoints as instructed above)
torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_posttrain
```

Explanation of the command:

- The trainer and the passed (master) config script will, in the background, load the detailed experiment configurations defined in `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_sample_av.py`, and register the experiments configurations for all `hint_keys` (control modalities), covering both pretrain and post-train. We use [Hydra](https://hydra.cc/docs/intro/) for advanced configuration composition and overriding.

- The `CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain` corresponds to an experiment name registered in `ctrl_7b_tp_sample_av.py`. By specifiying this name, all the detailed config will be generated and then written to `checkpoints/cosmos_transfer1_pretrain/CTRL_7Bv1_sampleAV/CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain/config.yaml`.

- To customize your training, see `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_sample_av.py` to understand how the detailed configs of the model, trainer, dataloader etc. are defined, and edit as needed.

### 4. Launch Training

#### 4.a Launch Training of Cosmos-Transfer1-7B-Sample-AV
Now we can start a real training job! Removing the `--dryrun` and set `--nproc_per_node=8` will start a real training job on 8 GPUs, using Lidar conditioning:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain
```
#### 4.b Launch Training of Cosmos-Transfer1-7B-Sample-AV-Single2MultiView
In this example, we instead launch a training run of the Single2MultiView model with HDMap condition:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_sv2mv_57frames_control_input_hdmap_block3_pretrain
```

**Config group and override.** An `experiment` determines a complete group of configuration parameters (model architecture, data, trainer behavior, checkpointing, etc.). Changing the `experiment` value in the command above will decide which ControlNet model is trained, and whether it's pretrain or post-train. For example, replacing the experiment name in the command with `CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_posttrain` will post-train the LidarControl model from the downloaded checkpoint instead.

To customize your training, see the job (experiment) config in `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_sample_av.py` to understand how they are defined, and edit as needed.

It is also possible to modify config parameters from the command line. For example:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain trainer.max_iter=100 checkpoint.save_iter=40
```

This will update the maximum training iterations to 100 (default in the registered experiments: 999999999) and checkpoint saving frequency to 40 (default: 1000).

**Saving Checkpoints and Resuming Training.**
During training, the checkpoints will be saved in the structure below. Since we use TensorParallel across 8 GPUs, 8 checkpoints will be saved each time.

```
checkpoints/cosmos_transfer1_pretrain/CTRL_7Bv1_sampleAV/CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_pretrain/checkpoints/
├── iter_{NUMBER}.pt             # "master" checkpoint, saving metadata only
├── iter_{NUMBER}_model_mp_0.pt  # real TP checkpoints
├── iter_{NUMBER}_model_mp_1.pt
├── ...
├── iter_{NUMBER}_model_mp_7.pt
```

Since the `experiment` is uniquely associated with its checkpoint directory, rerunning the same training command after an unexpected interruption will automatically resume from the latest saved checkpoint.

### 5. Inference Using Trained Models

**Converting the TP checkpoints to FSDP checkpoint:** To convert Tensor Parallel (TP) checkpoints to Fully Sharded Data Parallel (FSDP) format, use the conversion script `convert_ckpt_tp_to_fsdp.py`. This script requires the same number of GPUs as your TP size (e.g., if you trained with TP_SIZE=8, you need 8 GPUs for conversion).

Example usage for Sample-AV models:
```bash
# For single-view models
torchrun --nproc_per_node=8 convert_ckpt_tp_to_fsdp.py \
    --experiment CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_posttrain \
    --checkpoint-path checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_sampleAV/CTRL_7Bv1pt3_t2w_121frames_control_input_lidar_block3_posttrain/checkpoints/iter_000000100.pt

# For SingleToMultiView models
torchrun --nproc_per_node=8 convert_ckpt_tp_to_fsdp.py \
    --experiment CTRL_7Bv1pt3_t2w_sv2mv_57frames_control_input_hdmap_block3_posttrain \
    --checkpoint-path checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_sampleAV/CTRL_7Bv1pt3_t2w_sv2mv_57frames_control_input_hdmap_block3_posttrain/checkpoints/iter_000000100.pt
```

Optional arguments:
- `--output-directory`: Custom directory for saving FSDP checkpoints (default: automatically generated from checkpoint path)
- `--include-base-model`: Include base model in ControlNet checkpoint (default: False)

The script will create two files in the output directory:
1. `*_reg_model.pt`: Regular model checkpoint
2. `*_ema_model.pt`: EMA model checkpoint

The EMA model checkpoint (`*_ema_model.pt`) typically presents better quality results and is recommended for running inference in the next stage. For more details about the conversion process and available options, refer to the script's docstring.

**Run inference:** Follow the steps in the [inference README](./inference_cosmos_transfer1_7b_sample_av.md).
