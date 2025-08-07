# Distilling Cosmos-Transfer1 Models

In July 2025, we released a distilled version of the Cosmos-Transfer1-7B Edge model. We distilled the original 35-step Cosmos-Transfer1-7B Edge model into a single-step model while preserving output quality.

We now provide our distillation recipe and training code, so that you can replicate the diffusion step distillation process using your own models and data.

In this document, we provide the following:

- Our recipe for distilling Cosmos-Transfer1-7B Edge.
- Steps to distill your own Cosmos-Transfer1-7B models using your data.

The model is distilled separately for each control input type.

## Model Support Matrix

We support the following Cosmos-Transfer1 models for distillation. Review the available models and their compute requirements for distillation to determine the best model for your use case. We use FSDP, CP8, and gradient checkpointing for training.

| Model Name                              | Model Status | Compute Requirements for Distillation  |
|-----------------------------------------|--------------|----------------------------------------|
| Cosmos-Transfer1-7B [Depth]             | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Edge]              | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Keypoint]          | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Segmentation]      | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Vis]               | **Supported**| 8 NVIDIA GPUs*                         |

**\*** 80GB GPU memory required for distillation. `H100-80GB` or `A100-80GB` GPUs are recommended.

## Environment setup

Please refer to the training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

## Download Checkpoints

Please refer to the [training guide](/examples/training_cosmos_transfer_7b.md#download-checkpoints) for instructions on downloading checkpoints.

## Recipe

Our recipe is a two-stage distillation pipeline.

Stage 1: Knowledge Distillation (KD)

- We generated a synthetic dataset of 10,000 noise-video pairs using the teacher model for Knowledge Distillation.
- Having a strong warmup phase proved critical for subsequent DMD2 success, as this synthetic data approach significantly outperformed an alternative L2 regression warmup using real data.
- We trained the KD phase using a learning rate of 1e-5 and global batch size of 64 for 10,000 iterations.

Stage 2: Improved Distribution Matching Distillation (DMD2)

- We applied DMD2, a state-of-the-art distribution-based distillation approach that blends adversarial distillation with variational score distillation.
- The primary challenge was memory constraints from concurrent network copies (student model, teacher model, fake score network, and discriminator). We addressed this through FSDP, CP8, gradient checkpointing, and gradient accumulation to achieve an effective batch size of 64 on 16 nodes.
- We trained the DMD2 phase using a learning rate of 5e-7, guidance scale of 5, GAN loss weight of 1e-3, student update frequency of 5, and global batch size of 64 for 24,000 iterations.

## Steps

For each distillation stage, there are 3 steps involved: preparing a dataset, preparing checkpoints, and launching training.

### Knowledge Distillation (KD)

For Knowledge Distillation, you need to first prepare a dataset of teacher-generated videos and their corresponding inputs.

The dataset should contain the following:

- `videos`: generated video using the teacher model, mp4 format
- `t5_xxl`: T5 embedding of the text input used to generate the video, numpy array, shape (num_tokens, embed_dim)
- `noise`: noise input used to generate the video, numpy array, shape (16, 16, H // 8, W // 8)
- `edge`: edge control input used to generate the video, mp4 format

The dataset directory should be structured as follows:

```
datasets/kd/
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
├── noise/
│   ├── *.pickle
└── edge/
    └── *.mp4
```

File naming must be consistent across modalities. For example, to train an EdgeControl model with a video named `videos/example1.mp4`, the corresponding annotation files should be: `t5_xxl/example1.pickle`, `noise/example1.pickle`, and `edge/example1.mp4`.

#### 1. Prepare Videos and Captions

The first step is to prepare a dataset with videos and captions. You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

The videos and captions will be used to generate data for KD training, and they will also be used later for DMD2 training.

#### 2. Computing T5 Text Embeddings

Run the following command to pre-compute T5-XXL embeddings for the video captions used for training:

```bash
# The script will read the captions, save the T5-XXL embeddings in pickle format.
PYTHONPATH=$(pwd) python scripts/get_t5_embeddings.py --dataset_path datasets/kd
```

#### 3. Obtaining the Control Input Data

Next, we generate the control input data corresponding to each video. If you already have accurate control input data (e.g., ground truth depth, segmentation masks, or human keypoints), you can skip this step -- just ensure your files are organized in the above structure, and follow the data format as detailed in [Process Control Input Data](process_control_input_data_for_training.md).

Otherwise, follow the steps below to obtain the control input signals from the input RGB videos. Specifically:

- DepthControl requires a depth video that is frame-wise aligned with the corresponding RGB video. This can be obtained by, for example, running [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) on the input videos.

- SegControl requires a `.pickle` file in the SAM2 output format containing per-frame segmentation masks. See [Process Control Input Data](process_control_input_data_for_training.md) for detailed format requirements.

- KeypointControl requires a `.pickle` file containing 2D human keypoint annotations for each frame. See [Process Control Input Data](process_control_input_data_for_training.md) for detailed format requirements.

For VisControl and EdgeControl models:

- In our post-training pipeline, we compute these modalities on-the-fly. However, for Knowledge Distillation, we need to precompute the control input data for these modalities. We need the control inputs to correspond to the data used to generate the teacher output videos, rather than the control signals that are computed from the teacher output videos themselves.

- You can adapt our inference pipeline to additionally save the vis or edge control inputs used to generate the teacher output videos.

After you have prepared your KD dataset, run the following command to sanity check the dataset:

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/datasets/example_kd_dataset.py
```

#### 4. Generating Teacher Data and Saving Noise Input

Follow the steps in the [inference README](./inference_cosmos_transfer1_7b.md) to generate output videos using the teacher model. You will need to modify the inference pipeline to additionally save the noise inputs used to generate the teacher data.

#### 5. Combining the Base and Control Checkpoints

On Hugging Face, we provide the base model and control checkpoints in separate files. However, our distillation training codebase assumes a different format, where the base model and control checkpoints are combined in a single file.

Run the following command to combine the base model and control checkpoints used for distillation:

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/scripts/combine_base_ctrl_ckpt.py --ctrl_type edge
```

Inside the `checkpoints/nvidia/Cosmos-Transfer1-7B/` directory, the script will load `base_model.pt` and `edge_control.pt`, and it will save the combined checkpoint under `checkpoints_teacher/edge_control.pt`. The distillation codebase will then load the teacher model checkpoint directly from this location.

#### 6. Dry-run a Training Job

As a sanity check, run the following command to dry-run an example training job. The command will generate a full configuration of the experiment.

```bash
export OUTPUT_ROOT=checkpoints # default value

torchrun --nproc_per_node=1 -m cosmos_transfer1.distillation.train --dryrun --config=cosmos_transfer1/distillation/config/config_ctrl_kd.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_kd_train
```

Explanation of the command:

- The trainer and the passed base config script will, in the background, load the detailed experiment configurations defined in `cosmos_transfer1/distillation/config/experiment/kd/ctrl_7B_kd.py`, and register the experiments configurations for all `hint_keys` (control modalities). We use [Hydra](https://hydra.cc/docs/intro/) for advanced configuration composition and overriding.

- The `DISTILL_CTRL_7Bv1_edge_fsdp_kd_train` corresponds to an experiment name registered in `ctrl_7B_kd.py`. By specifiying this name, all the detailed config will be generated and then written to `checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_kd_train/config.yaml`.

- To customize your training, see `cosmos_transfer1/distillation/config/experiment/kd/ctrl_7B_kd.py` to understand how the detailed configs of the model, trainer, dataloader etc. are defined, and edit as needed. The base config can be found in `cosmos_transfer1/distillation/config/config_ctrl_kd.py`, and the core distillation model config can be found in `cosmos_transfer1/distillation/config/base/model.py`. See `cosmos_transfer1/distillation/config/registry.py` for the comprehensive list of registered configs you can use in your experiments.

- For demo purposes, the sample experiment config uses mock data that you can find in `cosmos_transfer1/distillation/datasets/mock_distill_dataset.py`. Once your dataset is set up properly, you can replace the `data_train` and `data_val` defaults to use your own data instead:

```python
{"override /data_train": f"kd_transfer_train_data_{hint_key}"},
{"override /data_val": f"kd_transfer_val_data_{hint_key}"},
```

#### 7. Launch Training

Now we can start a real training job! Removing the `--dryrun` and setting `--nproc_per_node=8` will start a real training job on 8 GPUs:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_kd.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_kd_train
```

**Config group and override.** An `experiment` determines a complete group of configuration parameters (model architecture, data, trainer behavior, checkpointing, etc.). Changing the `experiment` value in the command above will decide which ControlNet model is trained.

To customize your training, see the experiment config in `cosmos_transfer1/distillation/config/experiment/kd/ctrl_7B_kd.py` to understand how they are defined, and edit as needed.

It is also possible to modify config parameters from the command line. For example:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_kd.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_kd_train trainer.max_iter=100 checkpoint.save_iter=50
```

This will update the maximum training iterations to 100 (default in the registered experiments: 100_000) and checkpoint saving frequency to 50 (default: 500).

**Saving Checkpoints and Resuming Training.**
During training, the checkpoints will be saved in the structure below in FSDP DCP format.

```
checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_kd_train/checkpoints/
├── iter_{NUMBER}.pt             # "master" checkpoint, saving metadata only
├── iter_{NUMBER}_model_net      # model checkpoint folder
├── iter_{NUMBER}_optim_net      # optimizer checkpoint folder
├── iter_{NUMBER}_scheduler.pt   # scheduler checkpoint file
```

Since the `experiment` is uniquely associated with its checkpoint directory, rerunning the same training command after an unexpected interruption will automatically resume from the latest saved checkpoint.

**Saving Training Visualizations.**
During training, visualizations will be saved to `checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_kd_train/EveryNDrawSampleDistillation/`. Each visualization will show the following from top-to-bottom: student model single-step sample, teacher model single-step sample, control input, ground truth data. You can find the callback used to generate the visualizations in `cosmos_transfer1/distillation/callbacks/every_n_draw_sample.py`.

### Improved Distribution Matching Distillation (DMD2)

#### 1. Prepare the Training Data

The dataset format expected for DMD2 training distillation is the same as the format used for post-training. See the [post-training guide](./training_cosmos_transfer_7b.md#example) for instructions on preparing the dataset.

After you have prepared your DMD2 dataset, use the following script to sanity check the dataset:

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/datasets/example_transfer_dataset.py
```

#### 2. Combining the Base and Control Checkpoints

On Hugging Face, we provide the base model and control checkpoints in separate files. However, our distillation training codebase assumes a different format, where the base model and control checkpoints are combined in a single file.

Run the following command to combine the base model and control checkpoints used for distillation:

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/scripts/combine_base_ctrl_ckpt.py --ctrl_type edge
```

Inside the `checkpoints/nvidia/Cosmos-Transfer1-7B/` directory, the script will load `base_model.pt` and `edge_control.pt`, and it will save the combined checkpoint under `checkpoints_teacher/edge_control.pt`. The distillation codebase will then load the teacher model checkpoint directly from this location.

#### 3. Prepare the Negative Prompt Embedding

In the student update phase, we apply classifier-free guidance to the teacher model, using either a negative prompt or dropout to generate the unconditional prediction. We recommend using a negative prompt for optimal results.

Run the following command to precompute the negative prompt embedding used in the DMD2 training pipeline:

```bash
PYTHONPATH=$(pwd) python cosmos_transfer1/distillation/scripts/save_negative_prompt.py
```

In the script, we provide the recommended negative prompt used for Cosmos-Transfer1. The embedding will be saved to `datasets/negative_prompt/transfer1.pkl`.

#### 4. Dry-run a Training Job

As a sanity check, run the following command to dry-run an example training job. The command will generate a full configuration of the experiment.

```bash
export OUTPUT_ROOT=checkpoints # default value

torchrun --nproc_per_node=1 -m cosmos_transfer1.distillation.train --dryrun --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train
```

Explanation of the command:

- The trainer and the passed base config script will, in the background, load the detailed experiment configurations defined in `cosmos_transfer1/distillation/config/experiment/dmd2/ctrl_7B_dmd2.py`, and register the experiments configurations for all `hint_keys` (control modalities). We use [Hydra](https://hydra.cc/docs/intro/) for advanced configuration composition and overriding.

- The `DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train` corresponds to an experiment name registered in `ctrl_7B_dmd2.py`. By specifiying this name, all the detailed config will be generated and then written to `checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train/config.yaml`.

- To customize your training, see `cosmos_transfer1/distillation/config/experiment/dmd2/ctrl_7B_dmd2.py` to understand how the detailed configs of the model, trainer, dataloader etc. are defined, and edit as needed. The base config can be found in `cosmos_transfer1/distillation/config/config_ctrl_dmd2.py`, and the core distillation model config can be found in `cosmos_transfer1/distillation/config/base/model.py`. See `cosmos_transfer1/distillation/config/registry.py` for the comprehensive list of registered configs you can use in your experiments.

- For demo purposes, the sample experiment config uses mock data that you can find in `cosmos_transfer1/distillation/datasets/mock_distill_dataset.py`. Once your dataset is set up properly, you can replace the `data_train` and `data_val` defaults to use your own data instead:

```python
{"override /data_train": f"dmd2_transfer_train_data_{hint_key}"},
{"override /data_val": f"dmd2_transfer_val_data_{hint_key}"},
```

#### 5. Launch Training

Now we can start a real training job! Removing the `--dryrun` and setting `--nproc_per_node=8` will start a real training job on 8 GPUs:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train
```

**Config group and override.** An `experiment` determines a complete group of configuration parameters (model architecture, data, trainer behavior, checkpointing, etc.). Changing the `experiment` value in the command above will decide which ControlNet model is trained.

To customize your training, see the experiment config in `cosmos_transfer1/distillation/config/experiment/dmd2/ctrl_7B_dmd2.py` to understand how they are defined, and edit as needed.

It is also possible to modify config parameters from the command line. For example:

```bash
torchrun --nproc_per_node=8 -m cosmos_transfer1.distillation.train --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train trainer.max_iter=100 checkpoint.save_iter=50
```

This will update the maximum training iterations to 100 (default in the registered experiments: 100_000) and checkpoint saving frequency to 50 (default: 500).

**Loading from KD Checkpoints.**
In order to initialize the student model using a KD warmup checkpoint, override the `checkpoint.load_path` field in the experiment config. It is also necessary to set `skip_load_discriminator=True` and `skip_load_fake_score=True` to ensure that the checkpoint is loaded correctly.

**Saving Checkpoints and Resuming Training.**
During training, the checkpoints will be saved in the structure below in FSDP DCP format.

```
checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_kd_train/checkpoints/
├── iter_{NUMBER}.pt                    # "master" checkpoint, saving metadata only
├── iter_{NUMBER}_model_discriminator   # model checkpoint folder for discriminator
├── iter_{NUMBER}_model_fake_score      # model checkpoint folder for fake score network
├── iter_{NUMBER}_model_net             # model checkpoint folder for student model
├── iter_{NUMBER}_optim_discriminator   # optimizer checkpoint folder for discriminator
├── iter_{NUMBER}_optim_fake_score      # optimizer checkpoint folder for fake score network
├── iter_{NUMBER}_optim_net             # optimizer checkpoint folder for student model
├── iter_{NUMBER}_scheduler.pt          # scheduler checkpoint file
```

Since the `experiment` is uniquely associated with its checkpoint directory, rerunning the same training command after an unexpected interruption will automatically resume from the latest saved checkpoint.

**Saving Training Visualizations.**
During training, visualizations will be saved to `checkpoints/cosmos_transfer1_distill/DISTILL_CTRL_7Bv1/DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train/EveryNDrawSampleDistillation/`. Each visualization will show the following from top-to-bottom: student model single-step sample, teacher model single-step sample, control input, ground truth data. You can find the callback used to generate the visualizations in `cosmos_transfer1/distillation/callbacks/every_n_draw_sample.py`.

#### 6. Inference Using Distilled Models

**Converting the FSDP DCP checkpoints to a consolidated PyTorch checkpoint:** To convert FSDP DCP checkpoints to a consolidated PyTorch format, use the conversion script `convert_fsdp_dcp_to_native_ckpt.py`.

Example usage (with 8 GPUs):

```bash
PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 cosmos_transfer1/distillation/scripts/convert_fsdp_dcp_to_native_ckpt.py \
    --config=cosmos_transfer1/distillation/config/config_ctrl_dmd2.py -- \
    experiment=DISTILL_CTRL_7Bv1_edge_fsdp_dmd2_train \
    job.name=checkpoint_conversion \
    checkpoint.load_path=<path_to_checkpoint>
```

The script will save the consolidated checkpoint under `iter_xxxxx_full.pt` in the same checkpoint directory as the FSDP DCP checkpoints.

**Run inference:** Follow the steps in the [inference README](./inference_cosmos_transfer1_7b.md#example-2-distilled-single-control-edge) to run inference on the distilled model.
