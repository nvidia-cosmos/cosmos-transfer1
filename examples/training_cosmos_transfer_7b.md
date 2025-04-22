# Training Cosmos-Transfer1 Models
In this document, we provide examples and steps to:
- Build your own Cosmos-Transfer1 models, training from scratch; or
- Post-train Cosmos-Transfer1 models from our checkpoint using your data.

The model is trained separately for each control input type.


### Model Support Matrix
We support the following Cosmos-Transfer models for pre-training and post-training. Review the available models and their compute requirements for post-training and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|------------------------------------------|--------------|----------------------------------------|
| Cosmos-Transfer1-7B [Depth]             | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Segmentation]      | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Edge]              | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1-7B [Vis]               | **Supported**| 8 NVIDIA GPUs*                         |
| Cosmos-Transfer1pt1-7B [Keypoint]       | **Supported**| 8 NVIDIA GPUs*                         |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### Environment setup

Please refer to the Post-training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

### Download Checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token. Set the access token to 'Read' permission (default is 'Fine-grained').

2. Log in to Hugging Face with the access token:

```bash
huggingface-cli login
```

3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e). Note that this will require about 300GB of free storage.

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
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
- Base model (`base_model.pt`) and tokenizer models (under `Cosmos-Tokenize1-CV8x8x8-720p`): Required for all training.
- Control modality-specific model checkpoint (e.g., `seg_control.pt`): Only needed for post-training that specific control. Not needed if training from scratch.
- Other folders such as `depth-anything`, `facebook/sam2-hiera-large` etc.: optional. These are helper modules to process the video data into the respective control modalities such as depth and segmentation.

### Example
There are 3 steps to train a Cosmos-Transfer1 model: preparing a dataset, prepare checkpoints, and launch training.

In the example below, we use a subset of [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) dataset to demonstrate the steps for preparing the data and launching training. After preprocessing, your dataset directory should be structured as follows:
```
datasets/hdvila/
├── metas/
│   ├── *.json
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
├── keypoint/
│   ├── *.pickle
├── depth/
│   ├── *.mp4
├── seg/
│   ├── *.pickle
└── <your control input modality>/
    ├── <your files>
```

File naming must be consistent across modalities. For example, to train a SegControl model with a video named `videos/example1.mp4`, the corresponding annotation files should be: `seg/example1.pickle`.

Note: Only the folder corresponding to your chosen control input modality is required. For example, if you're training with depth as the control input, only the `depth/` subfolder is needed.

#### 1. Prepare Videos and Captions

The first step is to prepare a dataset with videos and captions. You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

Here we use a subset of sample videos from HD-VILA-100M as an example:

```bash
# Download metadata with video urls and captions
mkdir -p datasets/hdvila
cd datasets/hdvila
wget https://huggingface.co/datasets/TempoFunk/hdvila-100M/resolve/main/hdvila-100M.jsonl
```

Run the following command to download the sample videos used for training:

```bash
# Requirements for Youtube video downloads & video clipping
pip install pytubefix ffmpeg
```

```bash
# The script will downlaod the original HD-VILA-100M videos, save the corresponding clips, the captions and the metadata.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_example_data.py --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip
```

#### 2. Computing T5 Text Embeddings
Run the following command to pre-compute T5-XXL embeddings for the video captions used for training:

```bash
# The script will read the captions, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings.py --dataset_path datasets/hdvila
```

#### 3. Obtaining the Control Input Data
Next, we generate the control input data corresponding to each video. If you already have accurate control input data (e.g., ground truth depth, segmentation masks, or human keypoints), you can skip this step -- just ensure your files are organized in the above structure, and follow the data format as detailed in [Process Control Input Data](process_control_input_data_for_training.md).

Here, as an example, we show show how to obtain the control input signals from the input RGB videos. Specifically:

- DepthControl requires a depth video that is frame-wise aligned with the corresponding RGB video. This can be obtained by, for example, running [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) on the input videos.

- SegControl requires a `.pickle` file in the SAM2 output format containing per-frame segmentation masks. See [Process Control Input Data](process_control_input_data_for_training.md) for detailed format requirements.

- KeypointControl requires a `.pickle` file containing 2D human keypoint annotations for each frame. See [Process Control Input Data](process_control_input_data_for_training.md) for detailed format requirements.

For VisControl and EdgeControl models: training is self-supervised. These models get control inputs (e.g., by applying blur or extracting Canny edges) from the input videos on-the-fly during training. Therefore, you do not need to prepare control input data separately for these modalities.




#### 4. Splitting the Checkpoints to TensorParallel Checkpoints
Due to the large model size, we leverage TensorParallel (TP) to split the model weights across multiple GPUs. We use 8 for the TP size.

```bash
# Will split the Base model checkpoint into 8 TP checkpoints
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B/base_model.pt
# Example: for EdgeControl checkpoint splitting for post-train.
PYTHONPATH=. python scripts/convert_ckpt_fsdp_to_tp.py checkpoints/nvidia/Cosmos-Transfer1-7B/edge_control.pt
```
This will generate the TP checkpoints under `checkpoints/checkpoints_tp/*_mp_*.pt`, which we load in the training below.

#### 5. Launch Training
Now we can start training! Run the following command to dry-run an example training job with the above data:
```bash
export OUTPUT_ROOT=checkpoints # default value

# Training from scratch
torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain

# Post-train from our provided checkpoint (need to first split checkpoint into TP checkpoints as instructed above)
torchrun --nproc_per_node=1 -m cosmos_transfer1.diffusion.training.train --dryrun --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_posttrain
```

Explanation of the command:

- The trainer and the passed (master) config script will, in the background, load the detailed experiment configurations defined in `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_121frames.py`, and register the experiments configurations for all `hint_keys` (control modalities), covering both pretrain and post-train. We use [Hydra](https://hydra.cc/docs/intro/) for advanced configuration composition and overriding.

- The `CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain` corresponds to an experiment name registered in `ctrl_7b_tp_121frames.py`. By specifiying this name, all the detailed config will be loaded. The full configuration is also written to `checkpoints/cosmos_transfer1_pretrain/CTRL_7Bv1_lvg/CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain/config.yaml`.

- To customize your training, see `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_121frames.py` to understand how the detailed configs of the model, trainer, dataloader etc. are defined, and edit as needed.

- Removing the `--dryrun` and set `--nproc_per_node=8` will start a real training job on 8 GPUs:

    ```bash
    torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/config/config_train.py -- experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain
    ```

- Change the `experiment` value will decide which control modality model is trained, and whether it's pretrain or post-train. For example, replacing the experiment name in the command with `CTRL_7Bv1pt3_lvg_tp_121frames_control_input_depth_block3_posttrain` will post-train the DepthControl model from the downloaded checkpoint instead.

- The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`. See the job config to understand how they are determined:

    ```python
    # In cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_121frames.py
    config = LazyDict(
        dict(
            ...
            job=dict(
                project="cosmos_transfer1_pretrain",
                group="CTRL_7Bv1_lvg",
                name="CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain",
            ),
            ...
        )
    )
    ```

    During the training, the checkpoints will be saved in the below structure. Since we use TensorParallel across 8 GPUs, 8 checkpoints will be saved each time.

    ```
    checkpoints/cosmos_transfer1_pretrain/CTRL_7Bv1_lvg/CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3_pretrain/checkpoints/
    ├── iter_{NUMBER}.pt             # "master" checkpoint, saving metadata only
    ├── iter_{NUMBER}_model_mp_0.pt  # real TP checkpoints
    ├── iter_{NUMBER}_model_mp_1.pt
    ├── ...
    ├── iter_{NUMBER}_model_mp_7.pt
    ```

- Since the `experiment` is uniquely associated with its checkpoint directory, rerunning the same training command after an unexpected interruption will automatically resume from the latest saved checkpoint.
