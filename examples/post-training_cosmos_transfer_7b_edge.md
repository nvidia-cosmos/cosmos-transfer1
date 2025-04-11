## Post-training diffusion-based EdgeControl models

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Transfer1-7B           | **Supported**    | 8 NVIDIA GPUs*                           |

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

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e):

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

Note that this will require about 300GB of free storage. Not all these checkpoints will be used in every generation.

5. The downloaded files should be in the following structure:

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

### Examples

Post-training a Cosmos-Transfer1 model enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.

#### 1. Download a Dataset

The first step is to download a dataset with videos and captions.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

For example, you can use a subset of [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) dataset for post-training.

```bash
# Download metadata with video urls and captions
mkdir -p datasets/hdvila
cd datasets/hdvila
wget https://huggingface.co/datasets/TempoFunk/hdvila-100M/resolve/main/hdvila-100M.jsonl
```

Run the following command to download the sample videos used for post-training:

```bash
# Requirements for Youtube video downloads & video clipping
pip install pytubefix ffmpeg
```

```bash
# The script will downlaod the original HD-VILA-100M videos, save the corresponding clips, the captions and the metadata.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_example_data.py --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip
```

#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will read the captions, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings.py --dataset_path datasets/hdvila
```

Dataset folder format:
```
datasets/hdvila/
├── metas/
│   ├── *.json
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

Training a VisControl or EdgeControl model is self-supervised: we apply blurs and/or compute canny edges of the input videos on-the-fly during training. Therefore, for these two modalities there is no need to prepare the control input videos separately.

#### 3. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_transfer1.diffusion.training.train --config=cosmos_transfer1/diffusion/training/config/config.py --experiment=CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3
```

checkpoints/cosmos_transfer1/CTRL_7Bv1_lvg/CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3/config.yaml
This command will use ``cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_121frames.py` to register experiments for all `hint_keys` (control modalities).

Then the model will be post-trained using the above hdvila dataset.
See the function `make_ctrlnet_config_7b_training` defined in `cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_121frames.py` to understand how the detailed configs of the model, trainer, dataloader etc. are defined. For the data specifically:

```python
num_frames = 121
example_video_dataset = L(Dataset)(
    dataset_dir="datasets/hdvila",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train = L(DataLoader)(
    dataset=example_video_dataset,
    sampler=L(get_sampler)(dataset=example_video_dataset),
    batch_size=1,
    drop_last=True,
)
...

config = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train,
        ...
    )
)
...
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `cosmos_transfer1_posttrain`, `GROUP` is `CTRL_7Bv1_lvg`, `NAME` is `CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3`.

See the job config to understand how they are determined.
```python
edgecontrol_7b_example_hdvila = LazyDict(
    dict(
        ...
        job=dict(
            project="cosmos_transfer1_posttrain",
            group="CTRL_7Bv1_lvg",
            name="CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/cosmos_transfer1_posttrain/CTRL_7Bv1_lvg/CTRL_7Bv1pt3_lvg_tp_121frames_control_input_edge_block3/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```