<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-transfer1/)

[NVIDIA Cosmos](https://www.nvidia.com/cosmos/) is a developer-first world foundation model platform designed to help Physical AI developers build their Physical AI systems better and faster. Cosmos contains

1. Pre-trained models (available via Hugging Face) under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) that allows commercial use of the models for free.
2. Training scripts under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) for post-training the models for various downstream Physical AI applications.

## Key Features

Cosmos-Transfer1 is a pre-trained, diffusion-based conditional world model designed for multimodal, controllable world generation. It creates world simulations based on multiple spatial control inputs across various modalities, such as segmentation, depth, and edge maps. Cosmos-Transfer1 offers the flexibility to weight different conditional inputs differently at varying spatial locations and temporal instances, enabling highly customizable world generation. This capability is particularly useful for various world-to-world transfer applications, including Sim2Real.

The model is available via [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e). The post-training scripts will be released soon!

## Examples

* [Installation instructions and inference examples for Cosmos-Transfer1-7B](examples/inference_cosmos_transfer1_7b.md)
* [Installation instructions and inference examples for Cosmos-Transfer1-7B-Sample-AV](examples/inference_cosmos_transfer1_7b_sample_av.md)
* [Installation instructions and inference examples for Cosmos-Transfer1-7B-4KUpscaler](examples/inference_cosmos_transfer1_7b_4kupscaler.md)
* Cosmos-Transfer1 post-training is coming soon!

The code snippet below provides a gist of the inference usage.

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/robot_sample \
    --controlnet_specs assets/robot_sample_spec.json \
    --offload_text_encoder_model
```

<p align="center">
<video src="https://github.com/user-attachments/assets/54994029-18a9-4e79-859b-e6325179fdb7">
  Your browser does not support the video tag.
</video>
</p>

<p align="center">
<video src="https://github.com/user-attachments/assets/55daed44-5a2d-4af1-b547-e610f5ff32c6">
  Your browser does not support the video tag.
</video>
</p>

## Model Family

| Model name | Description | Try it out | Supported Hardware |
|------------|----------|----------|----------|
| [Cosmos-Transfer1-7B](https://huggingface.co/nvidia/Cosmos-Transfer1-7B) | World Generation with Adaptive Multimodal Control |[Inference](examples/inference_cosmos_transfer1_7b.md)   | 80GB H100 |
| [Cosmos-Transfer1-7B-Sample-AV](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV) | Cosmos-Transfer1 for autonomous vehicle tasks | [Inference](examples/inference_cosmos_transfer1_7b_sample_av.md) | 80GB H100 |


## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
