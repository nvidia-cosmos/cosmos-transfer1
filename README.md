<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Main Repo](https://github.com/NVIDIA/Cosmos) | [Product Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e) | [Paper](https://arxiv.org/abs/2503.14492) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-transfer1/)

Cosmos-Transfer1 is a key branch of Cosmos World Foundation Models (WFMs) specialized for multimodal controllable conditional world generation or world2world transfer. The three main branches of Cosmos WFMs are [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict1), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer1), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1). We visualize the architecture of Cosmos-Transfer1 in the following figure.

<p align="center">
    <img src="assets/transfer1_diagram.png" alt="Cosmos-Transfer1 Architecture Diagram">
</p>


Cosmos-Transfer1 includes the following:

- **ControlNet-based single modality conditional world generation** where a user can generate visual simulation based on one of the following modalities: segmentation video, depth video, edge video, blur video, LiDAR video, or HDMap video. Cosmos-Transfer1 generates a video based on the signal modality conditional input, a user text prompt, and, optionally, an input RGB video frame prompt (which could be from the last video generation result when operating in the autoregressive setting). We will use Cosmos-Transfer1-7B [Modality] to refer to the model operating in this setting. For example, Cosmos-Transfer1-7B [Depth] refers to a depth ControlNet model.

- **MultiControlNet-based multimodal conditional world generation** where a user can generate visual simulation based on any combination of segmentation video, depth video, edge video, and blur video (LiDAR video and HDMap in the AV sample) with a spatiotemporal control map to control the stregnth of each modality across space and time. Cosmos-Transfer1 generates a video based on the multimodal conditional inputs, a user text prompt, and, optionally, an input RGB video frame prompt (This could be from the last video generation result when operating in the autoregressive setting.). This is the preferred mode of Cosmos-Transfer. We will refer it as Cosmos-Transfer1-7B.

- **4KUpscaler** for upscaling a 720p-resolution video to a 4K-resolution video.

- **Post-training scripts** for helping Physical AI builders post-train pre-trained Cosmos-Transfer1 for their applications [Coming soon].

- **Pre-training scripts** for helping Physical AI builders train their own Cosmos-Transfer1 models from scratch [Coming soon].

### Example Model Behavior

[Cosmos-Transfer LiDAR + HDMap Conditional Inputs -> World](https://github.com/nvidia-cosmos/cosmos-transfer1)

<video src="https://github.com/user-attachments/assets/71faa274-a238-47c9-b2ae-5b3ea08cb643"> Your browser does not support the video tag. </video>

[Cosmos-Transfer Multimodal Conditional Inputs -> World](https://github.com/nvidia-cosmos/cosmos-transfer1)

<video src="https://github.com/user-attachments/assets/f04f430a-dc64-4ef8-b66a-70625edf860c"> Your browser does not support the video tag. </video>

## Getting Started

We provide a comphrehensive set of examples to illustrate how to perform inference, post-training, etc, with Cosmos-Transfer1. Click a relevant example below and start your Cosmos journey.

### Inference with pre-trained Cosmos-Transfer1 models

* [Inference with pre-trained Cosmos-Transfer1-7B](examples/inference_cosmos_transfer1_7b.md) **[with multi-GPU support]**
* [Inference with pre-trained Cosmos-Transfer1-7B-Sample-AV](examples/inference_cosmos_transfer1_7b_sample_av.md) **[with multi-GPU support]**
* [Inference with pre-trained Cosmos-Transfer1-7B-4KUpscaler](examples/inference_cosmos_transfer1_7b_4kupscaler.md)
* Inference with pre-trained Cosmos-Transfer1-7B [Depth]: Coming soon
* Inference with pre-trained Cosmos-Transfer1-7B [Segmentation]: Coming soon
* Inference with pre-trained Cosmos-Transfer1-7B [Edge]: Coming soon
* Inference with pre-trained Cosmos-Transfer1-7B [Vis]: Coming soon

### Post-train pre-trained Cosmos-Transfer1 models

* Coming soon

### Build your own Cosmos-Transfer1 models from scratch

* Coming soon

## Cosmos-Transfer1 Models

* [Cosmos-Transfer1-7B](https://huggingface.co/nvidia/Cosmos-Transfer1-7B): multimodal controllable conditional world generation with adaptive spatiotemporal control map. The supported modalities include segmentation, depth, canny edge, and blur visual.

* [Cosmos-Transfer1-7B [Depth|Segmentation|Edge|Vis]](https://huggingface.co/nvidia/Cosmos-Transfer1-7B): single modality controllable conditional world generation. This refers to Cosmos-Transfer1-7B operates on the single modality case and is reduced to a ControlNet.

* [Cosmos-Transfer1-7B-Sample-AV](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV): multimodal controllable conditional world generation with adaptive spatiotemporal control map specialized for autonomous vehicle applications. The supported modalities include LiDAR and HDMap.

* [Cosmos-Transfer1-7B [LiDAR|HDMap]](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-Sample-AV): single modality controllable conditional world generation for autonomous vehicle applications. This refers to Cosmos-Transfer1-7B-Sample-AV operates on the single modality case and is reduced to a ControlNet.

* [Cosmos-Transfer1-7B-4KUpscaler](https://huggingface.co/nvidia/Cosmos-Transfer1-7B-4KUpscaler): 4K upscaler to super-resolute 720p videos to 4K videos.


## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
