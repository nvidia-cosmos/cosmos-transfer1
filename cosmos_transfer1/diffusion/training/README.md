> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/NVIDIA/Cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-Transfer1, Cosmos 3 delivers significantly stronger transfer capabilities, enabling higher-fidelity transformation, adaptation, and simulation across diverse domains, sensors, environments, and embodiments.
>
> Beyond improving transfer quality, Cosmos 3 unifies capabilities that previously required multiple specialized models. A single Cosmos 3 model can reason, predict future world states, transfer across domains and modalities, and generate actions and policies for embodied agents within one unified architecture.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/NVIDIA/Cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-Transfer1. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

# Training Modules

This folder contains specialized versions of models and modules optimized for training. While some components (for example, the `GeneralDIT` defined in `training/networks/general_dit.py`) may appear duplicated from elsewhere in the repository, they include training-specific functionality including gradient checkpointing, training steps, tensor parallel and sequence parallel support, etc.
