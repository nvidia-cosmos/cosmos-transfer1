# Training Modules

This folder contains specialized versions of models and modules optimized for training. While some components (for example, the `GeneralDIT` defined in `training/networks/general_dit.py`) may appear duplicated from elsewhere in the repository, they include training-specific functionality including gradient checkpointing, training steps, tensor parallel and sequence parallel support, etc.
