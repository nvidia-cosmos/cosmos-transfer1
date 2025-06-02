## Environment setup

Clone the `cosmos-transfer1` source code
```bash
git clone git@github.com:nvidia-cosmos/cosmos-transfer1.git
cd cosmos-transfer1
git submodule update --init --recursive
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.12.x`. 

### Inference using conda

Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

The below commands create the `cosmos-transfer1` conda environment and install the dependencies for inference:
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1
# Install the dependencies.
pip install -r requirements.txt
# Install vllm
1. pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
2. export VLLM_ATTENTION_BACKEND=FLASHINFER
3. pip install vllm==0.9.0
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]
```

To test the environment setup for inference run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Inference using docker

Alternatively, if you are more familiar with a containerized environment, you can build the dockerfile and run it to get an environment with all the packages pre-installed. There is no need for conda in this setup, therefore remember to remove the prefix `CUDA_HOME=$CONDA_PREFIX` from the example commands that you see in this repo.
    
This requires docker to be already present on your system with the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
docker build -f Dockerfile . -t nvcr.io/$USER/cosmos-transfer1:latest
```

Note: In case you encounter permission issues while mounting local files inside the docker, you can share the folders from your current directory to all users (including docker) using this helpful alias 
```
alias share='sudo chown -R ${USER}:users $PWD && sudo chmod g+w $PWD'
```
before running the docker.

### Training

The below commands creates the `cosmos-transfer` conda environment and installs the dependencies for training. This is the same as required for inference but with an additional package `apex` for training with bfloat16.
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]
```

You can test the environment setup for post-training with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```
