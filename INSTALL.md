## Environment setup

Clone the `cosmos-transfer1` source code
```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git
cd cosmos-transfer1
git submodule update --init --recursive
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.12.x`.

### libnvrtc check

Check libnvrtc.so exists

`find /usr -name "libnvrtc.so*" 2>/dev/null | head -n 10`

If no output then below are the steps to install libnvrtc

####  Steps to install libnvrtc

1. determine cuda version using nvidia-smi command
`nvidia-smi | grep "CUDA Version"`
2. If the cuda version output is 12.8 set CUDA_VERSION=12-8
3. CUDA_VERSION=12-8
4. apt-get update && apt-get install -y wget gnupg
5. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

If you are using ubuntu 20.04 instead of ubuntu 22.04 then run

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb

6. dpkg -i cuda-keyring_1.1-1_all.deb
7. apt-get -y update && apt-get install -y cuda-nvrtc-$CUDA_VERSION libcublas-$CUDA_VERSION libcurand-$CUDA_VERSION libcusparse-$CUDA_VERSION

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
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.2
# Install decord
pip install decord==0.6.0

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/apex-0.1+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/flash_attn-2.6.3+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/natten-0.21.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/transformer_engine-1.13.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12

apt-get install -y libmagic1
```

To test the environment setup for inference run
```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Inference using docker

If you prefer to use a containerized environment, you can build and run this repo's dockerfile to get an environment with all the packages pre-installed. This environment does not use conda. So, there is no need to specify `CUDA_HOME=$CONDA_PREFIX` when invoking this repo's scripts.

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

The below commands creates the `cosmos-transfer` conda environment and installs the dependencies for training. This is the same as required for inference.
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file cosmos-transfer1.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-transfer1
# Install the dependencies.
pip install -r requirements.txt
# Install vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.2
# Install decord
pip install decord==0.6.0
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
```

You can test the environment setup for post-training with
```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```
