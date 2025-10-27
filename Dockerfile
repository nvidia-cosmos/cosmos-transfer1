# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
RUN apt-get install -y libglib2.0-0

# Copy the cosmos-transfer1.yaml and requirements.txt files to the container
COPY ./cosmos-transfer1.yaml /cosmos-transfer1.yaml
COPY ./requirements_docker.txt /requirements.txt

RUN ls -l /usr/lib/python3/dist-packages/blinker-1.7.0.dist-info && rm -rf /usr/lib/python3/dist-packages/blinker-1.7.0.dist-info
RUN echo "Installing dependencies. This will take a while..." && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install -v --upgrade --no-build-isolation --no-dependencies sam2==1.1.0 && \
    pip install -v decord==0.6.0 && \
    echo "Environment setup complete"

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/apex-0.1+cu128.torch271-cp312-cp312-linux_x86_64.whl

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/flash_attn-2.6.3+cu128.torch271-cp312-cp312-linux_x86_64.whl

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/natten-0.21.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/transformer_engine-1.13.0+cu128.torch271-cp312-cp312-linux_x86_64.whl

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

RUN pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl

# Create Python symlink
RUN ln -s /usr/bin/python3.12 /usr/bin/python
# Fix megatron core package issue
RUN apt-get install -y libmagic1

RUN mkdir -p /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
