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
    pip install transformer-engine[pytorch] && \
    pip install decord==0.6.0 && \
    git clone https://github.com/NVIDIA/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex && \
    rm -rf apex && \
    pip install -v decord==0.6.0 && \
    echo "Environment setup complete"

# Create Python symlink
RUN ln -s /usr/bin/python3.12 /usr/bin/python
RUN apt-get install -y libmagic1

RUN mkdir -p /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
