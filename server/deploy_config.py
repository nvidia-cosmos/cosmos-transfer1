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

import os


class Config:
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/mnt/pvc/cosmos-transfer1")
    output_dir = os.getenv("OUTPUT_DIR", "/mnt/pvc/gradio_output")
    uploads_dir = os.getenv("UPLOADS_DIR", "/mnt/pvc/gradio/uploads")
    num_gpus = int(os.environ.get("NUM_GPU", 1))
    factory_module = os.getenv("FACTORY_MODULE", "server.gradio_app")
    factory_function = os.getenv("FACTORY_FUNCTION", "create_pipeline")
