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
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
    output_dir = os.getenv("OUTPUT_DIR", "outputs/")
    uploads_dir = os.getenv("UPLOADS_DIR", "uploads/")
    log_file = os.getenv("LOG_FILE", "output.log")
    num_gpus = int(os.environ.get("NUM_GPU", 1))
    factory_module = os.getenv("FACTORY_MODULE", "cosmos_transfer1.diffusion.inference.transfer_pipeline")
    factory_function = os.getenv("FACTORY_FUNCTION", "create_transfer_pipeline")
