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

import importlib
import os
import sys

if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\033[91m[ERROR]\033[0m Python 3.10+ is required. You have: \033[93m{detected}\033[0m")
    sys.exit(1)

if "CONDA_PREFIX" not in os.environ:
    print("\033[93m[WARNING]\033[0m Cosmos should be run under a conda environment.")

print("Attempting to import critical packages...")

packages = [
    "torch",
    "torchvision",
    "transformers",
    "megatron.core",
    "transformer_engine",
]
all_success = True
te_success = True

for package in packages:
    try:
        _ = importlib.import_module(package)
    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m Package not successfully imported: \033[93m{package}\033[0m")
        if package == "transformer_engine":
            te_success = False
        else:
            all_success = False
    else:
        print(f"\033[92m[SUCCESS]\033[0m {package} found")

if all_success:
    print("-----------------------------------------------------------")
    if not te_success:
        print(
            "\033[93m[WARNING]\033[0m Cosmos environment setup is successful (\033[93mtransformer-engine\033[0m is not available)."
        )
    else:
        print("\033[92m[SUCCESS]\033[0m Cosmos environment setup is successful!")
