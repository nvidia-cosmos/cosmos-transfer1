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

import copy

from cosmos_transfer1.diffusion.config.base.net import FADITV2Config
from cosmos_transfer1.diffusion.networks.general_dit_ctrl_enc import GeneralDITEncoder
from cosmos_transfer1.diffusion.networks.general_dit_ctrl_enc_multiview import GeneralDITMultiviewEncoder

num_blocks = FADITV2Config["num_blocks"]
FADITV2EncoderConfig = copy.deepcopy(FADITV2Config)
FADITV2EncoderConfig["_target_"] = GeneralDITEncoder
FADITV2EncoderConfig["layer_mask"] = [True if i > num_blocks // 2 else False for i in range(num_blocks)]

num_blocks = FADITV2Config["num_blocks"]
FADITV2MultiCamEncoderConfig = copy.deepcopy(FADITV2Config)
FADITV2MultiCamEncoderConfig["_target_"] = GeneralDITMultiviewEncoder
FADITV2MultiCamEncoderConfig["layer_mask"] = [True if i > num_blocks // 2 else False for i in range(num_blocks)]
