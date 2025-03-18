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

from enum import Enum

from cosmos_transfer1.auxiliary.tokenizer.networks.configs import continuous_image as continuous_image_dict
from cosmos_transfer1.auxiliary.tokenizer.networks.configs import continuous_video as continuous_video_dict
from cosmos_transfer1.auxiliary.tokenizer.networks.configs import discrete_image as discrete_image_dict
from cosmos_transfer1.auxiliary.tokenizer.networks.configs import discrete_video as discrete_video_dict
from cosmos_transfer1.auxiliary.tokenizer.networks.continuous_image import ContinuousImageTokenizer
from cosmos_transfer1.auxiliary.tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from cosmos_transfer1.auxiliary.tokenizer.networks.discrete_image import DiscreteImageTokenizer
from cosmos_transfer1.auxiliary.tokenizer.networks.discrete_video import CausalDiscreteVideoTokenizer


class TokenizerConfigs(Enum):
    CI = continuous_image_dict
    DI = discrete_image_dict
    CV = continuous_video_dict
    DV = discrete_video_dict


class TokenizerModels(Enum):
    CI = ContinuousImageTokenizer
    DI = DiscreteImageTokenizer
    CV = CausalContinuousVideoTokenizer
    DV = CausalDiscreteVideoTokenizer
