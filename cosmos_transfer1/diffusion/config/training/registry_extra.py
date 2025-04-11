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

'''
Registry for training experiments, callbacks and data.
'''

from hydra.core.config_store import ConfigStore

import cosmos_transfer1.diffusion.config.registry as base_registry
import cosmos_transfer1.diffusion.config.training.registry as base_training_registry
from cosmos_transfer1.diffusion.config.transfer.conditioner import CTRL_HINT_KEYS


from cosmos_transfer1.diffusion.config.transfer.registry import register_experiment_ctrlnet

# TODO (qianlim) add config / tutorial for mock data
def register_data_ctrlnet(cs):
    pass

def register_configs():
    cs = ConfigStore.instance()

    # This will register all the basic configs: net, conditioner, tokenizer.
    base_registry.register_configs()

    # This will register training configs: optimizer, scheduler, callbacks, etc.
    base_training_registry.register_configs()

    # following will register data, experiment, callbacks
    # register_data_ctrlnet(cs)  # Coming soon
    register_experiment_ctrlnet(cs)
