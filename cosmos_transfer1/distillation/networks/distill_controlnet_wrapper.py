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

import torch
from torch.distributed import ProcessGroup

from cosmos_transfer1.diffusion.training.modules.blocks import DITBuildingBlock
from cosmos_transfer1.utils.lazy_config import instantiate as lazy_instantiate


class DistillControlNet(torch.nn.Module):
    r"""Wrapper class for the control net.

    This class wraps the control net (self.net_ctrl) and the base model (self.base_model) into a single class for distillation purpose.
    In distillation, both the control net and the base model are getting updated.
    For example, in DMD2, the student and the fake score are instantiated from this class.

    This class also accommodates the forward method of the control net, which requires the base model as an argument and
    call the base_model.net.

    Args:
        config (Config): Configuration

    """

    def __init__(self, config):
        super().__init__()

        self.cp_group = None
        self.net_ctrl = lazy_instantiate(config.net_ctrl)

        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = lazy_instantiate(config.net)

        self.base_model = BaseModel()

    def forward(self, *args, **kwargs):
        # The following line is necessary for their original forward method in class GeneralDITEncoder to work properly.
        kwargs["base_model"] = self.base_model
        return self.net_ctrl(*args, **kwargs)

    @property
    def fsdp_wrap_block_cls(self):
        return DITBuildingBlock

    def enable_context_parallel(self, cp_group: ProcessGroup):
        self.base_model.net.enable_context_parallel(cp_group)
        self.net_ctrl.enable_context_parallel(cp_group)
        self.cp_group = cp_group

    def disable_context_parallel(self):
        self.base_model.net.disable_context_parallel()
        self.net_ctrl.disable_context_parallel()
        self.cp_group = None

    def enable_sequence_parallel(self):
        self.base_model.net.enable_sequence_parallel()
        self.net_ctrl.enable_sequence_parallel()

    def disable_sequence_parallel(self):
        self.base_model.net.disable_sequence_parallel()
        self.net_ctrl.disable_sequence_parallel()

    def _set_sequence_parallel(self, status: bool):
        self.base_model.net._set_sequence_parallel(status)
        self.net_ctrl._set_sequence_parallel(status)

    @property
    def is_context_parallel_enabled(self):
        return (self.base_model.net.cp_group is not None) and (self.net_ctrl.cp_group is not None)
