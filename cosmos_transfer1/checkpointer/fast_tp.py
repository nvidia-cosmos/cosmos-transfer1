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

from typing import Any, Set

import torch

from cosmos_transfer1.checkpointer.ddp_checkpointer import StateDictItemPath
from cosmos_transfer1.checkpointer.tp_checkpointer import Checkpointer as TPCheckpointer
from cosmos_transfer1.diffusion.training.models.model import DiffusionModel
from cosmos_transfer1.utils import distributed, log, misc
from cosmos_transfer1.utils.easy_io import easy_io


class Checkpointer(TPCheckpointer):
    def load_broadcast_state_dict(
        self, checkpoint_path: str, model: DiffusionModel, resume_keys: Set
    ) -> dict[str, Any]:
        """
        Load state_dict and broadcast efficiently.

        This method optimizes checkpoint loading for distributed training for improved
        connection speed and reliability.

        The main steps are:
        1. Retrieve TP-rank-specific checkpoints for each GPU of DDP-rank 0
           and CP-rank 0.
        2. Each rank loads its corresponding checkpoint either from a local cache or
           receives it via broadcast.

        This approach ensures that each MP (Model Parallelism) rank loads its specific
        part of the model, which is crucial for scenarios where different parts of the
        model are distributed across multiple GPUs.

        The method supports both Tensor Parallelism (TP) and standard Data Parallel (DP)
        training. For TP, each rank can efficiently load its specific checkpoint from S3.
        For standard DDP without TP, the default broadcast mechanism is used.

        Args:
            checkpoint_path (str): The base path of the checkpoint in S3.
            model (DiffusionModel): The model being loaded.
            resume_keys (Set): Set of keys to resume from the checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the loaded state for each resumed key.

        Note:
            This implementation has been tested and optimized for 4K GPU training jobs,
            showing significant improvements in connection speed and overall efficiency.
        """
        state_dict = {}
        sorted_resume_keys = sorted(resume_keys)
        for key in sorted_resume_keys:
            _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
            _state_dict = easy_io.load(_ckpt_path, weights_only=False)
            state_dict[key] = _state_dict
            self.print(f"Loaded checkpoint from: {_ckpt_path}")
        distributed.barrier()
        return state_dict

    @misc.timer("checkpoint saving")
    def _save_worker(self, state_dict: dict[str, StateDictItemPath], checkpoint_file: str, rank: int = 0) -> None:
        """
        similar to the original _save_worker, but with the following changes:
        * fast_backend=False to avoid high CPU usage
        """
        try:
            for key, item in state_dict.items():
                self.print(f"Saving {key} to {item.save_path}")
                try:
                    easy_io.dump(
                        item.state_dict,
                        item.save_path,
                        # fast_backend=False,  # too cpu heavy
                    )
                    self.print(f"Saved {key} to {item.save_path}")
                except Exception as e:
                    self.print(f"Failed to save {key} to {item.save_path}: {str(e)}")
                    raise  # Re-raise the exception after logging

            # Synchronize only rank 0 of each model parallel group
            if self.mp_world_size > 1:
                torch.distributed.barrier(group=self.mp_gloo_pg)

            # Only rank 0 of MP group and rank 0 of DP with CP updates latest_checkpoint.txt
            if self.mp_rank == 0 and self.rank_dp_w_cp == 0:
                self._write_latest_checkpoint_file(checkpoint_file)

            if distributed.get_rank() == 0:  # only rank 0 saves trained_data_record
                if "trained_data_record" in state_dict["model"].state_dict:
                    self._write_trained_data_record(
                        checkpoint_file, state_dict["model"].state_dict["trained_data_record"]
                    )

            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to upload: {e}", rank0_only=not self.verbose)
