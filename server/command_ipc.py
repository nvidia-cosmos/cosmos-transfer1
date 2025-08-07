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


import json
import os
import time
from typing import Any, Dict, Optional

from loguru import logger as log


class WorkerCommand:
    """wrapper around file based IPC command"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_commands.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def _send_command_to_worker(self, rank: int, command: str, params: Optional[Dict[str, Any]] = None):
        command_file = f"/tmp/worker_{rank}_commands.json"
        command_data = {"command": command, "params": params or {}}

        with open(command_file, "w") as f:
            json.dump(command_data, f)

        log.debug(f"Sent command '{command}' to worker {rank}")

    def broadcast(self, task_name: str, task_params: Dict[str, Any]):
        """Broadcast non-blocking a task to all workers."""
        log.debug(f"Broadcasting task '{task_name}' to all workers...")

        for rank in range(self.num_workers):
            self._send_command_to_worker(rank, task_name, task_params)

    def wait_for_command(self, rank: int) -> Optional[Dict[str, Any]]:
        """wait blocking for a command from the worker.

        This is an infinite blocking call by design. we want to infintely wait until typically user is sending a request to the worker.
        """
        command_file = f"/tmp/worker_{rank}_commands.json"
        log.debug(f"worker {rank}: Waiting for command file {command_file}")
        while not os.path.exists(command_file):
            time.sleep(0.5)

        try:
            with open(command_file, "r") as f:
                command_data = json.load(f)
            os.remove(command_file)  # Remove command file after reading
            return command_data
        except Exception as e:
            log.error(f"Failed to read command file for worker {rank}: {e}")
            raise e


class WorkerException(Exception):
    def __init__(self, message, status_dict=None):
        super().__init__(message)
        self.details = status_dict or {}

    def __str__(self):
        return f"{super().__str__()}\nstatus of each worker: {json.dumps(self.details, indent=4)}"


class WorkerStatus:
    """wrapper around file based IPC status"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def cleanup(self):
        for rank in range(self.num_workers):
            for file_path in [f"/tmp/worker_{rank}_status.json"]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def signal_status(self, rank: int, status: str, message: str = "") -> None:
        """signal individual worker status per rank"""
        status_file = f"/tmp/worker_{rank}_status.json"

        log.debug(f"worker {rank} status: {status}, message: {message}")
        with open(status_file, "w") as f:
            json.dump(
                {"rank": rank, "status": status, "result": message},
                f,
            )

    def _get_worker_status(self, rank: int, timeout: int = 1800) -> Dict[str, Any]:
        status_file = f"/tmp/worker_{rank}_status.json"
        start_time = time.time()

        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                os.remove(status_file)
                return {"status": "timeout", "rank": rank}
            time.sleep(0.5)

        try:
            with open(status_file, "r") as f:
                status = json.load(f)

            # remove status file so we can do a blocking wait for next status
            log.debug(f"Worker {rank} removing status file {status_file}")
            os.remove(status_file)

            assert os.path.exists(status_file) is False, "status file should be removed after processing"
            return status

        except Exception:
            log.error(f"Failed to read status file for worker {rank}")
            return {"status": "unknown", "rank": rank}

    def wait_for_status(self, timeout: int = 1800) -> bool:
        statuses = {}
        """blocking call to wait for completion of all workers

            This functions waits for all workers to signal their status.
            Upon failure of any worker, it raises a WorkerException with a compound status dictionary.
        """

        # Collect statuses from all workers, ensure status file is removed after reading
        for rank in range(self.num_workers):
            statuses[rank] = self._get_worker_status(rank, timeout)

        for rank, worker_status in statuses.items():
            if worker_status.get("status") != "success":
                log.error(f"Worker {rank} failed: {worker_status.get('status', 'unknown')}")
                raise WorkerException(
                    f"Worker {rank} failed with status: {worker_status.get('status', 'unknown')}", statuses
                )

        log.debug("All workers reported success")
