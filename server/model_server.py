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
"""
Main function that creates 8 worker processes with torchrun.
Workers have control loops to wait for input from the main function.
Worker processes are now handled by worker_sandbox.py.
"""
import os
import subprocess
import time
from loguru import logger as log

from server.command_ipc import WorkerCommand, WorkerStatus
from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferPipeline


class ModelServer:
    """
    A class to manage distributed worker processes using torchrun.
    """

    def __init__(self, num_workers: int = 2):

        self.num_workers = num_workers
        self.process = None
        self.worker_command = WorkerCommand(num_workers)
        self.worker_status = WorkerStatus(num_workers)
        self._setup_environment()
        self.start_workers()

    def _setup_environment(self):
        self.env = os.environ.copy()

    def start_workers(self):

        self.worker_command.cleanup()
        self.worker_status.cleanup()

        log.info(f"Starting {self.num_workers} worker processes with torchrun")

        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_workers}",
            f"--nnodes=1",
            f"--node_rank=0",
            "server/model_worker.py",
        ]

        log.info(f"Running command: {' '.join(torchrun_cmd)}")

        # Launch worker processes
        try:
            self.process = subprocess.Popen(
                torchrun_cmd,
                env=self.env,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            log.info("waiting for workers to start...")
            self.worker_status.wait_for_status()

        except Exception as e:
            log.error(f"Error starting workers: {e}")
            self.stop_workers()
            raise e

    def stop_workers(self):
        if self.process is None:
            return

        self.worker_command.broadcast("shutdown", {})

        # Wait a bit for graceful shutdown
        time.sleep(2)

        # we sent a shutdown command to all workers,
        # now we poll the torchrun process as this the only process we've started from this process
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

        log.info("All workers shut down")
        self.process = None

    def infer(self, args: dict):

        try:
            self.worker_command.broadcast("inference", args)

            log.info("Waiting for tasks to complete...")
            self.worker_status.wait_for_status()

        except Exception as e:
            # only debug here. client need to handle errors
            log.debug(f"Error during inference: {e}")
            raise e
        finally:
            # just in case a worker was hanging and didn't clean up
            self.worker_command.cleanup()

    def __del__(self):
        self.cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.info("Exiting ModelServer context")
        self.cleanup()

    def cleanup(self):
        log.info("Cleaning up ModelServer")
        self.stop_workers()
        self.worker_command.cleanup()
        self.worker_status.cleanup()
