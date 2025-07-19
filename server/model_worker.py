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
Worker sandbox for distributed processing.
Contains the worker processes that run in each distributed node.
"""

import os
import sys
import time
import traceback
import torch.distributed as dist
from loguru import logger as log
from cosmos_transfer1.diffusion.inference.transfer_pipeline import TransferPipeline
from server.command_ipc import WorkerCommand, WorkerStatus
import sys
from loguru import logger
import gc
import torch

# Configure loguru with custom color for this worker process
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <white>{level}</white> | <green>{name}</green>:<yellow>{function}</yellow>:<yellow>{line}</yellow> - <white>{message}</white>",
    level="INFO",
)

"""
this function is running in processes started by torchrun.
# TODO: dynamically load model pipeline based on MODEL_MODULE and MODEL_CLASS
# TODO: pass through env var for CHECKPOINT_DIR, so far we use default
"""


class Config:
    output_dir = os.getenv("OUTPUT_DIR", "/mnt/pvc/gradio_outdir")
    model_module = os.getenv("MODEL_MODULE")
    model_class = os.getenv("MODEL_CLASS")


def worker_main():
    """
    Worker function that runs in each distributed process.
    Has a control loop to wait for input from the model server.
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Worker init {rank+1}/{world_size}")

    worker_cmd = WorkerCommand(world_size)
    worker_status = WorkerStatus(world_size)

    try:

        pipeline = None
        if Config.model_module and Config.model_class:
            # Dynamically import the model module and class
            # module = __import__(model_module, fromlist=[model_class])
            # model_class = getattr(module, model_class)
            log.info(f"initializing model {Config.model_class} from module {Config.model_module}")
            pipeline = TransferPipeline(num_gpus=world_size, output_dir=Config.output_dir)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            log.error("initializing model: MODEL_MODULE and MODEL_CLASS environment variables are not set.")
            time.sleep(10)

        worker_status.signal_status(rank, "success", "Worker initialized successfully")

        while True:
            try:
                command_data = worker_cmd.wait_for_command(rank)
                command = command_data.get("command")
                params = command_data.get("params", {})

                log.info(f"Worker {rank} running {command=} with parameters: {params}")

                # Process different commands
                if command == "inference":

                    # read input parameters from command
                    if pipeline:
                        # Create model parameters from params
                        params, _ = TransferPipeline.validate_params(**params)
                        pipeline.infer(params)
                    else:
                        log.error("run inference: Pipeline is not initialized")
                        time.sleep(10)

                    worker_status.signal_status(rank, "success", "result_placeholder")
                elif command == "shutdown":
                    log.info(f"Worker {rank} shutting down")
                    break
                else:
                    log.warning(f"Worker {rank} received unknown command: {command}")
                    worker_status.signal_status(rank, "error", f"Unknown command: {command}")

            except Exception as e:
                log.error(f"Worker {rank} error processing command: {e}")
                error_trace = traceback.format_exc()
                log.error(f"Stack trace:\n{error_trace}")

                worker_status.signal_status(rank, "error", str(e))

    except KeyboardInterrupt:
        log.info(f"Worker {rank} keyboard interrupt, shutting down gracefully...")
    finally:
        worker_cmd.cleanup()
        worker_status.cleanup()

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":

    worker_main()
