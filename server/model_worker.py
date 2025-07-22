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
import sys
import time
import traceback
import torch.distributed as dist
from loguru import logger as log
from server.command_ipc import WorkerCommand, WorkerStatus
from server.deploy_config import Config
import sys
from loguru import logger


# Configure loguru with custom color for this worker process
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <white>{level}</white> | <green>{name}</green>:<yellow>{function}</yellow>:<yellow>{line}</yellow> - <white>{message}</white>",
    level="INFO",
)


def create_pipeline():
    module = __import__(Config.factory_module, fromlist=[Config.factory_function])
    factory_function = getattr(module, Config.factory_function)
    log.info(f"initializing model using {Config.factory_module}.{Config.factory_function}")
    return factory_function()


"""
Entry point for the worker process. The worker process will wait for an inference request with inference parameters from the model server.
Upon completion of the request by the underlying model pipeline, the worker will signal to the server the completion by sending a status message.
"""


def worker_main():

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log.info(f"Worker init {rank+1}/{world_size}")

    worker_cmd = WorkerCommand(world_size)
    worker_status = WorkerStatus(world_size)

    try:

        pipeline = None
        if Config.model_module and Config.model_class:
            pipeline = create_pipeline()
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

                # Process commands
                if command == "inference":

                    if pipeline:
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

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":

    worker_main()
