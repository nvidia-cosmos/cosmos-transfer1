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
Sample gradio client interactions for the Cosmos Transfer1 Gradio app.

Example usage:

    # Synchronous inference with a file that already exists on the server
    python sample_client.py \
        --url http://localhost:8080/ \
        --example sync \
        --input_video_path path/to/video/on/server.mp4

    # Asynchronous upload + inference
    python sample_client.py \
        --url https://cosmos-transfer1.inference.dgxcloud.ai/ \
        --example async_with_upload \
        --input_video_path path/to/video/on/local/machine.mp4
"""

import argparse
import json
import time
import typing

import gradio_client.client as gradio_client
import gradio_client.utils as gradio_utils
from loguru import logger


def _request(input_video_path: str) -> typing.Dict[str, typing.Any]:
    return {
        "blur_strength": "medium",
        "canny_threshold": "medium",
        "edge": {"control_weight": 1.0},
        "guidance": 7.0,
        "input_video_path": input_video_path,
        "negative_prompt": "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
        "num_steps": 35,
        "prompt": "The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
        "seed": 1,
        "sigma_max": 70.0,
    }


def _sync_example(url: str, input_video_path: str):
    """
    Synchronous inference with an input video file that already exists on the server.

    - client.predict(api_name="/generate_video") -> local_video_path
    """
    logger.info("--------------------------------")
    logger.info("Synchronous inference with file on server")

    client = gradio_client.Client(url)

    request_dict = _request(input_video_path=input_video_path)
    request_text = json.dumps(request_dict)

    logger.info(f"generate_video_request: {request_text=}")
    result = client.predict(request_text, api_name="/generate_video")
    logger.info(f"generate_video_result: {result=}")

    local_video_path = result[0]["video"]
    logger.info(f"Local video path (downloaded to local machine): {local_video_path=}")


def _sync_with_upload_example(url: str, input_video_path: str):
    """
    Synchronous inference with an input video file that exists on the client machine.

    - client.predict(api_name="/upload_file") -> remote_path
    - client.predict(remote_path, api_name="/generate_video") -> local_video_path
    """
    logger.info("--------------------------------")
    logger.info("Synchronous inference with local file")

    client = gradio_client.Client(url)

    # Upload the local file to the server and get the remote path
    file_descriptor = gradio_utils.handle_file(input_video_path)
    upload_file_result_str = client.predict(file_descriptor, api_name="/upload_file")
    upload_file_result_dict = json.loads(upload_file_result_str)
    logger.info(f"{upload_file_result_dict=}")
    remote_path = upload_file_result_dict["path"]

    # Use the remote path in the request
    request_dict = _request(input_video_path=remote_path)
    request_text = json.dumps(request_dict)

    logger.info(f"generate_video_request: {request_text=}")
    result = client.predict(request_text, api_name="/generate_video")
    logger.info(f"generate_video_result: {result=}")

    local_video_path = result[0]["video"]
    logger.info(f"Local video path (downloaded to local machine): {local_video_path=}")


def _async_example(url: str, input_video_path: str):
    """
    Asynchronous inference with an input video file that already exists on the server.

    - client.submit(api_name="/generate_video") -> async job
    - wait_for_job() -> local_video_path
    """
    logger.info("--------------------------------")
    logger.info("Asynchronous inference with file on server")

    client = gradio_client.Client(url)

    request_dict = _request(input_video_path=input_video_path)
    request_text = json.dumps(request_dict)

    logger.info(f"generate_video_request: {request_text=}")
    job = client.submit(request_text, api_name="/generate_video")
    _job_status, result, _error = _async_wait_for_job(job)

    local_video_path = result[0]["video"]
    logger.info(f"Local video path (downloaded to local machine): {local_video_path=}")


def _async_with_upload_example(url: str, input_video_path: str):
    """
    Asynchronous inference with an input video file that exists on the client machine.

    - client.submit(api_name="/upload_file") -> async job
    - wait_for_job(upload_job) -> remote_path
    - client.submit(remote_path, api_name="/generate_video") -> async job
    - wait_for_job() -> local_video_path
    """
    logger.info("--------------------------------")
    logger.info("Asynchronous inference with local file")

    client = gradio_client.Client(url)

    # Upload the local file to the server, wait for job to complete, and get the remote path
    file_descriptor = gradio_utils.handle_file(input_video_path)
    upload_job = client.submit(file_descriptor, api_name="/upload_file")
    _job_status, upload_file_result_str, _error = _async_wait_for_job(upload_job)
    upload_file_result_dict = json.loads(upload_file_result_str)
    remote_path = upload_file_result_dict["path"]
    logger.info(f"{remote_path=}")

    # Use the remote path in the request, wait for job to complete, and get the local video path
    request_dict = _request(input_video_path=remote_path)
    request_text = json.dumps(request_dict)

    logger.info(f"generate_video_request: {request_text=}")
    generate_job = client.submit(request_text, api_name="/generate_video")
    _job_status, generate_result, _error = _async_wait_for_job(generate_job)
    local_video_path = generate_result[0]["video"]

    logger.info(f"Local video path (downloaded to local machine): {local_video_path=}")


def _async_wait_for_job(
    job: gradio_client.Job,
) -> typing.Tuple[gradio_client.StatusUpdate, typing.Any, typing.Optional[Exception]]:
    """
    Waits for a job to complete.

    Returns (tuple):
        - job_status: The status of the job
        - result: The result of the job
        - error: The error of the job
    """
    while not job.done():
        logger.info(f"Waiting for job {job=} {job.status()=}")
        time.sleep(5)

    job_status: gradio_client.StatusUpdate = job.status()
    if job_status.success:
        try:
            result = job.result(timeout=20)
            logger.info(f"[SUCCESS] {result=}")
            return job_status, result, None
        except Exception as e:
            logger.error(f"[EXCEPTION] {job_status=} {e=}")
            return job_status, None, e
    else:
        logger.warning(f"[NO_OUTPUT] {job_status=}")
        return job_status, None, None


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--url", required=True, type=str, default="http://localhost:8080/")
    args.add_argument(
        "--example", required=True, choices=["sync", "sync_with_upload", "async", "async_with_upload"], default="sync"
    )
    args.add_argument("--input_video_path", required=True, type=str, default="assets/example1_input_video.mp4")
    args = args.parse_args()

    if args.example == "sync":
        _sync_example(args.url, args.input_video_path)

    elif args.example == "sync_with_upload":
        _sync_with_upload_example(args.url, args.input_video_path)

    elif args.example == "async":
        _async_example(args.url, args.input_video_path)

    elif args.example == "async_with_upload":
        _async_with_upload_example(args.url, args.input_video_path)


if __name__ == "__main__":
    main()
