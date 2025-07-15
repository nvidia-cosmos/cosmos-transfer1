#!/usr/bin/env python
"""Rolling-window inference demo for Cosmos-Transfer1.

This script simulates an on-line scenario in which we repeatedly feed
HD-Map + LiDAR control inputs to the model and let it generate *one* new
RGB frame at every step while keeping up to N (≤120) previously generated
frames as frozen context.

The pipeline and weights stay resident in memory, so the script can be
used as a starting point for a realtime wrapper later on.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import imageio.v3 as iio
import numpy as np
import torch
import cv2

from cosmos_transfer1.diffusion.inference.world_generation_pipeline import (
    DiffusionControl2WorldGenerationPipeline,
)
from cosmos_transfer1.utils.io import save_video
from cosmos_transfer1.utils import log
from cosmos_transfer1.diffusion.inference.inference_utils import load_controlnet_specs, validate_controlnet_specs, read_and_resize_input
from cosmos_transfer1.checkpoints import BASE_7B_CHECKPOINT_AV_SAMPLE_PATH


def _read_control_videos(spec_json: str, num_total_frames: int) -> Dict[str, np.ndarray]:
    """Pre-load control inputs using the *exact* same preprocessing routine
    as the main inference pipeline.  This delegates all heavy-lifting to
    `read_and_resize_input` from `cosmos_transfer1.diffusion.inference.inference_utils`.

    Parameters
    ----------
    spec_json : str
        Path to the ControlNet spec JSON.
    num_total_frames : int
        Desired temporal length passed to the resize helper so that the
        resulting clips match the shape expectations of the diffusion loop.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from control key (e.g. "hdmap") to an ndarray with shape
        T×H×W×C (uint8).
    """

    with open(spec_json, "r", encoding="utf-8") as fp:
        spec = json.load(fp)

    videos: Dict[str, np.ndarray] = {}
    for key, info in spec.items():
        # Skip non-control entries (e.g. "input_video_path") or malformed items
        if not isinstance(info, dict) or "input_control" not in info:
            continue

        path = info["input_control"]
        log.info(f"Pre-loading control input '{key}' from {path}")

        # Match the interpolation strategy used inside get_ctrl_batch
        interpolation = cv2.INTER_NEAREST if key == "seg" else cv2.INTER_LINEAR

        ctrl_tensor, _fps, _aspect = read_and_resize_input(
            path,
            num_total_frames=num_total_frames,
            interpolation=interpolation,
        )  # C T H W torch.uint8

        # Convert to ndarray T H W C (uint8) expected by get_ctrl_batch
        ctrl_np: np.ndarray = ctrl_tensor.numpy().transpose(1, 2, 3, 0)
        videos[key] = ctrl_np

    return videos


class RollingWindowGenerator:
    """Maintains a growing context buffer and performs step-wise generation,
    supporting both online (streaming) and offline (pre-loaded) control inputs.
    """

    def __init__(
        self,
        pipeline: DiffusionControl2WorldGenerationPipeline,
        control_inputs: dict,
        online: bool = False,
        full_control_videos: Dict[str, np.ndarray] | None = None,
        max_context: int = 120,
        warmup_frames: int = 0,
        disable_guardrail: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.control_inputs = control_inputs  # Control spec for online, data for offline
        self.online = online
        self.full_control_videos = full_control_videos
        self.max_context = max_context
        self.warmup_frames = max(0, warmup_frames)
        self.disable_guardrail = disable_guardrail
        self.context_frames: List[np.ndarray] = []  # H×W×C uint8

        if self.online and self.full_control_videos is None:
            raise ValueError("`full_control_videos` must be provided for online mode.")

        # One-shot warm-up to fill the buffer before the first `step()` call.
        if self.warmup_frames > 0:
            if self.warmup_frames > self.max_context:
                raise ValueError("warmup_frames must be ≤ max_context")

            self.pipeline.num_input_frames = 0
            self.pipeline.cutoff_frame = -1

            step_control_inputs = (
                self._get_sliced_control_inputs(self.warmup_frames)
                if self.online
                else self.control_inputs
            )

            video = self._generate_video_chunk(
                control_inputs=step_control_inputs,
                input_video_tensor=None,
            )

            if video is None:
                raise RuntimeError("Failed to generate video during warm-up.")

            # Store the first `warmup_frames` in the context buffer
            for i in range(min(self.warmup_frames, video.shape[0])):
                self.context_frames.append(video[i])

            # Trim to max_context
            if len(self.context_frames) > self.max_context:
                self.context_frames = self.context_frames[-self.max_context :]

    def _generate_video_chunk(
        self, control_inputs: dict, input_video_tensor: torch.Tensor | None
    ) -> np.ndarray | None:
        """Helper to run model and guardrail. Returns a TCHW ndarray or None."""
        video, _ = self.pipeline._run_model_with_offload(
            prompt_embedding=getattr(self.pipeline, "_cached_prompt_emb"),
            negative_prompt_embedding=getattr(self.pipeline, "_cached_neg_prompt_emb"),
            video_path="",  # no disk video
            control_inputs=control_inputs,
            input_video_tensor=input_video_tensor,
        )

        if not self.disable_guardrail:
            video = self.pipeline._run_guardrail_on_video_with_offload(video)

        return video

    def _get_sliced_control_inputs(self, num_frames: int) -> dict:
        """Return a fresh control_inputs dict with control tensors sliced to `num_frames`."""
        if not self.online or self.full_control_videos is None:
            raise RuntimeError("_get_sliced_control_inputs called in non-online mode")

        sliced_inputs: dict = {}
        for key, cfg in self.control_inputs.items():
            new_cfg = cfg.copy()
            if key in self.full_control_videos:
                full_video = self.full_control_videos[key]
                new_cfg["input_control"] = full_video[:num_frames].copy()
            sliced_inputs[key] = new_cfg
        return sliced_inputs

    def _build_context_tensor(self) -> torch.Tensor | None:
        if not self.context_frames:
            return None
        frames = np.stack(self.context_frames, axis=0)  # T H W C
        frames = frames.transpose(3, 0, 1, 2)  # C T H W
        return torch.from_numpy(frames)

    def step(self) -> np.ndarray:
        """Generate one new RGB frame and update context buffer."""
        ctx_len = len(self.context_frames)
        ctx_tensor = self._build_context_tensor() if ctx_len > 0 else None

        # Get control inputs for this step
        if self.online:
            step_control_inputs = self._get_sliced_control_inputs(ctx_len + 1)
        else:
            step_control_inputs = self.control_inputs

        # Update pipeline settings for this step
        self.pipeline.num_input_frames = ctx_len
        self.pipeline.cutoff_frame = ctx_len if ctx_len > 0 else -1

        video = self._generate_video_chunk(
            control_inputs=step_control_inputs,
            input_video_tensor=ctx_tensor,
        )

        if video is None:
            raise RuntimeError("Failed to generate video at step.")

        # The pipeline returns context + new frames; we only need the next one.
        new_frame = video[ctx_len] if video.shape[0] > ctx_len else video[-1]

        # Update context buffer (rolling)
        self.context_frames.append(new_frame)
        if len(self.context_frames) > self.max_context:
            self.context_frames = self.context_frames[-self.max_context :]

        return new_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-window Cosmos-Transfer1 demo")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--controlnet_specs", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="rolling_output.mp4")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--sigma_max", type=float, default=70.0)
    parser.add_argument("--total_frames", type=int, default=240, help="How many frames to generate in total")
    parser.add_argument("--negative_prompt", type=str, default="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.")
    parser.add_argument("--input_video_path", type=str, default="", help="Optional input RGB video path")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale value")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the output video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--blur_strength", type=str, default="medium", choices=["very_low", "low", "medium", "high", "very_high"], help="Blur strength.")
    parser.add_argument("--canny_threshold", type=str, default="medium", choices=["very_low", "low", "medium", "high", "very_high"], help="Blur strength of canny threshold applied to input.")
    parser.add_argument("--offload_diffusion_transformer", action="store_true", help="Offload diffusion transformer after inference")
    parser.add_argument("--offload_text_encoder_model", action="store_true", help="Offload text encoder model after inference")
    parser.add_argument("--offload_guardrail_models", action="store_true", help="Offload guardrail models after inference")
    parser.add_argument("--upsample_prompt", action="store_true", help="Upsample prompt using Pixtral upsampler model")
    parser.add_argument("--offload_prompt_upsampler", action="store_true", help="Offload prompt upsampler model after inference")
    parser.add_argument("--warmup_frames", type=int, default=0, help="Number of initial frames to generate before using RGB context")
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable prompt and video guardrail checks")
    parser.add_argument("--online", action="store_true", help="Stream control inputs in an online fashion (one frame at a time)")
    args = parser.parse_args()

    # Load control spec (also used by pipeline internally)
    dummy_cfg = argparse.Namespace(
        controlnet_specs=args.controlnet_specs,
        checkpoint_dir=args.checkpoint_dir,
        sigma_max=args.sigma_max,
        input_video_path=args.input_video_path,
    )
    control_inputs_raw, _ = load_controlnet_specs(dummy_cfg)
    control_inputs = validate_controlnet_specs(dummy_cfg, control_inputs_raw)

    # Build pipeline *after* we have valid control inputs so that optional
    # components such as the prompt upsampler can be initialized correctly.
    pipeline = DiffusionControl2WorldGenerationPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=BASE_7B_CHECKPOINT_AV_SAMPLE_PATH,
        offload_network=args.offload_diffusion_transformer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        fps=args.fps,
        seed=args.seed,
        control_inputs=control_inputs,
        sigma_max=args.sigma_max,
        blur_strength=args.blur_strength,
        canny_threshold=args.canny_threshold,
        upsample_prompt=args.upsample_prompt,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
    )

    # ------------------------------------------------------------
    # Load control videos depending on the chosen mode.
    # ------------------------------------------------------------
    full_control_videos = _read_control_videos(args.controlnet_specs, args.total_frames)
    if not args.online:
        # For offline mode, embed the full video data directly into the spec dict
        for k, arr in full_control_videos.items():
            if k in control_inputs and isinstance(control_inputs[k], dict):
                control_inputs[k]["input_control"] = arr
        full_control_videos = None  # Not needed anymore for offline mode

    # Store prompt in the instance for convenience (rolling inference never
    # modifies the prompt once cached, so we keep the simple behaviour).
    setattr(pipeline, "guidance_prompt", args.prompt)

    # ------------------------------------------------------------
    # Cache prompt embeddings once to avoid repeating guard-rail and
    # text-encoder passes every generation step.
    # ------------------------------------------------------------
    # 1) Ensure the (potentially upsampled) prompt is safe unless guardrail is disabled
    if not args.disable_guardrail:
        assert pipeline._run_guardrail_on_prompt_with_offload(args.prompt), "Prompt failed guard-rail check"

    # 2) Embed the prompt and the negative prompt once
    prompt_embs, _ = pipeline._run_text_embedding_on_prompt_with_offload([
        args.prompt,
        args.negative_prompt,
    ])
    cached_prompt_emb = prompt_embs[0]
    cached_neg_prompt_emb = prompt_embs[1]

    # Attach cached embeddings to the pipeline object so that they are
    # accessible inside RollingWindowGenerator
    setattr(pipeline, "_cached_prompt_emb", cached_prompt_emb)
    setattr(pipeline, "_cached_neg_prompt_emb", cached_neg_prompt_emb)

    generator = RollingWindowGenerator(
        pipeline,
        control_inputs,
        online=args.online,
        full_control_videos=full_control_videos,
        warmup_frames=args.warmup_frames,
        disable_guardrail=args.disable_guardrail,
    )

    # Pre-fill generated_frames with the warm-up frames (if any)
    generated_frames: List[np.ndarray] = list(generator.context_frames)
    if len(generated_frames) > 0:
        log.info(f"Pre-filled {len(generated_frames)} warm-up frames into the output video.")

    # Generate the remaining frames
    for _ in range(args.total_frames - len(generated_frames)):
        frame = generator.step()
        generated_frames.append(frame)

    # Save result
    video_out = np.stack(generated_frames, axis=0)  # T H W C
    save_video(video=video_out, fps=pipeline.fps, H=video_out.shape[1], W=video_out.shape[2], video_save_quality=5, video_save_path=args.output)
    log.info(f"Saved rolling video to {args.output}")


if __name__ == "__main__":
    main() 