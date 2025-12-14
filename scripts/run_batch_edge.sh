#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
OUT="${1:-outputs/oneclip_manyworlds}"

# Use CUDA_HOME if already set; otherwise try common defaults; don't require CONDA_PREFIX
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi

PYTHONPATH="$(pwd)" \
python cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --video_save_folder "$OUT" \
  --controlnet_specs assets/inference_cosmos_transfer1_single_control_edge.json \
  --batch_input_path configs/prompt_packs/oneclip_manyworlds.jsonl \
  --offload_text_encoder_model

