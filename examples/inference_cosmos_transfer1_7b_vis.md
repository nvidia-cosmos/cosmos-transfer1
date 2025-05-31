# Transfer Inference Example: Single Control (Vis)

Here is another simple example of using the Vis control. Many steps are similar to the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge). The main difference is to use `assets/inference_cosmos_transfer1_single_control_vis.json` as the `--controlnet_specs`:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_vis \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_vis.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```
Same as the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge), the input video is a low-resolution 640 × 480 video.

<video src="https://github.com/user-attachments/assets/cb9bd7b8-3d8b-4648-a5dc-492c84dd5faa">
  Your browser does not support the video tag.
</video>

This will generate a 960 x 704 video that preserves the overall color palette, lighting, and coarse structure from the input video using the vis control. By guiding the generation with a blurred version of the input, the model maintains the original scene's visual feel while significantly enhancing visual quality, detail, and realism based on the provided prompt.

<video src="https://github.com/user-attachments/assets/2e9ef23c-5356-4d26-aedb-b19752560581">
  Your browser does not support the video tag.
</video>
