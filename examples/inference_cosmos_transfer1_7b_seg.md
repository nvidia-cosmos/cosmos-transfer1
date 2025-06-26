# Transfer Inference Example: Single Control (Segmentation)

Here is another simple example of using the Segmentation control. Many steps are similar to the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge). The main difference is to use `assets/inference_cosmos_transfer1_single_control_seg.json` as the `--controlnet_specs`:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_seg \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_seg.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU
```
Same as the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge), the input video is a low-resolution 640 × 480 video.

<video src="https://github.com/user-attachments/assets/91c99bc4-8cda-434e-ade8-6735ba9fbda4">
  Your browser does not support the video tag.
</video>

This will generate a 960 x 704 video that maintains the structural consistency from the input video while enhancing visual quality, detail, and realism.

<video src="https://github.com/user-attachments/assets/08058680-be85-4571-81b2-cd56b51e41f5">
  Your browser does not support the video tag.
</video>
