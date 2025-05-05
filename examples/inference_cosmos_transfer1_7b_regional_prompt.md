# Transfer Inference Example: Single Control (Depth)

Here is another simple example of using the Depth control. Many steps are similar to the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge). The main difference is to use `assets/inference_cosmos_transfer1_single_control_depth.json` as the `--controlnet_specs`:

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_depth \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_regional_prompt.json \
    --offload_text_encoder_model
```
Same as the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge), the input video is a low-resolution 640 × 480 video.

<video src="https://github.com/user-attachments/assets/14bf6d57-b200-45d0-add7-4f20b68b939b">
  Your browser does not support the video tag.
</video>

This will generate a 960 x 704 video that preserves the 3D spatial structure and scene depth from the input video while enhancing visual quality, detail, and realism.

<video src="https://github.com/user-attachments/assets/0e09caba-3550-45c4-95ce-28ca0af22d25">
  Your browser does not support the video tag.
</video>
