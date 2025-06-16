# Transfer Inference Example: Multi Control with regionwise prompts

Here is another simple example of using multi control with regionwise prompts. Many steps are similar to the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge). The main difference is to use `assets/regionalprompt_test/inference_cosmos_transfer1_single_control_regional_prompt_video_mask.json` as the `--controlnet_specs`:

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --video_save_folder outputs/example1_multi_control_regional_prompt_video_mask \
    --controlnet_specs assets/regionalprompt_test/inference_cosmos_transfer1_multi_control_regional_prompt_video_mask.json \
    --offload_text_encoder_model --sigma_max 80 --offload_guardrail_models
```

The input video is a 1280 × 704 video with other mask videos.

Input Video
<video src="https://github.com/user-attachments/assets/78ad4072-9092-4793-8cc2-88dbdabc4db2">
  Your browser does not support the video tag.
</video>

Environment Mask
<video src="https://github.com/user-attachments/assets/d2e8060b-3f04-41ae-9e9f-3f1341358a2f">
  Your browser does not support the video tag.
</video>

Forklift Mask
<video src="https://github.com/user-attachments/assets/5b86f58a-dcc9-49a9-b2d3-780d750b9fca">
  Your browser does not support the video tag.
</video>

Worker Mask
<video src="https://github.com/user-attachments/assets/a36a211f-2940-4945-b08a-ea7095b33529">
  Your browser does not support the video tag.
</video>

This will generate a 1280 x 704 video that preserves the 3D spatial structure and scene depth from the input video while enhancing visual quality, detail, and realism.

<video src="https://github.com/user-attachments/assets/bea4d41b-8be7-4920-89cd-5b7751bd069f">
  Your browser does not support the video tag.
</video>
