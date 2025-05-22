# Transfer Inference Example: Single Control (Depth) - Regional Prompt

Here is another simple example of using the Depth control with regional prompts. Many steps are similar to the [Edge example](/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge). The main difference is to use `assets/inference_cosmos_transfer1_single_control_regional_prompt.json` as the `--controlnet_specs`:

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_regional_prompt \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_regional_prompt.json \
    --offload_text_encoder_model
```

Regional Prompt is specified as shown below in the file assets/inference_cosmos_transfer1_single_control_regional_prompt.json

```
"regional_prompts": [
        {
            "prompt": "woman is wearing a red dress with white ruffles",
            "region_definitions_path": "assets/regionprompt_test/regionprompt_test_left.json"
        }
    ]
```

The input control video is a depth map video - `assets/example2_regional_prompt_depth.mp4`

<video src="https://github.com/user-attachments/assets/14bf6d57-b200-45d0-add7-4f20b68b939b">
  Your browser does not support the video tag.
</video>

This will generate a 960 x 704 video that preserves the 3D spatial structure and scene depth from the input video while enhancing visual quality, detail, and realism.

<video src="https://github.com/user-attachments/assets/0e09caba-3550-45c4-95ce-28ca0af22d25">
  Your browser does not support the video tag.
</video>


## Example 2
```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/example1_single_control_regional_prompt \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_regional_prompt_left_blue.json \
    --offload_text_encoder_model
```

Regional Prompt is specified as shown below in the file assets/inference_cosmos_transfer1_single_control_regional_prompt_left_blue.json

```
"regional_prompts": [
        {
            "prompt": "woman is wearing a blue dress with white ruffles",
            "region_definitions_path": "assets/regionprompt_test/regionprompt_test_left.json"
        }
    ]
```


#Using Multiple Prompts

## Example

PYTHONPATH=$(pwd) CHECKPOINT_DIR="/config/models/cosmos-transfer1" python cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir /config/models/cosmos-transfer1 \
    --video_save_folder outputs/example1_single_control_segment_regional_prompt \
    --controlnet_specs assets/inference_cosmos_transfer1_single_control_segment_regional_prompt.json \
    --offload_text_encoder_model --sigma_max 80 --offload_guardrail_models

Regional Prompt is specified as shown below in the file assets/inference_cosmos_transfer1_single_control_segment_regional_prompt.json

```
"regional_prompts": [
        {
            "prompt": "blue dress",
            "mask_prompt": "woman on the right"
        },
        {
            "prompt": "red dress",
            "region_definitions_path": "assets/regionprompt_test/regionprompt_test_left.json"
        }

    ]
```