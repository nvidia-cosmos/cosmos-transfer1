#! /bin/bash

PYTHONPATH=$(pwd) python $(pwd)/cosmos_transfer1/diffusion/inference/transfer.py \
     --checkpoint_dir /home/Nvidia/checkpoints/cosmos_transfer1 \
     --video_save_name output_video_2305_small \
     --video_save_folder outputs/sample_av_multi_control \
     --prompt "The video is captured from a camera mounted on a car. The camera is facing forward. The video showcases a scenic golden-hour drive through a suburban area, bathed in the warm, golden hues of the setting sun. The dashboard camera captures the play of light and shadow as the sun's rays filter through the trees, casting elongated patterns onto the road. The streetlights remain off, as the golden glow of the late afternoon sun provides ample illumination. The two-lane road appears to shimmer under the soft light, while the concrete barrier on the left side of the road reflects subtle warm tones. The stone wall on the right, adorned with lush greenery, stands out vibrantly under the golden light, with the palm trees swaying gently in the evening breeze. Several parked vehicles, including white sedans and vans, are seen on the left side of the road, their surfaces reflecting the amber hues of the sunset. The trees, now highlighted in a golden halo, cast intricate shadows onto the pavement. Further ahead, houses with red-tiled roofs glow warmly in the fading light, standing out against the sky, which transitions from deep orange to soft pastel blue. As the vehicle continues, a white sedan is seen driving in the same lane, while a black sedan and a white van move further ahead. The road markings are crisp, and the entire setting radiates a peaceful, almost cinematic beauty. The golden light, combined with the quiet suburban landscape, creates an atmosphere of tranquility and warmth, making for a mesmerizing and soothing drive." \
     --sigma_max 80 \
     --offload_text_encoder_model --is_av_sample \
     --controlnet_specs assets/sample_av_multi_control_spec_with_input_video.json \
     --num_gpus 1 \
     --num_steps 3 \
     --cutoff_frame 60 \
     --num_input_frames 60 \
     --save_intermediates
