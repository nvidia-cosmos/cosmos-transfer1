from gradio_client import Client, handle_file

client = Client("http://localhost:8080/")


result = client.predict(
    input_video={"video": handle_file("assets/example1_input_video.mp4")},
    prompt="The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.",
    negative_prompt="The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
    vis_enable=False,
    vis_weight=0.5,
    edge_enable=True,
    edge_weight=1,
    depth_enable=False,
    depth_weight=0.5,
    seg_enable=False,
    seg_weight=0.5,
    keypoint_enable=False,
    keypoint_weight=0.5,
    guidance_scale=7,
    num_steps=35,
    seed=1,
    sigma_max=70,
    blur_strength="medium",
    canny_threshold="medium",
    api_name="/infer_wrapper",
)

print(result)
