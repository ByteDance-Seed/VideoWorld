# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from imaginaire.utils.io import save_image_or_video
from videoworld2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B, PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS
from videoworld2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS,
    dit_path="checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
    text_encoder_path="checkpoints/t5-11b",
)

# Specify the input image path and text prompt.
image_path = "/opt/tiger/PointVIS/VideoWorld_Cosmos_Predict/assets/video2world/input3.mp4"
# prompt = "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
# prompt = "A high-definition video captures the scene of handicraft production. The first frame shows a pair of hands on a piece of paper, about to pinch the lower left and right corners of the paper and fold the paper in half along the horizontal axis. After the fold is completed, the width of the paper becomes half of the original. Then the two hands press the crease."
prompt = "A driving scene, move foward."
# Run the video generation pipeline.
video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.
save_image_or_video(video, "output/drive_start0.mp4", fps=16)