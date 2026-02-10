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
python3 examples/video2world_action.py \
  --model_size 2B \
  --dit_path "checkpoints/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt" \
  --input_video datasets/bridge/videos/test/13/rgb.mp4 \
  --input_annotation datasets/bridge/annotation/test/13.json \
  --num_conditional_frames 1 \
  --save_path output/action_video.mp4 \
  --guidance 0 \
  --seed 0 \
  --disable_guardrail \
  --disable_prompt_refiner