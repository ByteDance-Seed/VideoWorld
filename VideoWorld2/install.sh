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
# cd VideoWorld2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --user
pip install --no-build-isolation transformer-engine[pytorch]==1.12.0

cd ../
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .
cd /opt/tiger/PointVIS/VideoWorld2
python3 scripts/test_environment.py
pip install scikit-image
pip install fvcore
pip install qwen_vl_utils
pip install flash-attn==2.6.3 --no-build-isolation
pip install albumentations
pip install kornia
pip install lpips

sudo apt-get install git-lfs
git lfs install

mkdir -p checkpoints
mkdir -p datasets/Video-CraftBench