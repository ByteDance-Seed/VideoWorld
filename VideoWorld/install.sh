# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
pip install git+https://github.com/open-mmlab/mmengine.git@66fb81f7b392b2cd304fc1979d8af3cc71a011f5
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install sgfmill
pip install gym
pip install gateloop_transformer
pip install lpips
sudo rm -r /home/tiger/.local/lib/python3.9/site-packages/sgfmill/boards.py
sudo cp ./VideoWorld/tools/go_utils/boards.py /home/tiger/.local/lib/python3.9/site-packages/sgfmill/
# cd ../
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=./VideoWorld
cd VideoWorld
rm -r ../calvin/calvin_models/requirements.txt
cp ./calvin_req.txt ../calvin/calvin_models/requirements.txt
cd ../calvin
pip install setuptools==57
sh install.sh
pip install pytorch_lightning
pip install flamingo_pytorch
pip3 install git+https://github.com/openai/CLIP.git
cd ../

pip install transformers==4.31.0
pip install rotary_embedding_torch
pip install pycocotools
pip install kornia
#Katrain install 
pip install git+https://github.com/sanderland/katrain.git


#KataGo install 
git clone https://github.com/lightvector/KataGo.git
