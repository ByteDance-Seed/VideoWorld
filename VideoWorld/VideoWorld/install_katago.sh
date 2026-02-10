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
wget http://mirrors.kernel.org/ubuntu/pool/universe/libz/libzip/libzip5_1.5.1-0ubuntu1_amd64.deb
sudo apt install ./libzip5_1.5.1-0ubuntu1_amd64.deb

sudo apt-get update
sudo apt-get -y install clinfo
sudo apt-get -y install pocl-opencl-icd

sudo apt-get -y install ocl-icd-opencl-dev
sudo apt-get -y install nvidia-opencl-dev

sudo DEBIAN_FRONTEND=noninteractive apt-get -y install libmtdev-dev
sudo apt -y install libzip-dev 

git clone https://github.com/lightvector/KataGo.git
cd KataGo/cpp
# If you get missing library errors, install the appropriate packages using your system package manager and try again.
# -DBUILD_DISTRIBUTED=1 is only needed if you want to contribute back to public training.
cmake . -DUSE_BACKEND=OPENCL -DBUILD_DISTRIBUTED=1
make -j 4