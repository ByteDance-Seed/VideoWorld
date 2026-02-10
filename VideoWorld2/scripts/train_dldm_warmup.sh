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
EXP=videoworld2_dldm_openx_and_videocraft_wodit_warmup
torchrun --nproc_per_node=8 --master_port=52224 -m scripts.train --config=videoworld2/configs/base/config.py -- experiment=${EXP}
python3 scripts/extract_ldm_pth.py ${EXP} iter_000100000