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
"""video2dataset example configs"""
import os
from omegaconf import OmegaConf

configs_path = os.path.dirname(os.path.realpath(__file__))

CONFIGS = {
    "default": OmegaConf.load(os.path.join(configs_path, "default.yaml")),
    "downsample_ml": OmegaConf.load(os.path.join(configs_path, "downsample_ml.yaml")),
    "optical_flow": OmegaConf.load(os.path.join(configs_path, "optical_flow.yaml")),
    "caption": OmegaConf.load(os.path.join(configs_path, "caption.yaml")),
}
