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
import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.registry import MODELS

@MODELS.register_module()
class ConvConnector(nn.Module):
    def __init__(self,in_channels=256, out_channels=256):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
    def forward(self,x):
        return self._conv(x)

