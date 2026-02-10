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
import torch.nn as nn
from torch.nn import Module, ModuleList
from ..autoencoders.magvit import Sequential
from ..autoencoders.magvit_v2 import pad_at_dim
from einops.layers.torch import Rearrange
from .magvit_2_utils import BlurPoolND
from math import log2
from ldm.models.autoencoders.magvit_v2 import CausalConv3d
from ldm.registry import MODELS


def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

class DiscriminatorBlock(Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 padding=False):
        super().__init__()


        self.net_1 = nn.Sequential(
            CausalConv3d(input_channels, output_channels, [3,3,3],stride=[1,1,1]),
            leaky_relu(),
        )
        self.blur_res = BlurPoolND(input_channels,dims=3)

        self.blur_dir = BlurPoolND(output_channels,dims=3)

        self.net_2 = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w', p1=2, p2=2, p3=2),
            CausalConv3d(output_channels*8, output_channels, [1, 1, 1], [1, 1, 1])
        )
        self.net_3 = nn.Sequential(
            CausalConv3d(output_channels, output_channels, [3, 3, 3], stride=[1, 1, 1]),
            leaky_relu()
        )

        self.res = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w', p1=2, p2=2, p3=2),
            CausalConv3d(input_channels*8, output_channels, [1, 1, 1], [1, 1, 1])
        )
        self.pad = padding




    def forward(self,x):
        if self.pad:
            x = pad_at_dim(x, (1, 0), dim=2, value=0)
        # the 3D blur operation need to update, which cost many times
        # res = self.res(self.blur_res(x))
        res = self.res(x)

        x = self.net_1(x)
        # x = self.blur_dir(x)
        x = self.net_2(x)
        x = self.net_3(x)
        x = (x + res) * (2 ** -0.5)
        return x



@MODELS.register_module()
class MGV2Discriminator(Module):
    def __init__(self,
                 init_dim=128,
                 ):
        super().__init__()

        self.net_1 = nn.Sequential(
            CausalConv3d(3, init_dim, [3, 3, 3], stride=[1, 1, 1]),
            leaky_relu(),
        )

        self.res_net = nn.Sequential(
            DiscriminatorBlock(init_dim,init_dim*2,padding=True), # 18->9
            DiscriminatorBlock(init_dim*2,init_dim*4,padding=True), # 10->5
            DiscriminatorBlock(init_dim * 4, init_dim * 4,padding=True), # 6->3
            DiscriminatorBlock(init_dim * 4, init_dim * 4,padding=True), # 4->2
            DiscriminatorBlock(init_dim * 4, init_dim * 4),
        )

        self.net_3 = nn.Sequential(
            CausalConv3d(init_dim * 4, init_dim * 4, [3, 3, 3], stride=[1, 1, 1]),
            leaky_relu(),
        )


        self.to_logits = Sequential(
            Rearrange('b ... -> b (...)'),
            nn.Linear(init_dim *4* 16, init_dim *4),
            leaky_relu(),
            nn.Linear(init_dim * 4, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self,x):
        x = self.net_1(x)
        x = self.res_net(x)
        x = self.net_3(x)
        return self.to_logits(x)







