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
from ..autoencoders.magvit import Blur,exists,pair,LinearSpaceAttention,Residual,FeedForward,Sequential
from einops.layers.torch import Rearrange
from math import log2
from ldm.registry import MODELS


def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

class DiscriminatorBlock(Module):
    def __init__(self,
                 input_channels,
                 filters,
                 downsample=True,
                 antialiased_downsample=True):
        super().__init__()

        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None


    def forward(self,x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            x = self.downsample(x)

        x = (x + res) * (2 ** -0.5)
        return x



@MODELS.register_module()
class MGDiscriminator(Module):
    def __init__(self,
                 dim,
                 image_size,
                 set_num_layers=None,
                 channels=3,
                 max_dim=512,
                 linear_attn_dim_head=8,
                 linear_attn_heads=16,
                 ff_mult=4,
                 antialiased_downsample=False
                 ):
        super().__init__()
        image_size = pair(image_size)

        min_image_resolution = min(image_size)



        num_layers = int(log2(min_image_resolution) - 2) if set_num_layers is None else set_num_layers


        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample=is_not_last,
                antialiased_downsample=antialiased_downsample
            )

            attn_block = Sequential(
                Residual(LinearSpaceAttention(
                    dim=out_chan,
                    heads=linear_attn_heads,
                    dim_head=linear_attn_dim_head
                )),
                Residual(FeedForward(
                    dim=out_chan,
                    mult=ff_mult,
                    images=True
                ))
            )
            blocks.append(ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self,x):
        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)







