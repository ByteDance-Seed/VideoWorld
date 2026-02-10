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
from einops import rearrange

import math


class L2MultiheadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.q_weight)
        nn.init.zeros_(self.v_weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self,x):
        T,N,_ = x.shape

        q = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
                torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
                + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
                - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(
            self.head_dim
        )

        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)
        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(
            T, N, self.embed_dim
        )
        return self.out_proj(PXAV)

class SelfAttention(nn.Module):
    def __init__(self,in_channels, embed_dim=None, num_heads=1):
        super().__init__()
        embed_dim = in_channels if embed_dim is None else embed_dim
        self.use_conv = embed_dim != in_channels

        if self.use_conv:
            self.to_input = nn.Conv2d(in_channels, embed_dim, 1, bias=True)
            self.to_output = nn.Conv2d(embed_dim, in_channels, 1, bias=True)
        self.l2attn = L2MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self,input):
        attn_input = self.to_input(input) if self.use_conv else input
        batch, c, h, w = attn_input.shape
        # input: [N, C, H, W] --> [N, H, W, C] --> [N, HWC]
        attn_input = rearrange(attn_input,'n c h w-> n (h w) c')

        norm_is = self.ln1(attn_input)
        out1 = self.l2attn(norm_is) + attn_input
        norm_out1 = self.ln2(out1)
        out2 = self.ff(norm_out1.view(-1, c)).view(batch, -1, c)
        output = out2 + out1

        output = output[:, :h * w, :]
        output = output.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        output = self.to_output(output) if self.use_conv else output
        return output


class SelfAttention3D(nn.Module):
    def __init__(self,in_channels, embed_dim=None, num_heads=1):
        super().__init__()
        embed_dim = in_channels if embed_dim is None else embed_dim
        self.use_conv = embed_dim != in_channels

        if self.use_conv:
            self.to_input = nn.Conv2d(in_channels, embed_dim, 1, bias=True)
            self.to_output = nn.Conv2d(embed_dim, in_channels, 1, bias=True)
        self.l2attn = L2MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.in_channels = in_channels

    def forward(self,input):
        bs,ch,t,hi,wi = input.size()
        input = rearrange(input,'b c t h w -> b t c h w')
        input = rearrange(input, 'b t c h w -> (b t) c h w')
        attn_input = self.to_input(input) if self.use_conv else input
        batch, c, h, w = attn_input.shape
        # input: [N, C, H, W] --> [N, H, W, C] --> [N, HWC]
        attn_input = rearrange(attn_input,'n c h w-> n (h w) c')

        norm_is = self.ln1(attn_input)
        out1 = self.l2attn(norm_is) + attn_input
        norm_out1 = self.ln2(out1)
        out2 = self.ff(norm_out1.view(-1, c)).view(batch, -1, c)
        output = out2 + out1

        output = output[:, :h * w, :]
        output = output.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        output = self.to_output(output) if self.use_conv else output
        output = output.view(bs,t,self.in_channels , h,w)
        output = output.permute(0,2,1,3,4)
        return output