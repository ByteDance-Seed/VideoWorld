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

from typing import cast

import einops
from einops import rearrange, pack, unpack,repeat
from .l2attention import SelfAttention
import torch
import torch.nn.functional as F
from torch import nn
from ldm.registry import MODELS


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# class Attention(nn.Module):
#
#     def __init__(self, *args, num_channels: int, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self._num_channels = num_channels
#         self._group_norm = nn.GroupNorm(
#             32,
#             self._num_channels,
#             1e-6,
#         )
#         self._multihead_attention = nn.MultiheadAttention(
#             512,
#             1,
#             batch_first=True,
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self._group_norm(x)
#         x = einops.rearrange(x, 'b c h w -> b (h w) c')
#         x, _ = self._multihead_attention(x, x, x, need_weights=False)
#         x = einops.rearrange(
#             x,
#             'b (h w) c -> b c h w',
#             h=shortcut.shape[2],
#             w=shortcut.shape[3],
#         )
#         return shortcut + x


class Residual(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = nn.Sequential(
            nn.GroupNorm(
                32,
                in_channels,
                1e-6,
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.GroupNorm(
                32,
                out_channels,
                1e-6,
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
            ),
        )
        self._shortcut = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(
                in_channels,
                out_channels,
                1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shortcut(x) + self._residual(x)


class Layer(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        depth: int,
        with_attentions: bool,
        num_heads=8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        self._with_attentions = with_attentions

        residuals = nn.Sequential(
            Residual(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )
        residuals.extend(
            Residual(
                in_channels=out_channels,
                out_channels=out_channels,
            ) for _ in range(1, self._depth)
        )
        self._residuals = residuals

        if self._with_attentions:
            attentions = nn.Sequential(SelfAttention(out_channels, embed_dim=out_channels,num_heads=num_heads))
            attentions.extend(
                SelfAttention(out_channels, embed_dim=out_channels,num_heads=num_heads)
                for _ in range(1, self._depth)
            )
        else:
            attentions = nn.Sequential(nn.Identity()) * self._depth
        self._attentions = attentions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for residual, attention in zip(self._residuals, self._attentions):
            x = residual(x)
            x = attention(x)
        return x


class Downsample(nn.Module):

    def __init__(self, *args, num_channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_channels = num_channels
        self._pad = nn.ZeroPad2d((0, 1, 0, 1))
        self._conv = nn.Conv2d(
            num_channels,
            num_channels,
            3,
            2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        x = self._conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, *args, num_channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conv = nn.Conv2d(
            num_channels,
            num_channels,
            3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = F.interpolate(x.float(), scale_factor=2.0, mode='nearest')
        x = x.to(dtype)
        x = self._conv(x)
        return x


class VQGANMixin(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        width: int,
        width_mults: tuple[int, ...],
        depth_mult: int,
        attention_layer: int,
        refine_layer: int,
        resample_type: type,
        num_heads=8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._out_channels = out_channels
        self._width = width
        self._width_mults = width_mults
        self._depth_mult = depth_mult
        self._resample_type = resample_type
        self._refine_layer = refine_layer
        self._attention_layer = attention_layer

        widths = [self._width * wm for wm in self._width_mults]
        widths.insert(0, hidden_channels)

        assert 0 <= attention_layer < self.num_layers
        with_attentions = [False] * len(self._width_mults)
        with_attentions[attention_layer] = True

        assert 0 <= refine_layer <= self.num_layers
        refine_channels = widths[refine_layer]

        self._in_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            3,
            padding=1,
        )

        layers = [
            Layer(
                in_channels=ic,
                out_channels=oc,
                depth=self._depth_mult,
                with_attentions=wa,
                num_heads = num_heads,
            ) for ic, oc, wa in zip(
                widths[:-1],
                widths[1:],
                with_attentions,
            )
        ]
        self._layers = nn.Sequential(*layers)

        self._refine = nn.Sequential(
            Residual(
                in_channels=refine_channels,
                out_channels=refine_channels,
            ),
            # Attention(num_channels=refine_channels),
            SelfAttention(refine_channels, embed_dim=refine_channels, num_heads=num_heads),
            Residual(
                in_channels=refine_channels,
                out_channels=refine_channels,
            ),
        )

        resamples = [resample_type(num_channels=c) for c in widths[1:-1]]
        resamples.append(nn.Identity())
        self._resamples = nn.Sequential(*resamples)

        self._projector = nn.Sequential(
            nn.GroupNorm(
                32,
                widths[-1],
                1e-6,
            ),
            nn.SiLU(),
            nn.Conv2d(
                widths[-1],
                out_channels,
                3,
                padding=1,
            ),
        )

    @property
    def num_layers(self) -> int:
        return len(self._width_mults)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def last_parameter(self) -> nn.Parameter:
        conv = self._projector[-1]
        assert isinstance(conv, nn.Conv2d)
        return cast(nn.Parameter, conv.weight)

    def forward(self, x: torch.Tensor,code=None) -> torch.Tensor:

        x = self._in_conv(x)
        refine_countdown = self._refine_layer
        for layer, resample in zip(self._layers, self._resamples):
            if refine_countdown == 0:
                x = self._refine(x)
            refine_countdown -= 1
            x = layer(x)
            x = resample(x)
        if refine_countdown == 0:
            x = self._refine(x)
        x = self._projector(x)
        return x


@MODELS.register_module()
class VQGANEncoderL2(VQGANMixin):
    def __init__(
            self,
            *args,
            in_channels: int = 3,
            out_channels: int = 256,
            width: int = 128,
            width_mults: tuple[int, ...] = (1, 1, 2, 2, 4),
            depth_mult: int = 2,
            num_heads=8,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            in_channels=in_channels,
            hidden_channels=width,
            out_channels=out_channels,
            width=width,
            width_mults=width_mults,
            depth_mult=depth_mult,
            attention_layer=len(width_mults) - 1,
            refine_layer=len(width_mults),
            resample_type=Downsample,
            num_heads=num_heads,
            **kwargs,
        )

@MODELS.register_module()
class VQGANDecoderL2(VQGANMixin):
    def __init__(
            self,
            *args,
            in_channels: int = 256,
            out_channels: int = 3,
            width: int = 128,
            width_mults: tuple[int, ...] = (4, 2, 2, 1, 1),
            depth_mult: int = 3,
            num_heads=8,
            **kwargs,
    ) -> None:
        super().__init__(
            *args,
            in_channels=in_channels,
            hidden_channels=width * width_mults[0],
            out_channels=out_channels,
            width=width,
            width_mults=width_mults,
            depth_mult=depth_mult,
            attention_layer=0,
            refine_layer=0,
            resample_type=Upsample,
            num_heads=num_heads,
            **kwargs,
        )


