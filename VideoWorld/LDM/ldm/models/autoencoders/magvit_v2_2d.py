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
import torch.nn.functional as F

from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from typing import  Union, Tuple, Optional, List



from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from functools import wraps, partial
from ldm.registry import MODELS



def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def safe_get_index(it, ind, default = None):
    if ind < len(it):
        return it[ind]
    return default

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_odd(n):
    return not divisible_by(n, 2)

def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)







class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        self.act = None

        # if act_fn is None:
        #     self.act = None
        # else:
        #     self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = cond
        if self.act:
            emb = self.act(emb)
        emb = torch.mean(emb, dim=(2, 3), keepdim=False)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x



class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x


# strided conv downsamples



class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = Conv2dM(dim, dim_out * 4, [3,3],stride=[1,1])

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2) h w -> b  c (h p1) (w p2)', p1 = 2, p2 = 2),
            Rearrange('b c h w -> b c h w')
        )

        self.init_conv_(conv.conv)

    def init_conv_(self, conv):
        o, i,h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):

        out = self.net(x)

        return out


def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding)




class Conv2dM(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[ int, int]],
        stride=[1,1],
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        # kernel_size = cast_tuple(kernel_size, 3)

        height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)

        self.pad_mode = pad_mode
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad)

        # stride = (stride, 1, 1)
        dilation = (1, 1)
        self.conv = nn.Conv2d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode

        x = F.pad(x, self.spatial_padding, mode = pad_mode)
        return self.conv(x)


class Residual(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        kernel_size=[3,3],
        pad_mode: str = 'constant',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = nn.Sequential(
            nn.GroupNorm(32,in_channels,1e-6),
            nn.SiLU(),
            Conv2dM(in_channels, out_channels, kernel_size, pad_mode = pad_mode),
            nn.GroupNorm(32,out_channels,
                1e-6,
            ),
            nn.SiLU(),
            Conv2dM(out_channels, out_channels, kernel_size, pad_mode = pad_mode),
        )
        self._shortcut = (
            nn.Identity() if in_channels == out_channels else
            Conv2dM(in_channels, out_channels, [1,1], pad_mode = pad_mode)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shortcut(x) + self._residual(x)


class Downsample(nn.Module):

    def __init__(self, num_channels: int,with_time=False) -> None:
        super().__init__()
        self._num_channels = num_channels
        stride=[1,1]
        if with_time:
            stride = [2, 2]
        else:
            stride = [2, 2]

        self._conv = Conv2dM(
            num_channels,
            num_channels,
            kernel_size=[3,3],
            stride = stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return x


@MODELS.register_module()
class MagvitV2encoder2D(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        ('consecutive_residual', 4),
                        ('spatial_down', 1),
                        ('channel_residual', 1),
                        ('consecutive_residual', 3),
                        ('time_spatial_down', 1),
                        ('consecutive_residual', 4),
                        ('time_spatial_down', 1),
                        ('channel_residual', 1),
                        ('consecutive_residual', 3),
                        ('consecutive_residual', 4),
                    ),
                 input_conv_kernel_size: Tuple[int, int, int] = (7, 7),
                 pad_mode: str = 'constant',
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        self.conv_in = Conv2dM(channels, init_dim, input_conv_kernel_size, pad_mode=pad_mode)


        # encoder and decoder layers

        self.encoder_layers = ModuleList([])


        dim = init_dim

        self.has_cond_across_layers=[]

        for layer_def in layers:
            has_cond=False
            layer_type, *layer_params = cast_tuple(layer_def)


            if layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                encoder_layer = Sequential(
                    *[Residual(in_channels=dim, out_channels=dim) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_down':
                encoder_layer = Downsample(dim,with_time=False)

            elif layer_type == 'channel_residual':
                num_consecutive, = layer_params
                encoder_layer = Residual(in_channels=dim, out_channels=dim*2)
                dim = dim*2


            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.has_cond_across_layers.append(has_cond)


        layer_fmap_size = image_size


        # add a final norm just before quantization layer

        self.encoder_layers.append(Sequential(
            nn.GroupNorm(32, dim,1e-6),
                     nn.SiLU(),
            nn.Conv2d(dim,dim,[1,1],stride=[1,1])
        ))


        self.fmap_size = layer_fmap_size


    def encode(self, video: Tensor, cond: Optional[Tensor]=None,video_contains_first_frame=True):


        video = self.conv_in(video)


        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):
            layer_kwargs = dict()

            video = fn(video, **layer_kwargs)

        return video

    def forward(self,video_or_images: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        is_image = video_or_images.ndim == 4


        video = video_or_images
        video = video.squeeze(dim=2)


        # encoder
        x = self.encode(video, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return x, cond,video_contains_first_frame




@MODELS.register_module()
class MagvitV2Adadecoder2D(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        'residual',
                        'residual',
                        'residual'
                    ),
                 output_conv_kernel_size: Tuple[int, int, int] = ( 3, 3),
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        # whether to encode the first frame separately or not
        self.conv_out_first_frame = nn.Identity()


        self.decoder_layers = ModuleList([])


        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size
        has_cond_across_layers = []

        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            if layer_type == 'consecutive_residual':
                has_cond = False
                num_consecutive, = layer_params
                decoder_layer = Sequential(
                    *[Residual(in_channels=dim, out_channels=dim) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_up':
                has_cond = False
                decoder_layer = SpatialUpsample2x(dim)

            elif layer_type == 'channel_residual':
                has_cond = False
                num_consecutive, = layer_params
                decoder_layer = Residual(in_channels=dim* 2, out_channels=dim )
                dim = dim*2


            elif layer_type =='condation':
                has_cond = True
                decoder_layer = AdaGroupNorm(embedding_dim=init_dim*4, out_dim=dim, num_groups=32)

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            # self.decoder_layers.append(encoder_layer)

            self.decoder_layers.insert(0, decoder_layer)
            has_cond_across_layers.append(has_cond)
        self.decoder_layers.append(nn.GroupNorm(32, init_dim,1e-6),)
        self.decoder_layers.append(nn.SiLU(), )


        self.conv_out = Conv2dM(init_dim,channels,[1,1],stride=[1,1])


        self.fmap_size = layer_fmap_size

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)




    def decode(self,quantized: Tensor,cond: Optional[Tensor] = None,video_contains_first_frame = True):

        batch = quantized.shape[0]



        x = quantized

        for fn, has_cond, in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()

            if has_cond:
                layer_kwargs['cond']=quantized

            x = fn(x, **layer_kwargs)

        video = self.conv_out(x)


        return video

    def forward(self,quantized: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):

        # decode
        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)
        recon_video = recon_video.unsqueeze(dim=2)

        return recon_video
