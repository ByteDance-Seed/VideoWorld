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
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad
from typing import  Union, Tuple, Optional, List

from .attend_utils import Attend,RMSNorm,TaylorSeriesLinearAttn
from kornia.filters import filter3d
# from gateloop_transformer import SimpleGateLoopLayer

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from functools import wraps, partial
from falcon.registry import MODELS



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
        emb = torch.mean(emb, dim=(2, 3, 4), keepdim=False)
        emb = self.linear(emb)
        emb = emb[:, :, None, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x



class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x



class Blur(Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum('i, j -> i j', f, f)
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            f = einsum('i, j, k -> i j k', f, f, f)
            f = rearrange(f, '... -> 1 ...')

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        out = filter3d(x, f, normalized = True)

        if is_images:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out



class Conv3DMod(Module):
    def __init__(
            self,
            dim,
            *,
            spatial_kernel,
            time_kernel,
            causal=True,
            dim_out=None,
            demod=True,
            eps=1e-8,
            pad_mode='zeros'
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.eps = eps

        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel

        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod

        nn.init.kaiming_normal_(self.weights, a=0, mode='fan_in', nonlinearity='selu')

    def forward(
            self,
            fmap,
            cond: Tensor
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b = fmap.shape[0]

        # prepare weights for modulation

        weights = self.weights

        # do the modulation, demodulation, as done in stylegan2

        cond = rearrange(cond, 'b i -> b 1 i 1 1 1')

        weights = weights * (cond + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k0 k1 k2 -> b o 1 1 1 1', 'sum').clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t h w -> 1 (b c) t h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')
        fmap = F.pad(fmap, self.padding, mode=self.pad_mode)
        fmap = F.conv3d(fmap, weights, groups=b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)


# strided conv downsamples
class SpatialDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    def forward(self, x):
        x = self.maybe_blur(x, space_only = True)

        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.conv(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

class TimeDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride = 2)

    def forward(self, x):
        x = self.maybe_blur(x, time_only = True)

        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out


class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = CausalConv3d(dim, dim_out * 4, [3,3,3],stride=[1,1,1])

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b c t h w -> b t c h w'),
            Rearrange('b t (c p1 p2) h w -> b t c (h p1) (w p2)', p1 = 2, p2 = 2),
            Rearrange('b t c h w -> b c t h w')
        )

        self.init_conv_(conv.conv)

    def init_conv_(self, conv):
        o, i,t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i,t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):

        out = self.net(x)

        return out

class TimeSpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = CausalConv3d(dim, dim_out * 8, [3,3,3],stride=[1,1,1])

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2 p3 ) t h w -> b c (t p1) (h p2) (w p3)', p1 = 2, p2 = 2, p3=2),
        )

        self.init_conv_(conv.conv)

    def init_conv_(self, conv):
        o, i,t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 8, i, t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 8) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):

        out = self.net(x)

        return out


class TimeUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p) t -> b c (t p)', p = 2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        out = self.net(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out


def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding)


# class CausalConv3d(Module):
#     def __init__(
#         self,
#         chan_in,
#         chan_out,
#         kernel_size: Union[int, Tuple[int, int, int]],
#         pad_mode = 'constant',
#         **kwargs
#     ):
#         super().__init__()
#         kernel_size = cast_tuple(kernel_size, 3)
#
#         time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
#
#         assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
#
#         dilation = kwargs.pop('dilation', 1)
#         stride = kwargs.pop('stride', 1)
#
#         self.pad_mode = pad_mode
#         time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
#         height_pad = height_kernel_size // 2
#         width_pad = width_kernel_size // 2
#
#         self.time_pad = time_pad
#         self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
#
#         stride = (stride, 1, 1)
#         dilation = (dilation, 1, 1)
#         self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)
#
#     def forward(self, x):
#         pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'
#
#         x = F.pad(x, self.time_causal_padding, mode = pad_mode)
#         return self.conv(x)


class CausalConv3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride=[1,1,1],
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        # kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride_time = stride[0]

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride_time)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        # stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)


class Residual(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        kernel_size=[3,3,3],
        pad_mode: str = 'constant',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = nn.Sequential(
            nn.GroupNorm(32,in_channels,1e-6),
            nn.SiLU(),
            CausalConv3d(in_channels, out_channels, kernel_size, pad_mode = pad_mode),
            nn.GroupNorm(32,out_channels,
                1e-6,
            ),
            nn.SiLU(),
            CausalConv3d(out_channels, out_channels, kernel_size, pad_mode = pad_mode),
        )
        self._shortcut = (
            nn.Identity() if in_channels == out_channels else
            CausalConv3d(in_channels, out_channels, [1,1,1], pad_mode = pad_mode)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut = self._shortcut(x) 
        # import pdb;pdb.set_trace()
        # x = self._residual[0](x) #GroupNorm
        # x = self._residual[1](x)
        # x = self._residual[2](x) #CausalConv3d
        # x = self._residual[3](x)
        # x = self._residual[4](x) #GroupNorm
        # x = self._residual[5](x) #CausalConv3d
        # return shortcut + x
        return self._shortcut(x) + self._residual(x)


class Downsample(nn.Module):

    def __init__(self, num_channels: int,with_time=False) -> None:
        super().__init__()
        self._num_channels = num_channels
        stride=[1,1,1]
        if with_time:
            stride = [2, 2, 2]
        else:
            stride = [1, 2, 2]

        self._conv = CausalConv3d(
            num_channels,
            num_channels,
            kernel_size=[3,3,3],
            stride = stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return x



class CausalConvTranspose3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size=[3,3,3],
        stride=[1,1,1],
        **kwargs
    ):
        super().__init__()
        # kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        time_stride=stride[0]


        self.upsample_factor = time_stride

        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        padding = (0, height_pad, width_pad)

        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding = padding, **kwargs)

    def forward(self, x):
        assert x.ndim == 5
        t = x.shape[2]

        out = self.conv(x)

        out = out[..., :(t * self.upsample_factor), :, :]
        return out
class Upsample(nn.Module):

    def __init__(self, num_channels: int,with_time=False) -> None:
        super().__init__()
        self._num_channels = num_channels
        stride=[1,1,1]
        if with_time:
            stride = [2, 2, 2]
        else:
            stride = [1, 2, 2]

        self._conv = CausalConvTranspose3d(
            num_channels,
            num_channels,
            kernel_size=[3,3,3],
            stride = stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return x



@MODELS.register_module()
class MagvitV2encoder(Module):
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
                 input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
                 output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 pad_mode: str = 'constant',
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode=pad_mode)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        # encoder and decoder layers

        self.encoder_layers = ModuleList([])

        self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode=pad_mode)

        dim = init_dim

        time_downsample_factor = 1
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

            elif layer_type == 'time_spatial_down':
                encoder_layer = Downsample(dim, with_time=True)

                time_downsample_factor *= 2

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.has_cond_across_layers.append(has_cond)


        layer_fmap_size = image_size


        # add a final norm just before quantization layer

        self.encoder_layers.append(Sequential(
            nn.GroupNorm(32, dim,1e-6),
                     nn.SiLU(),
            nn.Conv3d(dim,dim,[1,1,1],stride=[1,1,1])
        ))

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size


    def encode(self, video: Tensor, cond: Optional[Tensor]=None,video_contains_first_frame=True):
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # whether to pad video or not

        if video_contains_first_frame:
            video_len = video.shape[2]

            video = pad_at_dim(video, (self.time_padding, 0), value=0., dim=2)

            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)
        # import pdb;pdb.set_trace()
        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], 'b c * h w')
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        # encoder layers

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

        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True

        else:
            video = video_or_images

        batch, channels, frames = video.shape[:3]

        assert divisible_by(frames - int(video_contains_first_frame),
                            self.time_downsample_factor), f'number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}'

        # encoder
        x = self.encode(video, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return x, cond,video_contains_first_frame


@MODELS.register_module()
class MagvitV2decoder(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        'residual',
                        'residual',
                        'residual'
                    ),
                 output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        # whether to encode the first frame separately or not
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        self.decoder_layers = ModuleList([])


        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size
        time_downsample_factor = 1
        has_cond_across_layers = []

        for layer_def in layers:
            has_cond=False
            layer_type, *layer_params = cast_tuple(layer_def)

            if layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                decoder_layer = Sequential(
                    *[Residual(in_channels=dim, out_channels=dim) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_up':
                decoder_layer = SpatialUpsample2x(dim)

            elif layer_type == 'channel_residual':
                num_consecutive, = layer_params
                decoder_layer = Residual(in_channels=dim* 2, out_channels=dim )
                dim = dim*2


            elif layer_type == 'time_spatial_up':
                decoder_layer = TimeSpatialUpsample2x(dim)

                time_downsample_factor *= 2

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            # self.decoder_layers.append(encoder_layer)

            self.decoder_layers.insert(0, decoder_layer)
            has_cond_across_layers.append(has_cond)
        self.decoder_layers.append(nn.GroupNorm(32, init_dim,1e-6),)
        self.decoder_layers.append(nn.SiLU(), )


        # self.conv_out = Sequential(
        #     nn.GroupNorm(32, init_dim,1e-6),
        #              nn.SiLU(),
        #     nn.Conv3d(init_dim,channels,[3,3,3],stride=[1,1,1]))
        self.conv_out = CausalConv3d(init_dim,channels,[1,1,1],stride=[1,1,1])

        # self.conv_in = CausalConv3d(512, 512, [3, 3, 3], stride=[1, 1, 1])


        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)




    def decode(self,quantized: Tensor,cond: Optional[Tensor] = None,video_contains_first_frame = True):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        #conditioning if needed



        x = quantized
        # x = self.conv_in(x)

        for fn, has_cond, in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()


            x = fn(x, **layer_kwargs)

        # to pixels
        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :,
                                                                                        (self.time_padding + 1):]
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            video, _ = pack([outff, out], 'b c * h w')
        else:
            video = self.conv_out(x)

            # if video were padded, remove padding
            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    def forward(self,quantized: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):

        # decode
        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return recon_video



@MODELS.register_module()
class MagvitV2Adadecoder(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        'residual',
                        'residual',
                        'residual'
                    ),
                 output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        # whether to encode the first frame separately or not
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        self.decoder_layers = ModuleList([])


        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size
        time_downsample_factor = 1
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


            elif layer_type == 'time_spatial_up':
                has_cond = False
                decoder_layer = TimeSpatialUpsample2x(dim)

                time_downsample_factor *= 2

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


        # self.conv_out = Sequential(
        #     nn.GroupNorm(32, init_dim,1e-6),
        #              nn.SiLU(),
        #     nn.Conv3d(init_dim,channels,[3,3,3],stride=[1,1,1]))
        self.conv_out = CausalConv3d(init_dim,channels,[1,1,1],stride=[1,1,1])

        # self.conv_in = CausalConv3d(512, 512, [3, 3, 3], stride=[1, 1, 1])


        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)




    def decode(self,quantized: Tensor,cond: Optional[Tensor] = None,video_contains_first_frame = True):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        #conditioning if needed



        x = quantized
        # x = self.conv_in(x)

        for fn, has_cond, in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()

            if has_cond:
                layer_kwargs['cond']=quantized

            x = fn(x, **layer_kwargs)

        # to pixels
        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :,
                                                                                        (self.time_padding + 1):]
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            video, _ = pack([outff, out], 'b c * h w')
        else:
            video = self.conv_out(x)

            # if video were padded, remove padding
            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    def forward(self,quantized: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):

        # decode
        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return recon_video







