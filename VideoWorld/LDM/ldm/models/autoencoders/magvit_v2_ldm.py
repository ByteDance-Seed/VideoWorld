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
import numpy as np
from typing import  Union, Tuple, Optional, List

from kornia.filters import filter3d
from .l2attention import SelfAttention3D

import math
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from functools import wraps, partial
from ldm.registry import MODELS

from .magvit_v2 import *
from .qformer import QFormer, QFormerMF, PositionEmbeddingRandom, QFormerMFSep




class ResidualLA(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        kernel_size=[3,3,3],
        pad_mode: str = 'constant',
        alpha=4,
        frame_num=5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = nn.Sequential(
            nn.GroupNorm(alpha*frame_num,in_channels*frame_num,1e-6),
            nn.SiLU(),
            CausalConv3d(in_channels, out_channels, kernel_size, pad_mode = pad_mode),
            nn.GroupNorm(alpha*frame_num,out_channels*frame_num,
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
        # x: B, C, T, H, W
        # import pdb;pdb.set_trace()
        shortcut = self._shortcut(x)
        for layer in self._residual:
            if isinstance(layer, nn.GroupNorm):
                b, c, t, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4)
                x = x.flatten(1, 2)
                x = layer(x)
                x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            else:
                x = layer(x)
        return shortcut + x
    

@MODELS.register_module()
class MagvitV2LAencoder(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 pre_out_layer=-1,
                 use_la_norm=False,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        ('consecutive_/opt/tiger/mmagicinit/test_la', 4),
                        ('spatial_down', 1),
                        ('channel_/opt/tiger/mmagicinit/test_la', 1),
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
                 separate_first_frame_encoding=False,
                 frame_num=4,
                 act_embedding_num=2,
                 sep_qformer=False,
                 time_padding=3
                 
                 
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.sep_qformer = sep_qformer
        self.frame_num = frame_num
        # initial encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode=pad_mode)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        # encoder and decoder layers

        self.encoder_layers = ModuleList([])

        dim = init_dim

        time_downsample_factor = 1
        self.has_cond_across_layers=[]
        # norm_type = ResidualLA if use_la_norm else Residual
        norm_type=ResidualLA
        for layer_def in layers:
            has_cond=False
            layer_type, *layer_params = cast_tuple(layer_def)


            if layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                encoder_layer = Sequential(
                    *[norm_type(in_channels=dim, out_channels=dim, frame_num=frame_num+time_padding) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_down':
                encoder_layer = Downsample(dim,with_time=False)

            elif layer_type == 'channel_residual':
                num_consecutive, = layer_params
                encoder_layer = norm_type(in_channels=dim, out_channels=dim*2, frame_num=frame_num+time_padding)
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
        self.time_padding = time_downsample_factor - 1 if time_padding == 0 else time_padding

        self.fmap_size = layer_fmap_size

        self.time_padding = time_padding 
        self.pre_out_layer = pre_out_layer
    
        
        self.pos_embedding = PositionEmbeddingRandom(256)
        self.act = nn.Embedding(act_embedding_num, 512)
        if self.sep_qformer:
            self.qformer = QFormerMFSep(2, 512, 4, 512, qformer_num=act_embedding_num, time_padding=time_padding)
        else:
            self.qformer = QFormerMF(2, 512, 4, 512)
        # import pdb;pdb.set_trace()
    def encode(self, video: Tensor, cond: Optional[Tensor]=None,video_contains_first_frame=True):
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame
        # import pdb;pdb.set_trace()
        
        # whether to pad video or not
    
        if video_contains_first_frame:
            video_len = video.shape[2]

            video = pad_at_dim(video, (self.time_padding, 0), value=0., dim=2) #B, 3, T+time_padding, H, W, pad from left

            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video) #stride 1, [B, C=128, T+3, 128,128]

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], 'b c * h w')
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        # encoder layers
        pre_encode_out = None
        for idx, (fn, has_cond) in enumerate(zip(self.encoder_layers, self.has_cond_across_layers)):
            layer_kwargs = dict()

            video = fn(video, **layer_kwargs)
            if idx == self.pre_out_layer:
                pre_encode_out = video[:, :, :(self.time_padding + 1)]
                
        video = video[:, :, self.time_padding:]
        video_length = video.shape[2]
        dense_pe = torch.stack([self.pos_embedding((16, 16)) for i in range(video_length)]).permute(1, 0, 2, 3)[None]
        dense_pe = torch.repeat_interleave(dense_pe, video.shape[0], dim=0)
        query = torch.repeat_interleave(self.act.weight[None], video.shape[0], dim=0)
        video, _ = self.qformer(video, dense_pe, query)
        video = video.permute(0, 2, 1)[:, :, :, None, None]
     
        return video, pre_encode_out

    def forward(self,video_or_images: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):
        # import pdb;pdb.set_trace()
        assert video_or_images.ndim in {4, 5} #B, C, T, H, W

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
        x, pre_encode_out = self.encode(video, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return x, cond, video_contains_first_frame, pre_encode_out



@MODELS.register_module()
class MagvitV2LAAdadecoder(Module):
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
                 separate_first_frame_encoding=False,
                 use_pre_video=True,
                 use_pre_encode=False,
                 noframe1code=False,
                 add_code=False,
                 time_padding=3
                 
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.noframe1code = noframe1code
        self.add_code = add_code
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
        # import pdb;pdb.set_trace()
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
        self.time_padding = time_downsample_factor - 1 if time_padding == 0 else time_padding

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)

       
        self.use_pre_video = use_pre_video
        self.use_pre_encode = use_pre_encode
        if self.use_pre_video:
            input_conv_kernel_size = (9, 9, 9)
            pad_mode = 'constant'
            self.pre_layers = nn.Sequential(*[CausalConv3d(3, 512, input_conv_kernel_size, pad_mode=pad_mode, stride=(1, 8, 8)),
                                # Downsample(128, with_time=False),
                                # ResidualLA(in_channels=128, out_channels=256),
                                # Downsample(256, with_time=False),
                                # ResidualLA(in_channels=256, out_channels=512),
                                # Downsample(512, with_time=False),
                                # ResidualLA(in_channels=512, out_channels=512),
                                # nn.GroupNorm(32, dim,1e-6),
                                # nn.SiLU(),
                            ])
        
    def last_parameter(self):
        # conv = self.conv_out.conv.weight
        # assert isinstance(conv, nn.Conv2d)
        return self.conv_out.conv.weight

    def decode(self,quantized: Tensor,cond: Optional[Tensor] = None,video_contains_first_frame = True, video_or_images=None, pre_encode_out=None):
        # import pdb;pdb.set_trace()
        quantized = torch.ones_like(quantized) if self.noframe1code else quantized
        quantized = torch.repeat_interleave(quantized, pre_encode_out.shape[-2], dim=-2)
        quantized = torch.repeat_interleave(quantized, pre_encode_out.shape[-2], dim=-1)
        quantized = pre_encode_out + quantized if self.add_code else torch.cat((pre_encode_out, quantized), dim=2) 
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
                video_contains_first_frame = True,
                video_or_images=None,
                pre_encode_out=None):

        # decode
        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame, video_or_images=video_or_images, pre_encode_out=pre_encode_out)

        return recon_video
