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
from typing import Literal, Tuple
import torch
import torch.nn as nn
from math import pi
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from itertools import pairwise
from einops import einsum, rearrange, repeat
from einops import pack, unpack
from einops.layers.torch import Rearrange
from .magvit_v2 import CausalConv3d


def exists(var):
    return var is not None

def default(var, val):
    return var if exists(var) else val


class Downsample(nn.Module):
    def __init__(
        self,
        time_factor : int = 1,
        space_factor : int = 1,
    ) -> None:
        super().__init__()
        
        self.time_factor = time_factor
        self.space_factor = space_factor
        
        self.go_down = None
    
    @property
    def factor(self) -> int:
        return self.time_factor * (self.space_factor ** 2)
    
    def forward(
        self,
        inp : Tensor,
        **kwargs,
    ) -> Tensor:
        return self.go_down(inp)
    

class SpaceTimeDownsample(Downsample):
    '''Space-Time Downsample module.
    '''
    
    def __init__(
        self,
        in_channels : int,
        kernel_size : int,
        out_channels : int,
        time_factor : int = 2,
        space_factor : int = 2,
        **kwargs
    ) -> None:
        super().__init__(
            time_factor=1 / time_factor,
            space_factor=1 / space_factor,
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        self.go_down = CausalConv3d(
            in_channels,
            default(out_channels, in_channels),
            kernel_size = kernel_size,
            stride = (time_factor, space_factor, space_factor),
            **kwargs,
        )


class ForwardBlock(nn.Module):
    
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        hid_dim : int = 256,
        block  : nn.Module = nn.Linear,
        act_fn : nn.Module = nn.GELU,
        num_groups : int = 1,
        last_act : bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        out_dim = default(out_dim, in_dim)
        if isinstance(hid_dim, int): hid_dim = (hid_dim,)
        hid_dim = default(hid_dim, ())
        
        dims = (in_dim,) + hid_dim + (out_dim,)
        
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_dim),
            *[nn.Sequential(
                block(inp_dim, out_dim, **kwargs),
                act_fn() if l < len(dims) - 2 or last_act else nn.Identity()
            ) for l, (inp_dim, out_dim) in enumerate(pairwise(dims))],
        )
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        return self.net(inp)
    

# Adapted from lucidrains/rotary-embedding-torch at:
# https://github.com/lucidrains/rotary-embedding-torch/
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim : int,
        kind: Literal['1d', '2d', 'const'] = '1d',
        theta = 10000,
        max_freq = 10,
        num_freq = 1,
        learned_freq = False,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
    ) -> None:
        super().__init__()
        
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        if kind == '1d':
            freq = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif kind == '2d':
            freq = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif kind == 'const':
            freq = torch.ones(num_freq).float()
                
        self.freq = nn.Parameter(freq, requires_grad=learned_freq)
        
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor
        
        self.default_seq_dim = -2
        
    def forward(
        self,
        seq : Tensor,
        seq_dim : int = None,
        offset = 0,
    ) -> Tensor:
        seq_dim = default(seq_dim, self.default_seq_dim)
        seq_len = seq.shape[seq_dim]
        
        freq = self.freq
        
        # Get sequence position
        pos = (torch.arange(seq_len, device=freq.device) + offset) / self.interpolate_factor
        
        freq = einsum(pos, freq, '..., f -> ... f')
        freq = repeat(freq, '... n -> ... (n r)', r = 2)
        
        if seq_dim == -3: freq = rearrange(freq, 'n d -> n 1 d')
        
        # Apply rotary embedding
        return self.apply(freq, seq, seq_dim = seq_dim)
        
    def apply(
        self,
        freq : Tensor,
        seq  : Tensor,
        start_index : int = 0,
        scale : float = 1.,
        seq_dim : int = -2
    ) -> Tensor:
        dtype = seq.dtype

        if seq.ndim == 3:
            seq_len = seq.shape[seq_dim]
            freq = freq[-seq_len:]

        rot_dim = freq.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= seq.shape[-1], f'feature dimension {seq.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        t_left, seq, t_right = seq[..., :start_index], seq[..., start_index:end_index], seq[..., end_index:]
        
        seq = (seq * freq.cos() * scale) + (self.rotate_half(seq) * freq.sin() * scale)
        out = torch.cat((t_left, seq, t_right), dim = -1)
        
        return out.type(dtype)
    
    def rotate_half(self, inp : Tensor) -> Tensor:
        inp = rearrange(inp, '... (d r) -> ... d r', r = 2)
        x1, x2 = inp.unbind(dim = -1)
        inp = torch.stack((-x2, x1), dim = -1)
        return rearrange(inp, '... d r -> ... (d r)')
    
    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

class Adapter(nn.Module):
    def __init__(
        self,
        qry_dim : int,
        n_head : int,
        d_head : int,        
        key_dim : int = None,
        val_dim : int = None,
        block : nn.Module | Tuple[nn.Module, ...] = nn.Linear,
        qry_kwargs : dict = {},
        key_kwargs : dict = {},
        val_kwargs : dict = {},
        bias : bool = False,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        if issubclass(block, nn.Module):
            block = (block, block, block)
        
        self.to_q = block[0](qry_dim, n_head * d_head, bias=bias, **qry_kwargs) if qry_dim != n_head * d_head else nn.Identity()
        self.to_k = block[1](key_dim, n_head * d_head, bias=bias, **key_kwargs) if key_dim != n_head * d_head else nn.Identity()
        self.to_v = block[2](val_dim, n_head * d_head, bias=bias, **val_kwargs) if val_dim != n_head * d_head else nn.Identity()
        
        self.n_head = n_head
        
    def forward(
        self,
        qry : Tensor,
        key : Tensor  = None,
        val : Tensor  = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        key = default(key, qry)
        val = default(val, key)
        
        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)
        
        qkv, ps = pack([q, k, v], '* n d')
        qkv = rearrange(qkv, 'qkv n (h d) -> qkv h n d', h=self.n_head)
                
        return unpack(qkv, ps, '* n h d')

# Inspired by the two cool repos implementations at:
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
# https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/magvit2_pytorch.py#L255
class Attention(nn.Module):
    '''
    Standard self-attention module as originally introduced
    in the paper "Attention is All You Need". This module
    uses the flash-attention implementation offered by
    PyTorch >= 2.0.
    '''
    
    def __init__(
        self,
        n_head : int,
        d_head : int,
        d_inp : int  = None,
        d_out : int  = None,
        bias : bool = False,
        scale : float  = None,
        causal : bool = False,
        dropout : float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.d_inp = default(d_inp, n_head * d_head)
        self.d_out = default(d_out, self.d_inp)
        
        self.norm = nn.LayerNorm(n_head * d_head)
        self.embed = nn.Identity()
        
        self.to_qkv = Adapter(
            qry_dim=self.d_inp,
            n_head=n_head,
            d_head=d_head,
            bias=bias,
            **kwargs,
        )
        
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(n_head * d_head, self.d_out, bias=bias) if self.d_out != n_head * d_head else nn.Identity(),
        )
        
        self.scale = default(scale, n_head * d_head ** -0.5)
        self.causal = causal
        self.dropout = dropout
        
    def forward(
        self,
        qry : Tensor,
        key : Tensor  = None,
        val : Tensor  = None,
        mask : Tensor  = None,
    ) -> Tensor:
        '''
        Apply self-attention mechanism to the input sequence.

        Args:
            qry (Tensor): Input sequence tensor of shape (batch_size, sequence_length, embedding_size).
            mask (Tensor, optional): Mask tensor of shape (batch_size, sequence_length) indicating which
                elements in the sequence should be masked. Defaults to None.

        Returns:
            Tensor: Output tensor after applying self-attention mechanism of shape
                (batch_size, sequence_length, embedding_size).
        '''
        
        qry = self.embed(qry)
        qry = self.norm(qry)
        
        key = default(key, qry)
        val = default(val, key)
        
        # Project the input sequence into query, key, and value
        q, k, v = self.to_qkv(qry, key, val)
        
        # Compute the self-attention using fast flash-attention
        attn = scaled_dot_product_attention(q, k, v,
            attn_mask=mask,
            is_causal=self.causal,
            dropout_p=self.dropout,
            scale=self.scale,
        )
        
        # Project the output back to the original embedding dimension
        out = self.to_out(attn)
        
        return out

class SpatialAttention(Attention):
    '''
    Attention module that applies self-attention across the
    spatial dimensions of the input tensor, expected to be
    either an image (4D tensor) or a video (5D tensor).
    '''
    
    def __init__(
        self,
        n_head : int,
        d_head : int,
        d_inp : int  = None,
        d_out : int  = None,
        bias : bool = False,
        embed : bool = True,
        scale : float  = None,
        causal : bool = False,
        dropout : float = 0.0,
        transpose : bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            n_head,
            d_head,
            d_inp,
            d_out,
            bias,
            scale,
            causal,
            dropout,
            **kwargs,
        )
        
        # Use 2d-rotary embedding for spatial attention
        self.embed = RotaryEmbedding(self.d_inp, kind='2d') if embed else nn.Identity()
        
        self.transpose = transpose
    
    def forward(
        self,
        video : Tensor,
        cond : Tensor  = None,
        mask: Tensor  = None,
        transpose: bool  = None,
    ) -> Tensor:
        transpose = default(transpose, self.transpose)
        
        pattern = 'b c ... h w' if transpose else 'b ... h w c'
        inp = rearrange(video, f'{pattern} -> b ... h w c')
        b, *t, h, w, c = video.shape
        
        inp, t_ps = pack([inp], '* h w c')        
        inp, s_ps = pack([inp], 'b * c')
        
        # We expect the condition to be space-wise, i.e. of shape (batch, h * w, feat)
        cond = repeat(cond, 'b hw c -> (b t) hw c', t=t if exists(t) else 1) if exists(cond) else None
        
        out = super().forward(
            inp,
            key=cond,
            mask = mask,
        )
        
        out = unpack(out, s_ps, 'b * c')[0]
        out = unpack(out, t_ps, '* h w c')[0]
        
        return rearrange(out, f'b ... h w c -> {pattern}')

class TemporalAttention(Attention):
    '''
    Attention module that applies self-attention across the
    temporal dimension of the input tensor, expected to be
    a 5D tensor of shape (batch, feat, time, height, width).
    '''
    
    def __init__(
        self,
        n_head : int,
        d_head : int,
        d_inp : int  = None,
        d_out : int  = None,
        bias : bool = False,
        embed : bool = True,
        scale : float  = None,
        causal : bool = False,
        dropout : float = 0.0,
        transpose : bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            n_head,
            d_head,
            d_inp,
            d_out,
            bias,
            scale,
            causal,
            dropout,
            **kwargs,
        )
        
        # Use 1d-rotary embedding for temporal attention
        self.embed = RotaryEmbedding(self.d_inp, kind='1d') if embed else nn.Identity()
        
        self.transpose = transpose
    
    def forward(
        self,
        video : Tensor,
        cond : Tensor  = None,
        mask : Tensor  = None,
        transpose : bool  = None,
    ) -> Tensor:
        transpose = default(transpose, self.transpose)
        
        pattern = 'b c t h w' if transpose else 'b t h w c'
        inp = rearrange(video, f'{pattern} -> b h w t c')
        b, h, w, *_ = inp.shape
        inp, ps = pack([inp], '* t c')
        
        # We expect the condition to be time-wise, i.e. of shape (batch, time, feat)
        cond = repeat(cond, 'b t c -> (b h w) t c', h=h, w=w) if exists(cond) else None
        
        out = super().forward(
            inp,
            key=cond,
            mask=mask,
        )
        
        out = unpack(out, ps, '* t c')[0]
        return rearrange(out, f'b h w t c -> {pattern}')
    
class SpaceTimeAttention(nn.Module):
    
    def __init__(
        self,
        n_head : int ,
        d_head : int ,
        d_inp : int  = None,
        d_out : int  = None,
        hid_dim : int   = None,
        bias : bool = False,
        embed : bool  = True,
        scale : float  = None,
        dropout : float = 0.0,
        kernel_size : int = 3,
        transpose : bool = False,
        time_attn_kw : dict = {},
        space_attn_kw : dict = {},
    ) -> None:
        super().__init__()
        
        if isinstance(n_head, int):
            n_head = (n_head, n_head)
        if isinstance(d_head, int):
            d_head = (d_head, d_head)
        if isinstance(embed, bool):
            embed = (embed, embed)
        
        self.space_attn = SpatialAttention(
            n_head=n_head[0],
            d_head=d_head[0],
            d_inp=d_inp,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[0],
            causal=False,
            dropout=dropout,
            transpose=transpose,
            **space_attn_kw,
        )
        
        self.temp_attn = TemporalAttention(
            n_head=n_head[1],
            d_head=d_head[1],
            d_inp=None,
            d_out=None,
            bias=bias,
            scale=scale,
            embed=embed[1],
            # * Causal attention for temporal attention
            causal=True, 
            dropout=dropout,
            transpose=transpose,
            **time_attn_kw,
        )
        
        self.ffn = ForwardBlock(
            n_head[1] * d_head[1],
            out_dim=d_out,
            hid_dim=hid_dim,
            num_groups=n_head[1],
            bias=bias,
            block=nn.Conv3d,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        
        pattern = 'b c t h w' if transpose else 'b t h w c'
        self.ffn = nn.Sequential(
            Rearrange(f'{pattern} -> b c t h w'),
            self.ffn,
            Rearrange(f'b c t h w -> {pattern}'),
        )
        
        self.in_channels  = default(d_inp, n_head[0] * d_head[0])
        self.out_channels = default(d_out, n_head[1] * d_head[1])
        
        space_hid = d_head[0] * n_head[0]
        time_hid  = d_head[1] * n_head[1]
        self.time_skip  = nn.Identity() # Don't need this at the moment
        self.space_skip = nn.Conv3d(d_inp, space_hid, 1) if exists(d_inp) and d_inp != space_hid else nn.Identity()
        self.ffn_skip   = nn.Conv3d(time_hid,  d_out, 1) if exists(d_out) and time_hid != d_out  else nn.Identity()
        
    def forward(
        self,
        video : Tensor,
        cond  : Tuple[Tensor, Tensor] | Tensor  = None,
        mask  : Tensor  = None,
    ) -> Tensor:
        if not isinstance(cond, tuple):
            cond = (cond, cond)
        
        space_cond, time_cond = cond
        
        # We feed the video first through the spatial attention
        # and then through the temporal attention mechanism.
        # NOTE: Positional embeddings are added within the attention
        video = self.space_attn(video, cond=space_cond, mask=mask) + self.space_skip(video)
        video = self.temp_attn (video, cond=time_cond , mask=mask) + self.time_skip (video)
        video = self.ffn(video) + self.ffn_skip(video)
        
        return video