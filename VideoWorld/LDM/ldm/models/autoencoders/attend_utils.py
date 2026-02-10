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
from torch import nn,einsum,Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps,partial
from packaging import version
import importlib

from einops import rearrange,pack, unpack,repeat
from einops.layers.torch import Rearrange
from torch.nn import Module, ModuleList
from rotary_embedding_torch import RotaryEmbedding


class RMSNorm(Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def compact(arr):
    return [*filter(exists, arr)]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask


class Attend(nn.Module):
    def __init__(
            self,
            *,
            dropout=0.,
            causal=False,
            heads=None,
            scale=None,
            flash=False,
            onnxable=False,
            sdp_kwargs: dict = dict(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            )
    ):
        super().__init__()
        self.scale = scale

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        #flash attention
        self.flash = flash and torch.cuda.is_available()
        assert not (flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.sdp_kwargs = sdp_kwargs

    def flash_attn(self,
                   q, k, v,
                   mask = None,
                   attn_bias = None):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # manage scale, since scale is not customizable in sdp, back around it

        if exists(self.scale):
            q = q * self.scale / (q.shape[-1] ** -0.5)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        ## handle kv cache - this should be bypassable in updated flash attention 2
        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
            self,
            q, k, v,
            mask=None,
            attn_bias=None,
            prev_attn=None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle zero kv, as means for allowing network to attend to nothing

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        dots = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device=device)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = dots.softmax(dim=-1)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)

        return out


def shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = -1)


def second_taylor_expansion(x: Tensor):
    dtype, device, dim = x.dtype, x.device, x.shape[-1]

    x, ps = pack([x], '* d')

    lead_dims = x.shape[0]

    # exp(qk) = 1 + qk + (qk)^2 / 2

    x0 = x.new_ones((lead_dims,))
    x1 = x
    x2 = einsum('... i, ... j -> ... i j', x, x) * (0.5 ** 0.5)

    # concat - dimension D now becomes (1 + D + D ^2)
    # in paper, they had to heavily reduce the attention head dimension to make this work

    out, _ = pack([x0, x1, x2], 'b *')
    out, = unpack(out, ps, '* d')
    return out

Cache = namedtuple('Cache', [
    'seq_len',
    'last_token',
    'kv_cumsum',
    'k_cumsum'
])

class TaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 16,
        heads = 8,
        causal = False,
        one_headed_kv = False,
        rotary_emb = False,
        combine_heads = True,
        gate_value_heads = False,
        prenorm = False,
        shift_tokens = False,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.shift_tokens = shift_tokens
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.heads = heads
        self.dim_hidden = dim_inner

        self.causal = causal
        self.causal_linear_attn_fn = None

        if causal:
            if not exists(importlib.util.find_spec('fast_transformers')):
                print('pytorch-fast-transformers must be installed. `pip install pytorch-fast-transformers` first')
                exit()

            from fast_transformers.causal_product import CausalDotProduct
            self.causal_linear_attn_fn = CausalDotProduct.apply

        kv_heads = heads if not one_headed_kv else 1
        dim_kv_inner = dim_head * (heads if not one_headed_kv else 1)

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.one_headed_kv = one_headed_kv

        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.to_kv = nn.Sequential(
            nn.Linear(dim, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = kv_heads)
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if gate_value_heads else None

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = nn.Identity()

        if combine_heads:
            self.to_out = nn.Sequential(
                nn.Linear(dim_inner, dim, bias = False),
                nn.Dropout(dropout)
            )

    def forward(
        self,
        x,       #TensorType['batch', 'seq', 'dim', float],
        mask=None,      #Optional[TensorType['batch', 'seq', bool]] = None,
        context=None,    #Optional[TensorType['batch', 'target_seq', 'dim', float]] = None,
        eps = 1e-5,
        cache = None,
        return_cache = False
    ):
        """
        einops
        b - batch
        h - heads
        d - query / key head dimension
        e - value head dimension
        n - source query sequence length
        m - target key / value sequence length
        """
        orig_input, seq_len, is_cross_attn = x, x.shape[-2], exists(context)
        assert not (exists(self.rotary_emb) and is_cross_attn), 'rotary embedding does not work with cross attention'

        # token shift from rwkv

        if self.shift_tokens:
            if exists(cache):
                x, ps = pack([cache.last_token, x], 'b * d')

            x = shift(x)

            if exists(cache):
                _, x = unpack(x, ps, 'b * d')

        # pre rmsnorm

        normed = self.norm(x)

        # queries, keys, values

        q = self.to_q(normed)
        k, v = self.to_kv(default(context, normed))

        # maybe rotary

        if exists(self.rotary_emb):
            rotate_fn = self.rotary_emb.rotate_queries_or_keys

            if exists(cache):
                rotate_fn = partial(rotate_fn, offset = cache.seq_len)

            q, k = map(rotate_fn, (q, k))

        # scale

        q = q * self.scale

        # 2nd taylor expansion for exp(qk)

        q, k = map(second_taylor_expansion, (q, k))

        # linear attention

        if self.causal:
            assert not exists(mask), 'masking does not make sense for autoregressive linear attention'
            assert not is_cross_attn, 'causal does not make sense with cross attention'

            if self.one_headed_kv:
                k, v = map(lambda t: repeat(t, 'b 1 n d -> b h n d', h = self.heads), (k, v))

            if exists(cache):
                assert seq_len == 1
                old_seq_len, _, kv_cumsum_cache, k_cumsum_cache = cache

                kv = einsum('b h n d, b h n e -> b h d e', k, v)

                kv_cumsum = kv + kv_cumsum_cache
                k_cumsum = k + k_cumsum_cache

                num = einsum('b h n d, b h d e -> b h n e', q, kv_cumsum)
                den = einsum('... n d, ... n d -> ... n', q, k_cumsum)
                den = rearrange(den, '... -> ... 1')

                out = num / den.clamp(min = eps)

                if return_cache:
                    new_cache = Cache(old_seq_len + 1, orig_input, kv_cumsum, k_cumsum)

            else:

                num = self.causal_linear_attn_fn(q, k, v)

                k_cumsum = k.cumsum(dim = -2)
                den = einsum('... n d, ... n d -> ... n', q, k_cumsum)
                den = rearrange(den, '... -> ... 1')

                out = num / den.clamp(min = eps)

                if return_cache:
                    new_kv_cache = einsum('b h n d, b h n e -> b h d e', k, v)
                    new_k_cumsum_cache = k_cumsum[..., -1:, :]
                    new_cache = Cache(seq_len, orig_input[:, -1:], new_kv_cache, new_k_cumsum_cache)

        else:
            assert not return_cache, 'cache is only needed for autoregressive'

            if exists(mask):
                mask = rearrange(mask, 'b n -> b 1 n 1')
                k = k.masked_fill(~mask, 0.)
                v = v.masked_fill(~mask, 0.)

            if self.one_headed_kv:
                k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))

                kv = einsum('b n d, b n e -> b d e', k, v)
                qk_inv = 1. / einsum('b h n d, b m d -> b h n', q, k).clamp(min = eps)
                out = einsum('b h n d, b d e, b h n -> b h n e', q, kv, qk_inv)

            else:
                kv = einsum('b h n d, b h n e -> b h d e', k, v)
                qk_inv = 1. / einsum('b h n d, b h m d -> b h n', q, k).clamp(min = eps)
                out = einsum('b h n d, b h d e, b h n -> b h n e', q, kv, qk_inv)

        # gate value heads

        if exists(self.to_v_gates):
            out = out * self.to_v_gates(x)

        # merge heads

        out = self.merge_heads(out)

        # maybe combine heads

        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_cache