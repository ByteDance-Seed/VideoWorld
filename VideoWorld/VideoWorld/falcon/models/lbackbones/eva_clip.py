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
import math
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import nn

from falcon.registry import MODELS


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                         self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),  # noqa: B008
            attn_drop=0.,
            proj_drop=0.):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(
            torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(
            3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(
                self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float('-inf'))
                attn_mask = new_attn_mask
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, mlp_width)),
                         ('ln',
                          LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                         ('gelu', act_layer()),
                         ('c_proj', nn.Linear(mlp_width, d_model))]))

    def attention(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def cross_attention(self,
                        x: torch.Tensor,
                        context: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None):
        return self.attn(
            x, context, context, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class TextTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        width: int,
        layers: int,
        heads: int,
        context_length: int,
        embed_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            act_layer=act_layer,
        )
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, width), requires_grad=False)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer(
            'attn_mask', self.build_attention_mask(), persistent=False)
        self.init_parameters()

        for _, p in self.transformer.named_parameters():
            p.requires_grad = False

        for _, p in self.token_embedding.named_parameters():
            p.requires_grad = False

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        # nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        proj_std = (self.transformer.width ** -0.5) * (
                (2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_features(self, text: torch.Tensor):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        seq_len = x.size(1)
        positional_embedding = self.positional_embedding[:seq_len, :]
        x = x + positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        attn = self.attn_mask[:seq_len, :seq_len]
        x = self.transformer(x, attn_mask=attn)
        x_seq = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x_seq)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_cls = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return x_seq, x_cls

    def forward(self, x: torch.Tensor):
        x_seq, x_cls = self.forward_features(x)
        if self.text_projection is not None:
            x_cls = x_cls @ self.text_projection
        return x_seq, x_cls


@MODELS.register_module()
class EvaCLIPTextModel(BaseModule):

    def __init__(self,
                 width=768,
                 vocab_size=49408,
                 max_position_embeddings=77,
                 num_attention_heads=12,
                 layers=12,
                 projection_dim=1024,
                 gradient_checkpointing=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.text = TextTransformer(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=num_attention_heads,
            context_length=max_position_embeddings,
            embed_dim=projection_dim)

    def init_weights(self):
        super().init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        x_seq, x_cls = self.text(input_ids)
        return {
            'text_outputs': [
                x_seq,
            ],
            'proj_feature': x_cls
        }
