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
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import inspect
import math
from typing import Callable

import torch
from mmengine.model import BaseModule
from packaging import version
from torch import Tensor, nn


class NewGELUActivation(nn.Module):
    """Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).

    Also see the Gaussian Error Linear Units paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """Original Implementation of the GELU activation function in Google BERT
    repo when initially created. For.

    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(version.parse(torch.__version__).base_version
                         ) < version.parse('1.4') or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """Applies GELU approximation that is slower than QuickGELU but more
    accurate.

    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 +
                              torch.tanh(input * 0.7978845608 *
                                         (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """Applies GELU approximation that is fast but somewhat inaccurate.

    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """Clip the range of possible GeLU outputs between [min, max]. This is
    especially useful for quantization purpose, as it allows mapping negatives
    values in the GeLU spectrum. For more information on this trick, please
    refer to https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(
                f'min should be < max (got min: {min}, max: {max})')

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Module):
    """See Gaussian Error Linear Units (Hendrycks et al.,
    https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear Unit) was
    originally introduced and coined, and see Sigmoid-Weighted Linear Units for
    Neural Network Function Approximation in Reinforcement Learning (Elfwing et
    al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated Activation
    Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where
    the SiLU was experimented with later."""

    def __init__(self):
        super().__init__()
        if version.parse(version.parse(
                torch.__version__).base_version) < version.parse('1.7'):
            self.act = self._silu_python
        else:
            self.act = nn.functional.silu

    def _silu_python(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class MishActivation(nn.Module):
    """See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra.,
    https://arxiv.org/abs/1908.08681).

    Also visit the official repository for the paper:
    https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(version.parse(
                torch.__version__).base_version) < version.parse('1.9'):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """Applies the linear activation function, i.e. forwarding input directly
    to output."""

    def forward(self, input: Tensor) -> Tensor:
        return input


ACT2FN = {
    'gelu': GELUActivation(),
    'gelu_10': ClippedGELUActivation(-10, 10),
    'gelu_fast': FastGELUActivation(),
    'gelu_new': NewGELUActivation(),
    'gelu_python': GELUActivation(use_gelu_python=True),
    'linear': LinearActivation(),
    'mish': MishActivation(),
    'quick_gelu': QuickGELUActivation(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'silu': SiLUActivation(),
    'swish': SiLUActivation(),
    'tanh': nn.Tanh(),
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f'function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}'
        )


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation('gelu_python')
gelu_new = get_activation('gelu_new')
gelu = get_activation('gelu')
gelu_fast = get_activation('gelu_fast')
quick_gelu = get_activation('quick_gelu')
silu = get_activation('silu')
mish = get_activation('mish')
linear_act = get_activation('linear')


def apply_chunking_to_forward(forward_fn: Callable[..., torch.Tensor],
                              chunk_size: int, chunk_dim: int,
                              *input_tensors) -> torch.Tensor:
    """This function chunks the `input_tensors` into smaller input tensor parts
    of size `chunk_size` over the dimension `chunk_dim`. It then applies a
    layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```
    """

    assert len(input_tensors
               ) > 0, f'{input_tensors} has to be a tuple/list of tensors'

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(
        inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f'forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input '
            'tensors are given')

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f'All input tenors have to be of the same shape: {tensor_shape}, '
                    f'found shape {input_tensor.shape[chunk_dim]}')

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f'The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk '
                f'size {chunk_size}')

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(
            input_tensor.chunk(num_chunks, dim=chunk_dim)
            for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(
            forward_fn(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class CLIPAttention(BaseModule):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')
        self.scale = self.head_dim ** -0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is'
                f' {attn_weights.size()}')

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is'
                    f' {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads,
                                                      tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads,
                                                      tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len,
                                  self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len,
                                       self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(BaseModule):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(BaseModule):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 num_attention_heads,
                 attention_dropout,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = hidden_size
        self.self_attn = CLIPAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(self,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=False):
        """
                Args:
                    hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                    attention_mask (`torch.FloatTensor`): attention mask of size
                        `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative
                        values. `(config.encoder_attention_heads,)`.
                    output_attentions (`bool`, *optional*):
                        Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                        returned tensors for more detail.
                """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoder(BaseModule):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 num_attention_heads,
                 num_hidden_layers,
                 attention_dropout,
                 output_attentions,
                 output_hidden_states,
                 use_return_dict,
                 gradient_checkpointing=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, intermediate_size, hidden_act,
                             num_attention_heads, attention_dropout)
            for _ in range(num_hidden_layers)
        ])
        self.gradient_checkpointing = gradient_checkpointing
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(self,
                inputs_embeds,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None else self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': encoder_states,
            'attentions': all_attentions
        }
