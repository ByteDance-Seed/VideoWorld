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
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.model import BaseModule
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, _expand_mask, _make_causal_mask

from falcon.registry import MODELS


@MODELS.register_module()
class LlamaModel(BaseModule):

    def __init__(self,
                 vocab_size=32000,
                 hidden_size=4096,
                 intermediate_size=11008,
                 num_hidden_layers=32,
                 num_attention_heads=32,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 gradient_checkpointing=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size,
                                         self.padding_idx)

        config = Config({
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'hidden_act': hidden_act,
            'max_position_embeddings': max_position_embeddings,
            'initializer_range': initializer_range,
            'rms_norm_eps': rms_norm_eps,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'tie_word_embeddings': tie_word_embeddings,
        })
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing

    def init_weights(self):
        super().init_weights()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either decoder_input_ids or decoder_inputs_embeds'
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and hidden_states.requires_grad:
            if use_cache:
                print_log(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and hidden_states.requires_grad:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns
        }
