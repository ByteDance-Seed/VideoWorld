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
from mmengine.dist import get_comm_device
from mmengine.logging import print_log
from mmengine.model import BaseModule
from transformers import GenerationMixin, GenerationConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, _expand_mask, _make_causal_mask

from falcon.registry import MODELS


@MODELS.register_module()
class LlamaFusionGenModel(BaseModule, GenerationMixin):

    def __init__(self,
                 hidden_size=3200,
                 intermediate_size=8640,
                 num_hidden_layers=26,
                 num_attention_heads=32,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 vocab_size=32000,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 return_dict=False,
                 output_hidden_states=False,
                 attention_bias=False,
                 gen_v_vocab_size=None,
                 last_only=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size,
                                         self.padding_idx)

        num_key_value_heads = num_attention_heads

        config = Config({
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'pretraining_tp': 1,
            'hidden_act': hidden_act,
            'max_position_embeddings': max_position_embeddings,
            'initializer_range': initializer_range,
            'rms_norm_eps': rms_norm_eps,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'tie_word_embeddings': tie_word_embeddings,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'is_encoder_decoder': False,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'attention_bias': attention_bias,
            'use_return_dict': return_dict
        })
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing
        self.gen_v_vocab_size = gen_v_vocab_size
        self.v_token_embeddings = nn.Embedding(8193, config['hidden_size'])
        if self.gen_v_vocab_size is None:
            self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, gen_v_vocab_size + 1, bias=False)

        # self.seg_align = None
        # if multi_d:
        #     self.seg_align = LlamaDecoderLayer(config)

        ################### this is for generation test and adapter the GenerationMixin
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(eos_token_id=8192, max_new_tokens=198, min_new_tokens=197,
                                                  pad_token_id=8192)
        self.device = get_comm_device()
        self.last_only = last_only

    def init_weights(self):
        super().init_weights()

    def can_generate(self):
        return True

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, text_len,
                                        past_key_values_length):
        combined_attention_mask = None
        total_len = input_shape[-1]
        bs = input_shape[0]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # visual attention is bi-attention
        # dtype = inputs_embeds.dtype
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, :, text_len:] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool), 0)
        #
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, text_len:, :text_len] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool),
        #     torch.finfo(dtype).min)

        # Put attentoin mask behind visal attention is necessary when when there are padding tokens in img prompts
        # print("attention_mask.size()={},input_shape={}".format(attention_mask.size(), input_shape))
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      visual_mask_ids=None,
                                      input_ids_length=None,
                                      **kwargs):
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]
        # position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        #     model_inputs = {"input_ids": input_ids}

        inputs_current_length = input_ids.size(1)

        if inputs_current_length - input_ids_length > 0:
            visual_mask_ids = input_ids[:, input_ids_length:, ]

        model_inputs = {"input_ids": input_ids[:, :input_ids_length]}
        # print(
        #     "attention_mask={},input_ids={},image_embeds={},img_prefix={},img_postfix={},img_prefix_mask={},img_postfix_mask={}".format(
        #         attention_mask.size(), input_ids.size(),
        #         image_embeds.size(), img_prefix.size(), img_postfix.size(), img_prefix_mask.size(),
        #         img_postfix_mask.size()))

        # print("input_ids={},attention_mask={},image_embeds={},visual_mask_ids={}".format(input_ids.size(),
        #                                                                                  attention_mask.size(),
        #                                                                                  image_embeds.size(),
        #                                                                                  visual_mask_ids))

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": False,
                "attention_mask": attention_mask[:, :input_ids_length],
                "image_embeds": image_embeds,
                "visual_mask_ids": visual_mask_ids
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            image_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            visual_mask_ids: Optional[torch.LongTensor] = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        img_seq_lenth = 0
        prompted_img_attn_mask = None
        pred_seq_lenth = 0
        pred_img_att_mask = None
        if image_embeds is not None:
            img_seq_lenth += image_embeds.size(1)
            img_attetion_mask = torch.ones((batch_size, image_embeds.size(1)),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            prompted_img_attn_mask = img_attetion_mask

        if visual_mask_ids is not None:
            img_pred_embeds = self.v_token_embeddings(visual_mask_ids)
            pred_seq_lenth = img_pred_embeds.size(1)
            pred_img_att_mask = torch.ones((batch_size, pred_seq_lenth),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask, pred_img_att_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds, img_pred_embeds], dim=1)
        else:
            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds], dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length + img_seq_lenth + pred_seq_lenth),
            inputs_embeds, seq_length, past_key_values_length)

        total_length = inputs_embeds.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                total_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, total_length)
        else:
            position_ids = position_ids.view(-1, total_length).long()

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
        # if self.seg_align is not None:
        #     layer_outputs = self.seg_align(hidden_states)
        #     hidden_states = layer_outputs[0]

        text_pred = self.lm_head(hidden_states[:, -pred_seq_lenth - 1:, :])
        # print("text_pred.size=", text_pred.size(1))

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return Config({
            'last_hidden_state': hidden_states,
            'text_pred': text_pred,
            "logits": text_pred,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns
        })


@MODELS.register_module()
class LlamaFusionPrefixGenModel(BaseModule, GenerationMixin):

    def __init__(self,
                 hidden_size=3200,
                 intermediate_size=8640,
                 num_hidden_layers=26,
                 num_attention_heads=32,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 vocab_size=32000,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 return_dict=False,
                 output_hidden_states=False,
                 attention_bias=False,
                 gen_v_vocab_size=None,
                 multi_d=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size,
                                         self.padding_idx)

        num_key_value_heads = num_attention_heads

        config = Config({
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'pretraining_tp': 1,
            'hidden_act': hidden_act,
            'max_position_embeddings': max_position_embeddings,
            'initializer_range': initializer_range,
            'rms_norm_eps': rms_norm_eps,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'tie_word_embeddings': tie_word_embeddings,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'is_encoder_decoder': False,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'attention_bias': attention_bias,
            'use_return_dict': return_dict
        })
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing
        self.gen_v_vocab_size = gen_v_vocab_size
        self.v_token_embeddings = nn.Embedding(8193, config['hidden_size'])
        self.v_token_proj = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] * 2),
            nn.SiLU(),
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
        )
        if self.gen_v_vocab_size is None:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, gen_v_vocab_size + 1, bias=False)

        # self.seg_align = None
        # if multi_d:
        #     self.seg_align = LlamaDecoderLayer(config)

        ################### this is for generation test and adapter the GenerationMixin
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(eos_token_id=8192, max_new_tokens=198, min_new_tokens=197,
                                                  pad_token_id=8192)
        self.device = get_comm_device()

    def init_weights(self):
        super().init_weights()

    def can_generate(self):
        return True

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, text_len,
                                        past_key_values_length):
        combined_attention_mask = None
        total_len = input_shape[-1]
        bs = input_shape[0]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # visual attention is bi-attention
        # dtype = inputs_embeds.dtype
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, :, text_len:] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool), 0)
        #
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, text_len:, :text_len] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool),
        #     torch.finfo(dtype).min)

        # Put attentoin mask behind visal attention is necessary when when there are padding tokens in img prompts
        # print("attention_mask.size()={},input_shape={}".format(attention_mask.size(), input_shape))
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      visual_mask_ids=None,
                                      input_ids_length=None,
                                      **kwargs):
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]
        # position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        #     model_inputs = {"input_ids": input_ids}

        inputs_current_length = input_ids.size(1)

        if inputs_current_length - input_ids_length > 0:
            visual_mask_ids = input_ids[:, input_ids_length:, ]

        model_inputs = {"input_ids": input_ids[:, :input_ids_length]}
        # print(
        #     "attention_mask={},input_ids={},image_embeds={},img_prefix={},img_postfix={},img_prefix_mask={},img_postfix_mask={}".format(
        #         attention_mask.size(), input_ids.size(),
        #         image_embeds.size(), img_prefix.size(), img_postfix.size(), img_prefix_mask.size(),
        #         img_postfix_mask.size()))

        # print("input_ids={},attention_mask={},image_embeds={},visual_mask_ids={}".format(input_ids.size(),
        #                                                                                  attention_mask.size(),
        #                                                                                  image_embeds.size(),
        #                                                                                  visual_mask_ids))

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": False,
                "attention_mask": attention_mask[:, :input_ids_length],
                "image_embeds": image_embeds,
                "visual_mask_ids": visual_mask_ids
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            image_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            visual_mask_ids: Optional[torch.LongTensor] = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        img_seq_lenth = 0
        prompted_img_attn_mask = None
        pred_seq_lenth = 0
        pred_img_att_mask = None
        if image_embeds is not None:
            img_seq_lenth += image_embeds.size(1)
            img_attetion_mask = torch.ones((batch_size, image_embeds.size(1)),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            prompted_img_attn_mask = img_attetion_mask

        if visual_mask_ids is not None:
            img_pred_embeds = self.v_token_embeddings(visual_mask_ids)
            img_pred_embeds = self.v_token_proj(img_pred_embeds)
            pred_seq_lenth = img_pred_embeds.size(1)
            pred_img_att_mask = torch.ones((batch_size, pred_seq_lenth),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask, pred_img_att_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds, img_pred_embeds], dim=1)
        else:
            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds], dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length + img_seq_lenth + pred_seq_lenth),
            inputs_embeds, seq_length, past_key_values_length)

        total_length = inputs_embeds.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                total_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, total_length)
        else:
            position_ids = position_ids.view(-1, total_length).long()

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
        # if self.seg_align is not None:
        #     layer_outputs = self.seg_align(hidden_states)
        #     hidden_states = layer_outputs[0]

        text_pred = self.lm_head(hidden_states[:, -pred_seq_lenth - 1:, :])
        # print("text_pred.size=", text_pred.size(1))

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return Config({
            'last_hidden_state': hidden_states,
            'text_pred': text_pred,
            "logits": text_pred,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns
        })


@MODELS.register_module()
class LlamaFusionGenMixModel(BaseModule, GenerationMixin):

    def __init__(self,
                 hidden_size=3200,
                 intermediate_size=8640,
                 num_hidden_layers=26,
                 num_attention_heads=32,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 vocab_size=32000,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 return_dict=False,
                 output_hidden_states=False,
                 attention_bias=False,
                 gen_v_vocab_size=None,
                 last_only=False,
                 flash_attn_2_enabled=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size,
                                         self.padding_idx)

        num_key_value_heads = num_attention_heads

        config = Config({
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'pretraining_tp': 1,
            'hidden_act': hidden_act,
            'max_position_embeddings': max_position_embeddings,
            'initializer_range': initializer_range,
            'rms_norm_eps': rms_norm_eps,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'tie_word_embeddings': tie_word_embeddings,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'is_encoder_decoder': False,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'attention_bias': attention_bias,
            'use_return_dict': return_dict,
            '_flash_attn_2_enabled': flash_attn_2_enabled
        })
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing
        self.gen_v_vocab_size = gen_v_vocab_size
        self.v_token_embeddings = nn.Embedding(8193, config['hidden_size'])

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.lm_v_head = nn.Linear(
            config.hidden_size, gen_v_vocab_size + 1, bias=False)

        # self.seg_align = None
        # if multi_d:
        #     self.seg_align = LlamaDecoderLayer(config)

        ################### this is for generation test and adapter the GenerationMixin
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(eos_token_id=8192, max_new_tokens=198, min_new_tokens=197,
                                                  pad_token_id=8192)
        self.device = get_comm_device()
        self.last_only = last_only

    def init_weights(self):
        super().init_weights()

    def can_generate(self):
        return True

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, text_len,
                                        past_key_values_length):
        combined_attention_mask = None
        total_len = input_shape[-1]
        bs = input_shape[0]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # visual attention is bi-attention
        # dtype = inputs_embeds.dtype
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, :, text_len:] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool), 0)
        #
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, text_len:, :text_len] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool),
        #     torch.finfo(dtype).min)

        # Put attentoin mask behind visal attention is necessary when when there are padding tokens in img prompts
        # print("attention_mask.size()={},input_shape={}".format(attention_mask.size(), input_shape))
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def get_output_embeddings(self):
        return self.lm_v_head

    def get_decoder(self):
        return self

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      visual_mask_ids=None,
                                      input_ids_length=None,
                                      **kwargs):
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]
        # position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        #     model_inputs = {"input_ids": input_ids}

        inputs_current_length = input_ids.size(1)

        if inputs_current_length - input_ids_length > 0:
            visual_mask_ids = input_ids[:, input_ids_length:, ]

        model_inputs = {"input_ids": input_ids[:, :input_ids_length]}
        # print(
        #     "attention_mask={},input_ids={},image_embeds={},img_prefix={},img_postfix={},img_prefix_mask={},img_postfix_mask={}".format(
        #         attention_mask.size(), input_ids.size(),
        #         image_embeds.size(), img_prefix.size(), img_postfix.size(), img_prefix_mask.size(),
        #         img_postfix_mask.size()))

        # print("input_ids={},attention_mask={},image_embeds={},visual_mask_ids={}".format(input_ids.size(),
        #                                                                                  attention_mask.size(),
        #                                                                                  image_embeds.size(),
        #                                                                                  visual_mask_ids))

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": False,
                "attention_mask": attention_mask[:, :input_ids_length],
                "image_embeds": image_embeds,
                "visual_mask_ids": visual_mask_ids
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            image_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pred_text=False,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            visual_mask_ids: Optional[torch.LongTensor] = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        img_seq_lenth = 0
        prompted_img_attn_mask = None
        pred_seq_lenth = 0
        pred_img_att_mask = None
        if image_embeds is not None:
            img_seq_lenth += image_embeds.size(1)
            img_attetion_mask = torch.ones((batch_size, image_embeds.size(1)),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            prompted_img_attn_mask = img_attetion_mask

        if visual_mask_ids is not None:
            img_pred_embeds = self.v_token_embeddings(visual_mask_ids)
            pred_seq_lenth = img_pred_embeds.size(1)
            pred_img_att_mask = torch.ones((batch_size, pred_seq_lenth),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask, pred_img_att_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds, img_pred_embeds], dim=1)
        else:
            attention_mask = torch.cat(
                [prompted_img_attn_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat(
                [image_embeds, inputs_embeds], dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length + img_seq_lenth + pred_seq_lenth),
            inputs_embeds, seq_length, past_key_values_length)

        total_length = inputs_embeds.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                total_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, total_length)
        else:
            position_ids = position_ids.view(-1, total_length).long()

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
        # if self.seg_align is not None:
        #     layer_outputs = self.seg_align(hidden_states)
        #     hidden_states = layer_outputs[0]

        text_pred = self.lm_head(hidden_states[:, -pred_seq_lenth - 1 - 32: -pred_seq_lenth, ])
        v_mask_pred = self.lm_v_head(hidden_states[:, -pred_seq_lenth - 1:, ])
        # print("text_pred.size=", text_pred.size(1))

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return Config({
            'last_hidden_state': hidden_states,
            'text_pred': text_pred,
            'v_pred': v_mask_pred,
            "logits": v_mask_pred,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns
        })


@MODELS.register_module()
class LlamaFusionGenUniModel(BaseModule, GenerationMixin):

    def __init__(self,
                 hidden_size=3200,
                 intermediate_size=8640,
                 num_hidden_layers=26,
                 num_attention_heads=32,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 initializer_range=0.02,
                 vocab_size=32000,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 tie_word_embeddings=False,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 return_dict=False,
                 output_hidden_states=False,
                 attention_bias=False,
                 last_only=False,
                 flash_attn_2_enabled=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size,
                                         self.padding_idx)

        num_key_value_heads = num_attention_heads

        config = Config({
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'pretraining_tp': 1,
            'hidden_act': hidden_act,
            'max_position_embeddings': max_position_embeddings,
            'initializer_range': initializer_range,
            'rms_norm_eps': rms_norm_eps,
            'use_cache': use_cache,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
            'tie_word_embeddings': tie_word_embeddings,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'is_encoder_decoder': False,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'attention_bias': attention_bias,
            'use_return_dict': return_dict,
            '_flash_attn_2_enabled': flash_attn_2_enabled
        })
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = gradient_checkpointing

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # self.seg_align = None
        # if multi_d:
        #     self.seg_align = LlamaDecoderLayer(config)

        ################### this is for generation test and adapter the GenerationMixin
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(eos_token_id=2, max_length=1024)
        self.device = get_comm_device()
        self.last_only = last_only

    def init_weights(self):
        super().init_weights()

    def can_generate(self):
        return True

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, text_len,
                                        past_key_values_length):
        combined_attention_mask = None
        total_len = input_shape[-1]
        bs = input_shape[0]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # visual attention is bi-attention
        # dtype = inputs_embeds.dtype
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, :, text_len:] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool), 0)
        #
        # tmp_mask = torch.zeros((bs, 1, total_len, total_len),
        #                        device=combined_attention_mask.device)
        # tmp_mask[:, :, text_len:, :text_len] = 1
        # combined_attention_mask = combined_attention_mask.masked_fill(
        #     tmp_mask.to(torch.bool),
        #     torch.finfo(dtype).min)

        # Put attentoin mask behind visal attention is necessary when when there are padding tokens in img prompts
        # print("attention_mask.size()={},input_shape={}".format(attention_mask.size(), input_shape))
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      **kwargs):
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]
        # position_ids = kwargs.get("position_ids", None)
        # if attention_mask is not None and position_ids is None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        #     model_inputs = {"input_ids": input_ids}

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": False,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds,
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            image_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pred_text=False,
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        img_seq_lenth = 0
        prompted_img_attn_mask = None
        pred_seq_lenth = 0
        pred_img_att_mask = None
        if image_embeds is not None:
            img_seq_lenth += image_embeds.size(1)
            img_attetion_mask = torch.ones((batch_size, image_embeds.size(1)),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)

            prompted_img_attn_mask = img_attetion_mask

        attention_mask = torch.cat(
            [prompted_img_attn_mask, attention_mask], dim=1)
        inputs_embeds = torch.cat(
            [image_embeds, inputs_embeds], dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length + img_seq_lenth + pred_seq_lenth),
            inputs_embeds, seq_length, past_key_values_length)

        total_length = inputs_embeds.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                total_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, total_length)
        else:
            position_ids = position_ids.view(-1, total_length).long()

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
        # if self.seg_align is not None:
        #     layer_outputs = self.seg_align(hidden_states)
        #     hidden_states = layer_outputs[0]

        text_pred = self.lm_head(hidden_states[:, img_seq_lenth:])

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return Config({
            'last_hidden_state': hidden_states,
            'text_pred': text_pred,
            "logits": text_pred,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns
        })
