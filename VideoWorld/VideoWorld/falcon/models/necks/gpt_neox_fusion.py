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
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dist import get_comm_device
from mmengine.logging import print_log
from mmengine.model import BaseModule
from torch import Tensor
from transformers import GenerationMixin, GenerationConfig
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from falcon.registry import MODELS


@MODELS.register_module()
class GPTNeoXModelFusion(BaseModule, GenerationMixin):
    def __init__(self,
                 vocab_size=50688,
                 hidden_size=4096,
                 num_hidden_layers=16,
                 num_attention_heads=32,
                 intermediate_size=16384,
                 hidden_act="gelu",
                 rotary_pct=0.25,
                 rotary_emb_base=10000,
                 max_position_embeddings=4096,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 use_cache=True,
                 bos_token_id=0,
                 eos_token_id=0,
                 tie_word_embeddings=False,
                 use_parallel_residual=True,
                 gradient_checkpointing=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        config = GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_parallel_residual=use_parallel_residual,
        )

        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

        ################### this is for generation test and adapter the GenerationMixin
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(bos_token_id=0,
                                                  eos_token_id=0, max_length=48)
        self.device = get_comm_device()

    def init_weights(self):
        super().init_weights()

    def can_generate(self):
        return True

    def get_input_embeddings(self):
        return self.embed_in

    def get_output_embeddings(self):
        return self.embed_out

    def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked=False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def prepare_inputs_for_generation(self, input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      img_prefix=None,
                                      img_prefix_mask=None,
                                      **kwargs):

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": False,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds,
                "img_prefix": img_prefix,
                "img_prefix_mask": img_prefix_mask,
            }
        )
        return model_inputs

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                image_embeds: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                img_prefix: Optional[torch.LongTensor] = None,
                img_prefix_mask: Optional[torch.Tensor] = None,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        self.dtype = inputs_embeds.dtype

        img_seq_lenth = 0
        pre_seq_length = 0
        prompted_img_embeds, prompted_img_attn_mask = [], []
        if img_prefix is not None:
            img_prefix_embeds = self.embed_in(img_prefix)
            pre_seq_length = img_prefix_embeds.size(1)
            prompted_img_embeds.append(img_prefix_embeds)
            prompted_img_attn_mask.append(img_prefix_mask)
        if image_embeds is not None:
            img_seq_lenth = image_embeds.size(1)
            img_attetion_mask = torch.ones((batch_size, image_embeds.size(1)),
                                           dtype=torch.bool,
                                           device=inputs_embeds.device)
            prompted_img_embeds.append(image_embeds)
            prompted_img_attn_mask.append(img_attetion_mask)

        if len(prompted_img_embeds) > 0 and image_embeds is not None:
            attention_mask = torch.cat(
                prompted_img_attn_mask + [attention_mask], dim=1)
            inputs_embeds = torch.cat(
                prompted_img_embeds + [inputs_embeds], dim=1)

        total_length = inputs_embeds.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, total_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, total_length)
        else:
            position_ids = position_ids.view(-1, total_length).long()

        # Attention mask
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and hidden_states.requires_grad:
            if use_cache:
                print_log(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and hidden_states.requires_grad:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)

        lm_logits = self.embed_out(hidden_states[:, -seq_length:, :])

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return Config({
            'last_hidden_state': hidden_states,
            'text_pred': lm_logits,
            "logits": lm_logits,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        })
