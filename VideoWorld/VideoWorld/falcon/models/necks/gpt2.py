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
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.model import BaseModule
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Block, GPT2Model
from transformers import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.generic import ModelOutput
from falcon.registry import MODELS
from torch.nn import CrossEntropyLoss
import math
import logging
from dataclasses import dataclass
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

@dataclass
class CausalLMOutputWithCrossAttentionsAug(ModelOutput):
    kl_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@MODELS.register_module()
class GPT2GenModel(BaseModule):
    def __init__(self, pretrain_path, vq_num, sepcial_token_num, max_position_embeddings=1024, use_text=True, init_cfg=None, use_aug_loss=False, no_ce_loss=False):
        super().__init__(init_cfg=init_cfg)
        # import pdb;pdb.set_trace()
        # if init_cfg is not None and init_cfg['type'] == 'Pretrained':
        # pretrain_path = init_cfg.checkpoint
        model_args = {
            'max_position_embeddings': max_position_embeddings,
            'ignore_mismatched_sizes':True,
            # 'gradient_checkpointing': gradient_checkpointing,
        }
        self.max_position_embeddings = max_position_embeddings
        self.vq_num = vq_num
        self.sepcial_token_num = sepcial_token_num
        # import pdb;pdb.set_trace()
        self.llm = GPT2LMHeadModel.from_pretrained(pretrain_path, **model_args) if not use_aug_loss else GPT2LMHeadModelAugLoss.from_pretrained(pretrain_path, **model_args)
        if use_text:
            self.llm.resize_token_embeddings(self.llm.get_input_embeddings().num_embeddings + sepcial_token_num + vq_num)
        else:
            self.llm.resize_token_embeddings(sepcial_token_num + vq_num)
        self.resize_pe_embeddings(max_position_embeddings)
        self.no_ce_loss = no_ce_loss
    def resize_pe_embeddings(self, new_max_position_embeddings):
        # import pdb;pdb.set_trace()
        old_embeddings = self.llm.transformer.wpe
        old_embedding_dim = old_embeddings.weight.shape[1]
        new_num_tokens = new_max_position_embeddings - old_embeddings.weight.shape[0]
        if new_num_tokens <= 0:
            return 
        
        new_embeddings = nn.Embedding(new_max_position_embeddings, old_embedding_dim, device=old_embeddings.weight.device,dtype=old_embeddings.weight.dtype,)
        self.llm._init_weights(new_embeddings)

        new_embeddings.weight.data[:old_embeddings.weight.shape[0], :] = old_embeddings.weight.data
        self.llm.transformer.wpe = new_embeddings

    def generate(self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_length: int = 512,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            eos_token_id=50256):
        # max_length = self.max_position_embeddings
        return self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, eos_token_id=eos_token_id)

    def _inference(
        self,
        tokens: torch.Tensor,
        past_key_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.llm(tokens, past_key_values=past_key_values)
        return output['logits'], output['past_key_values']

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        inputs_embeds=None, 
        return_dict=None,
        values=None,
        attn_lengths=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # import pdb;pdb.set_trace()
        if isinstance(self.llm, GPT2LMHeadModelAugLoss):
            output = self.llm(input_ids, attention_mask=attention_mask, labels=labels if not self.no_ce_loss else None, inputs_embeds=inputs_embeds, return_dict=return_dict, output_hidden_states=return_dict, values=values, attn_lengths=attn_lengths)
        else:
            output = self.llm(input_ids, attention_mask=attention_mask, labels=labels, inputs_embeds=inputs_embeds, return_dict=return_dict, output_hidden_states=return_dict)
        if return_dict:
            return output['logits'], output['loss'], output['hidden_states'][-1]
        return output['logits'], output.get('loss', None), output.get('kl_loss', None)


class GPT2LMHeadModelAugLoss(GPT2LMHeadModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        values = None,
        attn_lengths = None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        KLLoss = nn.KLDivLoss(reduction='none')
        lm_logits = self.lm_head(hidden_states)
        action_logits = lm_logits[:, 1:-2, -81:]
        all_kl_loss = 0
        temp = 2
        if values is not None:
            # import pdb;pdb.set_trace()
            
            for action_logit, value, attn_length in zip(action_logits, values, attn_lengths):
                attn_length = attn_length - 1
                value[value==-1] = 0
                # value = value[:, :-1]
                value = value.log()
                value = F.softmax(value, dim=1)
                action_logit = action_logit[:attn_length]
                action_logit = F.softmax(action_logit, dim=1)
                # import pdb;pdb.set_trace()
                # mau_kl = value * (value.log() - action_logit.log())
                kl_loss = KLLoss(action_logit.log(), value)
                if not (torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any()):
                    kl_loss[torch.isnan(kl_loss)] = 0
                    kl_loss[torch.isinf(kl_loss)] = 0
                    
                none_zero_num = (kl_loss > 0 ).sum()
                if none_zero_num == 0:
                    print_log("__________________none_zero_num == 0____________________", logger='current', level=logging.WARNING)
                    none_zero_num = 1
                kl_loss = kl_loss.sum() / none_zero_num
                all_kl_loss = all_kl_loss + kl_loss
                if kl_loss < 0 or kl_loss > 1:
                    print_log(f"---kl_loss---: {kl_loss}", logger='current', level=logging.WARNING)
                    kl_loss = action_logit.sum() * 0
            all_kl_loss = all_kl_loss / len(values) 
        # import pdb;pdb.set_trace()
        loss = None
        gamma = 2.0
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # probs = torch.exp(-ce_loss)
            # focal_loss = (1 - probs) ** gamma * ce_loss
            # none_ignore_num = (focal_loss != 0).sum()
            # loss = focal_loss.sum() / none_ignore_num

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentionsAug(
            kl_loss=all_kl_loss,
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

   
@MODELS.register_module()
class MMGPT2Model(BaseModule):
    def __init__(self, pretrain_path, vq_num, sepcial_token_num, mm_hidden_size, max_position_embeddings=1024, use_mm_proj=True, image_extra_pe=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # import pdb;pdb.set_trace()
        # if init_cfg is not None and init_cfg['type'] == 'Pretrained':
        # pretrain_path = init_cfg.checkpoint
        model_args = {
            'mm_hidden_size': mm_hidden_size,
            'use_mm_proj': use_mm_proj,
            'new_max_position_embeddings': max_position_embeddings,
            'image_extra_pe': image_extra_pe
        }
        self.sepcial_token_num = sepcial_token_num
        self.vq_num = vq_num
        self.llm = MMGPT2LMHeadModel.from_pretrained(pretrain_path, **model_args)
        self.llm.resize_token_embeddings(self.llm.get_input_embeddings().num_embeddings + sepcial_token_num + vq_num)
        # config.max_position_embeddings = kwargs.get('new_max_position_embeddings', 1024)
        self.llm.resize_pe_embeddings(max_position_embeddings)
        self.vocab_size = self.llm.get_input_embeddings().num_embeddings
    def generate(self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_length: int = 512,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            **kwargs):
        # import pdb;pdb.set_trace()
        return self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, **kwargs)

    def _inference(
        self,
        tokens: torch.Tensor,
        past_key_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.llm(tokens, past_key_values=past_key_values)
        return output['logits'], output['past_key_values']

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        image_features=None,
        return_dict=None,
        extra_position_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.llm(input_ids, attention_mask=attention_mask, labels=labels, image_features=image_features, return_dict=return_dict, extra_position_ids=extra_position_ids)
        return output['logits'], output['loss']

@MODELS.register_module()
class MMGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        config.mm_hidden_size = kwargs.get('mm_hidden_size', )
        config.use_mm_proj = kwargs.get('use_mm_proj', True)
        
        self.config = config
        config.max_position_embeddings = kwargs.get('new_max_position_embeddings', 1024)
        self.transformer = _GPT2Model(config)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size) if config.use_mm_proj else None
        self.dummy_feature = torch.zeros(1, config.mm_hidden_size)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.image_extra_pe = kwargs.get('image_extra_pe', False)
        self.gen_num = 0
        # Initialize weights and apply final processing
        self.post_init()

        
    
    def resize_pe_embeddings(self, new_max_position_embeddings):
        # import pdb;pdb.set_trace()
        old_embeddings = self.transformer.wpe
        old_embedding_dim = old_embeddings.weight.shape[1]
        new_num_tokens = new_max_position_embeddings - old_embeddings.weight.shape[0]
        if new_num_tokens <= 0:
            return 
        
        new_embeddings = nn.Embedding(new_max_position_embeddings, old_embedding_dim, device=old_embeddings.weight.device,dtype=old_embeddings.weight.dtype,)
        self._init_weights(new_embeddings)

        new_embeddings.weight.data[:old_embeddings.weight.shape[0], :] = old_embeddings.weight.data
        self.transformer.wpe = new_embeddings
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_features: Optional[torch.FloatTensor] = None,
        extra_position_ids=None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        # import pdb;pdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_features = self.mm_projector(image_features) if self.mm_projector is not None else image_features
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            extra_position_ids
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, image_features, extra_position_ids
        )
        # import pdb;pdb.set_trace()
        if attention_mask is not None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -inputs_embeds.shape[1] :] if input_ids is None else position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_position_ids=extra_position_ids if self.image_extra_pe else None
        )
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )



    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, image_features, extra_position_ids):

        # import pdb;pdb.set_trace()
        if image_features is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and image_features is not None
                and input_ids.shape[1] == 1
            ):
                # attention_mask = torch.ones(
                #     (input_ids.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                #     dtype=input_ids.dtype,
                #     device=input_ids.device,
                # )
                attention_mask = torch.cat([attention_mask, 
                                            torch.ones((input_ids.shape[0], past_key_values[-1][-1].shape[-2] + 1 - attention_mask.shape[1])).to(input_ids)
                                            ], dim=1)
                self.gen_num += 1
                if self.image_extra_pe:
                    # p = 7
                    # pe = positionalencoding2d(self.config.hidden_size, p, p)
                    # pe = pe.permute(1 ,2, 0).flatten(0, 1).to(attention_mask.device)
                    gen_pos = self.gen_num
                    extra_position_ids = torch.tensor(extra_position_ids).to(past_key_values[-1][-1])
                    if gen_pos > 1 and gen_pos < 49 + 1:
                        extra_position_ids = torch.cat([extra_position_ids, torch.zeros((input_ids.shape[0], 1)).to(input_ids)], dim=1)
                        extra_position_ids = torch.cat([extra_position_ids, torch.ones((input_ids.shape[0], gen_pos-1)).to(input_ids)], dim=1)
                    else:
                        extra_position_ids = torch.cat([extra_position_ids, torch.zeros((input_ids.shape[0], 1)).to(input_ids)], dim=1)
                    
            return input_ids, attention_mask, past_key_values, None, labels, extra_position_ids
    
        vit_attention_mask = torch.ones_like(image_features[:,:,0]).detach()
        attention_mask = torch.ones_like(input_ids).detach() if attention_mask is None else attention_mask
        new_input_embeds = []
        new_attention_mask = []
        new_extra_position_ids = [] if extra_position_ids is not None else None
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_embeds = self.transformer.wte(cur_input_ids)

                cur_input_embeds = (cur_input_embeds + (0.0 * self.mm_projector(self.dummy_feature)).sum()) if self.mm_projector is not None else cur_input_embeds
                new_attention_mask.append(attention_mask[batch_idx])
                new_input_embeds.append(cur_input_embeds)
                if extra_position_ids is not None:
                    new_extra_position_ids.append(extra_position_ids[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_new_attention_mask = []
            cur_new_extra_position_ids = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            if extra_position_ids is not None:
                cur_extra_position_ids = extra_position_ids[batch_idx]
            cur_attention_mask = attention_mask[batch_idx]
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                
                cur_vit_attention_mask = vit_attention_mask[cur_image_idx]
                image_token_start = image_token_indices[0]
                if extra_position_ids is not None:
                    cur_new_extra_position_ids.extend(cur_extra_position_ids[:image_token_start])
                    cur_new_extra_position_ids.extend([(cur_image_idx+1)*-1]*len(cur_vit_attention_mask))
                    cur_new_extra_position_ids.extend(cur_extra_position_ids[image_token_start + 1 : image_token_start + 2])

                cur_new_input_embeds.append(
                        self.transformer.wte(cur_input_ids[:image_token_start])
                    )
                cur_new_attention_mask.append(cur_attention_mask[:image_token_start])
                cur_new_attention_mask.append(cur_vit_attention_mask)
                cur_new_attention_mask.append(cur_attention_mask[image_token_start + 1 : image_token_start + 2])
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(
                        self.transformer.wte(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_new_labels.append(
                        cur_labels[image_token_start + 1 : image_token_start + 2]
                    )
                    cur_labels = cur_labels[image_token_start + 2 :]
                
                if extra_position_ids is not None:
                    cur_extra_position_ids = cur_extra_position_ids[image_token_start + 2 :]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 2 :]
                cur_attention_mask = cur_attention_mask[image_token_start + 2 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                
                cur_new_input_embeds.append(
                    self.transformer.wte(cur_input_ids)
                )
                cur_new_attention_mask.append(cur_attention_mask)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                if extra_position_ids is not None:
                    cur_new_extra_position_ids.extend(cur_extra_position_ids)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_attention_mask = torch.cat(cur_new_attention_mask, dim=0).bool()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_attention_mask.append(cur_new_attention_mask)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
            if extra_position_ids is not None:
                new_extra_position_ids.append(torch.tensor(cur_new_extra_position_ids).to(cur_new_input_embeds))
        
        
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        new_attention_mask = torch.stack(new_attention_mask, dim=0)
        attention_mask = new_attention_mask
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)
        if extra_position_ids is not None:
            new_extra_position_ids = torch.stack(new_extra_position_ids, dim=0)

        if attention_mask is not None and attention_mask.shape[1] < new_input_embeds.shape[1]:
            new_attn_mask_pad_left = torch.full(
                (
                    attention_mask.shape[0],
                    new_input_embeds.shape[1] - input_ids.shape[1],
                ),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat(
                (new_attn_mask_pad_left, attention_mask), dim=1
            )
            assert attention_mask.shape == new_input_embeds.shape[:2]       

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_extra_position_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, image_features=None,extra_position_ids=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        # import pdb;pdb.set_trace()
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "image_features": image_features,
                "extra_position_ids": extra_position_ids
            }
        )
        
        return model_inputs


@MODELS.register_module()
class _GPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(1024, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_position_ids=None
    ):  
        # import pdb;pdb.set_trace()
        # [ [(image_start_pos, image_length), ] ]
        if extra_position_ids is None:
            return super().forward(input_ids,
                                    past_key_values,
                                    attention_mask,
                                    token_type_ids,
                                    position_ids,
                                    head_mask,
                                    inputs_embeds,
                                    encoder_hidden_states,
                                    encoder_attention_mask,
                                    use_cache,
                                    output_attentions,
                                    output_hidden_states,
                                    return_dict)
        
        # assert inputs_embeds is not None
        # length = inputs_embeds.shape[1]
        hidden_size = self.embed_dim
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.view(-1, input_ids.size()[-1]))
            input_ids = None
        for bz, bz_extra_position_ids in enumerate(extra_position_ids):
            if inputs_embeds.shape[1] == 1:
                image_start_pos = torch.where(bz_extra_position_ids!=0)[0].max() if bz_extra_position_ids[-1] != 0 else -1
                if image_start_pos == -1:
                    continue
                image_length = len(bz_extra_position_ids) - image_start_pos
                p = 7
                pe = positionalencoding2d(hidden_size, p, p)
                pe = pe.permute(1 ,2, 0).flatten(0, 1).to(inputs_embeds) #length, dim
                pe = pe[image_length-1][None]
                inputs_embeds[bz] = inputs_embeds[bz] + pe
            else:
                unique_image_idx = torch.unique(bz_extra_position_ids).tolist()
                unique_image_idx.remove(0)
                # for image_idx in unique_image_idx:
                image_start_length = [(torch.where(bz_extra_position_ids==image_idx)[0][0], len(torch.where(bz_extra_position_ids==image_idx)[0])) for image_idx in unique_image_idx]
                for image_start_pos, image_length in image_start_length:
                    
                    p = int(image_length ** 0.5)
                    pe = positionalencoding2d(hidden_size, p, p)
                    pe = pe.permute(1 ,2, 0).flatten(0, 1).to(inputs_embeds) #length, dim
                    pe = pe[:image_length] if image_length < len(pe) else pe
                    inputs_embeds[bz, image_start_pos:image_start_pos+image_length] = inputs_embeds[bz, image_start_pos:image_start_pos+image_length] + pe
        

        return super().forward(input_ids,
                                past_key_values,
                                attention_mask,
                                token_type_ids,
                                position_ids,
                                head_mask,
                                inputs_embeds,
                                encoder_hidden_states,
                                encoder_attention_mask,
                                use_cache,
                                output_attentions,
                                output_hidden_states,
                                return_dict)


    
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe



@MODELS.register_module()
class MyGPT2Model(BaseModule, GenerationMixin):
    def __init__(self,
                hidden_size=1024,
                attn_pdrop=0.1,
                bos_token_id=50256,
                embd_pdrop=0.1,
                eos_token_id=50256,
                initializer_range=0.02,
                layer_norm_epsilon=1e-05,
                model_type='gpt2',
                n_ctx=1024,
                n_embd=1024,
                n_head=16,
                n_layer=24,
                n_positions=1024,
                n_special=0,
                predict_special_tokens=True,
                resid_pdrop=0.1,
                summary_activation=None,
                summary_first_dropout=0.1,
                summary_proj_to_labels=True,
                summary_type="cls_index",
                summary_use_proj=True,
                task_specific_params={
                    "text-generation":
                        {
                        "do_sample":True,
                        "max_length":50
                        },
                },
                vocab_size=50257,
                init_cfg=None):
        
        config = Config({
                'hidden_size':1024,
                'attn_pdrop':0.1,
                'bos_token_id':50256,
                'embd_pdrop':0.1,
                'eos_token_id':50256,
                'initializer_range':0.02,
                'layer_norm_epsilon':1e-05,
                'model_type':'gpt2',
                'n_ctx':1024,
                'n_embd':1024,
                'n_head':16,
                'n_layer':24,
                'n_positions':1024,
                'n_special':0,
                'predict_special_tokens':True,
                'resid_pdrop':0.1,
                'summary_activation':None,
                'summary_first_dropout':0.1,
                'summary_proj_to_labels':True,
                'summary_type':"cls_index",
                'summary_use_proj':True,
                'task_specific_params':{
                    "text-generation":
                        {
                        "do_sample":True,
                        "max_length":50
                        },
                },
                'vocab_size':50257,
                
        })
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def generate(self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_length: int = 512,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,):
        
        return self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

    def _inference(
        self,
        tokens: torch.Tensor,
        past_key_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.llm(tokens, past_key_values=past_key_values)
        return output['logits'], output['past_key_values']

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.llm(input_ids, attention_mask=attention_mask, labels=labels)
        return output['logits'], output['loss']