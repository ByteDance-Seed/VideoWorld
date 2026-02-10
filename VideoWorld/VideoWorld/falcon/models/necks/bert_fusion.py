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
import torch.nn as nn
from mmengine.config import Config
from mmengine.dist import get_comm_device
from mmengine.model import BaseModule
from transformers import GenerationMixin, GenerationConfig
from transformers.models.bert.modeling_bert import BertEncoder

from falcon.models.lbackbones.bert import BertOnlyMLMHead, BertPooler
from falcon.registry import MODELS


class BertEmbeddings(BaseModule):
    """Construct the embeddings from word, position and token_type
    embeddings."""

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 pad_token_id,
                 max_position_embeddings,
                 type_vocab_size,
                 hidden_dropout_prob,
                 prompt=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        if prompt:
            self.prompt_embedding = nn.Embedding(1, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = 'absolute'
        self.register_buffer(
            'position_ids',
            torch.arange(max_position_embeddings).expand((1, -1)))

        self.register_buffer(
            'token_type_ids',
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                                seq_length +
                                                past_key_values_length]

        if isinstance(token_type_ids, str):
            if token_type_ids == 'prompt':
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeddings = self.prompt_embedding(token_type_ids)
        else:
            if token_type_ids is None:
                if hasattr(self, 'token_type_ids'):
                    buffered_token_type_ids = self.token_type_ids[:, :
                                                                     seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                        input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(
                        input_shape,
                        dtype=torch.long,
                        device=self.position_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@MODELS.register_module()
class BertFusion(BaseModule, GenerationMixin):

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 pad_token_id=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 hidden_dropout_prob=0.1,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 is_decoder=False,
                 add_cross_attention=False,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 num_hidden_layers=12,
                 gradient_checkpointing=False,
                 initializer_range=0.02,
                 use_cache=True,
                 with_pooler=False,
                 add_input_proj=False,
                 input_dim=768,
                 video_gen=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.initializer_range = initializer_range
        self.has_cross_attn = add_cross_attention
        self.with_pooler = with_pooler
        self.is_decoder = is_decoder
        self.use_cache = use_cache

        self.embeddings = BertEmbeddings(vocab_size, hidden_size, pad_token_id,
                                         max_position_embeddings,
                                         type_vocab_size, hidden_dropout_prob)
        config = Config({
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'max_position_embeddings': max_position_embeddings,
            'is_decoder': is_decoder,
            'hidden_dropout_prob': hidden_dropout_prob,
            'add_cross_attention': add_cross_attention,
            'intermediate_size': intermediate_size,
            'hidden_act': hidden_act,
            'num_hidden_layers': num_hidden_layers,
            'position_embedding_type': 'absolute',
            'gradient_checkpointing': gradient_checkpointing,
            'chunk_size_feed_forward': 0,
            'layer_norm_eps': 1e-12,
            'is_encoder_decoder': False,
        })

        self.config = config

        self.encoder = BertEncoder(config)
        if self.with_pooler:
            self.pooler = BertPooler(hidden_size)
        else:
            self.pooler = None
        self.video_gen = video_gen
        if add_input_proj:
            self.input_proj = nn.Linear(input_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()


        self.cls = None

        self.cls = BertOnlyMLMHead(hidden_act, hidden_size, vocab_size)
        self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig(pad_token_id=pad_token_id, max_length=32)
        self.device = get_comm_device()



    def init_weights(self):
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    m.weight.data.normal_(mean=0.0, std=self.initializer_range)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    m.weight.data.normal_(mean=0.0, std=self.initializer_range)
                    if m.padding_idx is not None:
                        m.weight.data[m.padding_idx].zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)

    def invert_attention_mask(self, encoder_attention_mask):
        """Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:,
                                              None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None,
                                              None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (
                                                  1.0 - encoder_extended_attention_mask) * torch.finfo(
            encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask

    def can_generate(self):
        return True

    def get_output_embeddings(self):
        return self.cls

    def get_decoder(self):
        return self

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      image_embeds=None,
                                      attention_mask=None,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      **kwargs):

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                # "position_ids": position_ids,
                # "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
            image_embeds=None,
        past_key_values=None,
        use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):

        if self.is_decoder:
            use_cache = use_cache if use_cache is not None else self.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :
                                                                            seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        input_feat = embedding_output
        txt_len = input_feat.shape[1]

        image_embeds = self.input_proj(image_embeds)

        attention_mask = attention_mask.to(input_feat.dtype)
        attention_mask = torch.bmm(
            attention_mask.unsqueeze(dim=2),
            attention_mask.unsqueeze(dim=1)).to(input_feat.dtype)
        input_feat = torch.cat((image_embeds, input_feat), dim=1)
        bs, seq_len, _ = input_feat.size()
        attention_mask_fusion = torch.ones(
            (bs, seq_len, seq_len)).to(attention_mask)
        # print("attention_mask.size()={},txt_len={}".format(attention_mask.size(),txt_len))
        attention_mask_tril_cond = torch.arange(attention_mask.size(-1)).to(attention_mask)
        attention_mask.masked_fill_(
            attention_mask_tril_cond >= (attention_mask_tril_cond + 1).view(attention_mask.size(-1), 1), 0)

        attention_mask_fusion[:, -txt_len:, -txt_len:] = attention_mask
        attention_mask_fusion[:, :-txt_len, -txt_len:] = 0


        attention_mask_fusion = attention_mask_fusion.unsqueeze(1)
        attention_mask_fusion = attention_mask_fusion.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask_fusion = (1.0 - attention_mask_fusion) * torch.finfo(
            attention_mask.dtype).min

        encoder_outputs = self.encoder(
            input_feat,
            attention_mask=attention_mask_fusion,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        sequence_output = encoder_outputs['last_hidden_state']
        pred = None

        pred = self.cls(sequence_output[:, -txt_len:, :])

        return Config({
            'last_hidden_state': sequence_output,
            'pred': pred,
            "logits": pred,
            # 'past_key_values': next_cache,
            # 'hidden_states': all_hidden_states,
            # 'attentions': all_self_attns
        })
