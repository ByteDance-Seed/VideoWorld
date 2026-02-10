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
from mmengine.model import BaseModule

from falcon.models.lbackbones.utils import CLIPMLP, CLIPAttention, CLIPEncoder
from falcon.registry import MODELS


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask, dtype, tgt_len=None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len,
    src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min)


class CLIPTextEmbeddings(BaseModule):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        embed_dim = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings,
                                               embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        seq_length = input_ids.shape[
            -1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPTextTransformer(BaseModule):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings,
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

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

        embed_dim = hidden_size
        self.embeddings = CLIPTextEmbeddings(embed_dim, vocab_size,
                                             max_position_embeddings)
        self.encoder = CLIPEncoder(
            hidden_size,
            intermediate_size,
            hidden_act,
            num_attention_heads,
            num_hidden_layers,
            attention_dropout,
            output_attentions,
            output_hidden_states,
            use_return_dict,
            gradient_checkpointing=gradient_checkpointing)

        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None else self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is None:
            raise ValueError('You have to specify either input_ids')

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]),
            input_ids.to(torch.int).argmax(dim=-1)]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs.hidden_states,
            'attentions': encoder_outputs.attentions,
        }


@MODELS.register_module()
class CLIPTextModel(BaseModule):

    def __init__(self,
                 hidden_size=512,
                 vocab_size=49408,
                 max_position_embeddings=77,
                 intermediate_size=2048,
                 hidden_act='quick_gelu',
                 num_attention_heads=8,
                 num_hidden_layers=12,
                 attention_dropout=0.0,
                 initializer_factor=1.0,
                 initializer_range=0.02,
                 projection_dim=512,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 output_hidden_states=False,
                 grad=False,
                 use_return_dict=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.text_model = CLIPTextTransformer(
            hidden_size,
            vocab_size,
            max_position_embeddings,
            intermediate_size,
            hidden_act,
            num_attention_heads,
            num_hidden_layers,
            attention_dropout,
            output_attentions,
            output_hidden_states,
            use_return_dict,
            gradient_checkpointing=gradient_checkpointing)

        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.text_embed_dim = hidden_size
        self.projection_dim = projection_dim

        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False)

        for _, p in self.text_model.named_parameters():
            p.requires_grad = grad

    def init_weights(self):
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            nn.init.normal_(
                self.text_projection.weight,
                std=self.text_embed_dim ** -0.5 * self.initializer_factor)
            for m in self.modules():
                if isinstance(m, CLIPTextEmbeddings):
                    factor = self.initializer_factor
                    nn.init.normal_(
                        m.class_embedding,
                        mean=0.0,
                        std=m.embed_dim ** -0.5 * factor)
                    nn.init.normal_(
                        m.patch_embedding.weight,
                        std=self.initializer_range * factor)
                    nn.init.normal_(
                        m.position_embedding.weight,
                        std=self.initializer_range * factor)
                elif isinstance(m, CLIPAttention):
                    factor = self.initializer_factor
                    in_proj_std = (m.embed_dim ** -0.5) * (
                            (2 * m.num_hidden_layers) ** -0.5) * factor
                    out_proj_std = (m.embed_dim ** -0.5) * factor
                    nn.init.normal_(m.q_proj.weight, std=in_proj_std)
                    nn.init.normal_(m.k_proj.weight, std=in_proj_std)
                    nn.init.normal_(m.v_proj.weight, std=in_proj_std)
                    nn.init.normal_(m.out_proj.weight, std=out_proj_std)
                elif isinstance(m, CLIPMLP):
                    factor = self.initializer_factor
                    in_proj_std = ((m.hidden_size ** -0.5) *
                                   ((2 * m.num_hidden_layers) ** -0.5) * factor)
                    fc_std = (2 * m.hidden_size) ** -0.5 * factor
                    nn.init.normal_(m.fc1.weight, std=fc_std)
                    nn.init.normal_(m.fc2.weight, std=in_proj_std)

                if isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        return {'text_outputs': text_outputs, 'proj_feature': text_embeds}
