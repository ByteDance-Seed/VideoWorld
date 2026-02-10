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

from falcon.models.lbackbones.utils import CLIPEncoder
from falcon.registry import MODELS
import torch.nn.functional as F

# from falcon.utils import get_root_logger


class CLIPVisionEmbeddings(BaseModule):

    def __init__(self, hidden_size, image_size, patch_size, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)
        self.register_buffer('position_ids',
                             torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values=None):
        # import pdb;pdb.set_trace()
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings




class CLIPViT(BaseModule):

    def __init__(self,
                 image_size,
                 patch_size,
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
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)

        embed_dim = hidden_size
        self.embeddings = CLIPVisionEmbeddings(hidden_size, image_size,
                                               patch_size)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=1e-5)
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
        # self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.use_return_dict = use_return_dict

        


    def forward(self,
                pixel_values=None,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None else self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        # pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs.hidden_states,
            'attentions': encoder_outputs.attentions,
        }


@MODELS.register_module()
class CLIPVisionModel(BaseModule):

    def __init__(self,
                 image_size=224,
                 patch_size=32,
                 hidden_size=768,
                 intermediate_size=3072,
                 hidden_act='quick_gelu',
                 num_attention_heads=12,
                 num_hidden_layers=12,
                 attention_dropout=0.0,
                 initializer_factor=1.0,
                 initializer_range=0.02,
                 projection_dim=512,
                 gradient_checkpointing=False,
                 output_attentions=False,
                 output_hidden_states=False,
                 use_return_dict=False,
                 use_proj=True,
                 fix_vision=False,
                 use_mlp=False,
                 resize_pos_embed=False,
                 resize_shape=112,
                 init_cfg=None,
                 checkpoint=None):
        super().__init__(init_cfg=init_cfg)

        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vision_model = CLIPViT(
            image_size,
            patch_size,
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
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        # import pdb;pdb.set_trace()
        self.vision_embed_dim = hidden_size
        self.use_proj = use_proj
        if use_proj:
            self.projection_dim = projection_dim

            self.visual_projection = nn.Linear(
                self.vision_embed_dim, self.projection_dim, bias=False)
            if use_mlp:
                self.visual_projection = nn.Sequential(
                    nn.Linear(self.vision_embed_dim, self.vision_embed_dim * 2, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.vision_embed_dim * 2, self.projection_dim, bias=False),
                )
        else:
            self.visual_projection = None

        if fix_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # import pdb;pdb.set_trace()
        # if checkpoint is not None:
        #     pth = torch.load(checkpoint)
        #     self.load_state_dict(pth)

        # if resize_pos_embed:
        #     origin_p_num = image_size // patch_size
        #     new_p_num = resize_shape // patch_size

        #     origin_position_embedding_weight = self.vision_model.embeddings.position_embedding.weight
        #     origin_position_embedding_weight_cls = origin_position_embedding_weight[-1:]
        #     origin_position_embedding_weight = origin_position_embedding_weight[:-1].permute(1, 0).view(1, hidden_size, origin_p_num, origin_p_num)
        #     new_position_embedding_weight = F.interpolate(origin_position_embedding_weight, (new_p_num, new_p_num), mode='bilinear', align_corners=False)[0]
        #     new_position_embedding_weight = new_position_embedding_weight.flatten(-2).permute(1, 0)
        #     new_position_embedding_weight = torch.cat((new_position_embedding_weight, origin_position_embedding_weight_cls), dim=0)

        #     self.vision_model.embeddings.register_buffer("self.position_ids", torch.arange(new_p_num*new_p_num + 1).expand((1, -1)))
        #     self.vision_model.embeddings.position_embedding = nn.Embedding(new_p_num*new_p_num + 1, hidden_size)
        #     self.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(new_position_embedding_weight).to(origin_position_embedding_weight)

    def resize_abs_pos_embed(self, checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if 'vision_model.embeddings.position_embedding.weight' in state_dict:
            pos_embed_checkpoint = state_dict[
                'vision_model.embeddings.position_embedding.weight']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = self.vision_model.embeddings.position_embedding.weight.shape[-2] - \
                               self.vision_model.embeddings.num_patches
            orig_size = int(
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(self.vision_model.embeddings.num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=True)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat(
                    (extra_tokens, pos_tokens.squeeze(dim=0)), dim=0)
                state_dict[
                    'vision_model.embeddings.position_embedding.weight'] = new_pos_embed
                total_num = new_pos_embed.size(0)
                state_dict[
                    'vision_model.embeddings.position_ids'] = state_dict[
                                                                  'vision_model.embeddings.position_ids'][:, :total_num]
        return state_dict

    # def init_weights(self):
    #     if (isinstance(self.init_cfg, dict)
    #             and self.init_cfg['type'] == 'Pretrained'):
    #         logger = MMLogger.get_current_instance()
    #         checkpoint = _load_checkpoint(
    #             self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
    #         state_dict = self.resize_abs_pos_embed(checkpoint)
    #         del state_dict['visual_projection.weight']
    #         self.load_state_dict(state_dict, strict=False)
    #     # super(CLIPVisionModel, self).init_weights()
    #     if not (isinstance(self.init_cfg, dict)
    #             and self.init_cfg['type'] == 'Pretrained'):
    #         # if self.visual_projection:
    #         #     nn.init.normal_(
    #         #         self.visual_projection.weight,
    #         #         std=self.vision_embed_dim ** -0.5 * self.initializer_factor)
    #         for m in self.modules():
    #             if isinstance(m, CLIPVisionEmbeddings):
    #                 factor = self.initializer_factor
    #                 nn.init.normal_(
    #                     m.class_embedding,
    #                     mean=0.0,
    #                     std=m.embed_dim ** -0.5 * factor)
    #                 nn.init.normal_(
    #                     m.patch_embedding.weight,
    #                     std=self.initializer_range * factor)
    #                 nn.init.normal_(
    #                     m.position_embedding.weight,
    #                     std=self.initializer_range * factor)
    #             elif isinstance(m, CLIPAttention):
    #                 factor = self.initializer_factor
    #                 in_proj_std = (m.embed_dim ** -0.5) * (
    #                         (2 * self.num_hidden_layers) ** -0.5) * factor
    #                 out_proj_std = (m.embed_dim ** -0.5) * factor
    #                 nn.init.normal_(m.q_proj.weight, std=in_proj_std)
    #                 nn.init.normal_(m.k_proj.weight, std=in_proj_std)
    #                 nn.init.normal_(m.v_proj.weight, std=in_proj_std)
    #                 nn.init.normal_(m.out_proj.weight, std=out_proj_std)
    #             elif isinstance(m, CLIPMLP):
    #                 factor = self.initializer_factor
    #                 in_proj_std = ((self.hidden_size ** -0.5) *
    #                                ((2 * self.num_hidden_layers) ** -0.5) *
    #                                factor)
    #                 fc_std = (2 * self.hidden_size) ** -0.5 * factor
    #                 nn.init.normal_(m.fc1.weight, std=fc_std)
    #                 nn.init.normal_(m.fc2.weight, std=in_proj_std)
    #
    #             if isinstance(m, nn.LayerNorm):
    #                 m.bias.data.zero_()
    #                 m.weight.data.fill_(1.0)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self,
                pixel_values=None,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_features = None
        pooled_output = vision_outputs[1]
        if self.use_proj:
            # image_features = self.visual_projection(pooled_output)
            image_features = self.visual_projection(vision_outputs[0][:, 1:, :])

        return {
            'vision_outputs': vision_outputs,
            'proj_feature': image_features,
            'pooled_outputs': pooled_output
        }
