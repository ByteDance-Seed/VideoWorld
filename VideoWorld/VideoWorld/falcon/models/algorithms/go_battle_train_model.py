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
import torch.nn.functional as F
import os
from falcon.models.vbackbones.dvae_utils import ImageTokenizer, ImageTokenizerDecoder
from falcon.models.quantizer.vqgan import VQGANDecoder, VQGANEncoder
from falcon.models.quantizer.connector import ConvConnector
from falcon.registry import MODELS
from .base import BaseModel
import numpy as np
import cv2
import sgfmill
import mmcv
import copy
IMAGE_TOKEN_INDEX = -200
MASK_TOKEN_INDEX = -300
import random
import matplotlib.pyplot as plt
import subprocess
import imageio
import json

@MODELS.register_module()
class VideoWorldGoBattleTrainModel(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, mode='acc', battle_with_katago=None, kata_ana=False, vq_decoder_cfg={}, max_generate_length=300, work_dir='/opt/tiger', worker_index=0):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

        # import pdb;pdb.set_trace()
        self.vbackbone = MODELS.build(vbackbone)
        self.vq_num = neck['vq_num']
        self.neck = MODELS.build(neck)

        self.head = MODELS.build(head)
        self.v_token = MODELS.build(quantizer)

        self.post_encode = ConvConnector(in_channels=256, out_channels=256)
        self.pre_decode = ConvConnector(in_channels=256, out_channels=256)
        self.v_token_decoder = VQGANDecoder(**vq_decoder_cfg)
        self.pred_image = pred_image
        self.pred_action = pred_action
        self.work_dir = work_dir
        self.max_generate_length = max_generate_length
        # import pdb;pdb.set_trace()

        state_dict = torch.load(vbackbone['init_cfg']['checkpoint'])['state_dict']
        encoder_state_dict = {}
        decoder_state_dict = {}
        post_encode_state_dict = {}
        pre_decode_state_dict = {}
        quantizer_state_dict = {}
        for k in state_dict:
            if 'generator.encoder' in k:
                encoder_state_dict[k[len('generator.encoder.'):]] = state_dict[k]
            if 'generator.decoder' in k:
                key = k[len('generator.encoder.'):]
                decoder_state_dict[key] = state_dict[k]
            if 'generator.pre_decoder' in k:
                pre_decode_state_dict[k[len('generator.pre_decoder.'):]] = state_dict[k]
            if 'generator.post_encoder' in k:
                post_encode_state_dict[k[len('generator.post_encoder.'):]] = state_dict[k]
            if 'generator.quantizer' in k:
                quantizer_state_dict[k[len('generator.quantizer.'):]] = state_dict[k]
        # import pdb;pdb.set_trace()
        self.vbackbone.load_state_dict(encoder_state_dict)
        self.v_token_decoder.load_state_dict(decoder_state_dict)
        self.v_token.load_state_dict(quantizer_state_dict)
        self.post_encode.load_state_dict(post_encode_state_dict)
        self.pre_decode.load_state_dict(pre_decode_state_dict)

        for p in self.v_token.parameters():
            p.requires_grad = False
        for p in self.vbackbone.parameters():
            p.requires_grad = False
        for p in self.v_token_decoder.parameters():
            p.requires_grad = False
        for p in self.post_encode.parameters():
            p.requires_grad = False
        for p in self.pre_decode.parameters():
            p.requires_grad = False

        self.mode = mode
        self.board_size = 19
        self.sub_board_size = 9


        
    def check_rec(self, visual_ids, visual_mask_ids, img):
        # import pdb;pdb.set_trace()
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        size = int(visual_ids.shape[-1] **0.5)
        rec_visual_mask_ids = visual_mask_ids.reshape(-1, size, size) - num_text_embeddings
        rec_visual_mask_ids = self.v_token.indices_to_codes(rec_visual_mask_ids)
        rec_visual_mask_ids = self.pre_decode(rec_visual_mask_ids)
        rec_visual_mask_ids = self.v_token_decoder(rec_visual_mask_ids)
        rec_visual_mask = torch.clamp((rec_visual_mask_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)

        rec_visual_ids = visual_ids.reshape(-1, size, size) - num_text_embeddings
        rec_visual_ids = self.v_token.indices_to_codes(rec_visual_ids)
        rec_visual_ids = self.pre_decode(rec_visual_ids)
        rec_visual_ids = self.v_token_decoder(rec_visual_ids)
        rec_visual_ids = torch.clamp((rec_visual_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        for idx, (rec_visual_id, mask) in enumerate(zip(rec_visual_ids, rec_visual_mask)):
            import cv2
            import numpy as np
            rec_visual_id = rec_visual_id.cpu().numpy()[:, :, ::-1].astype(np.uint8)
            mask = mask.cpu().numpy()[:, :, ::-1].astype(np.uint8)
            show = np.concatenate((mask, rec_visual_id), axis=0)
            cv2.imwrite(f'/opt/tiger/rec_visualize/test_{idx}.jpg', show)

    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
            pred_label = torch.stack(pred_label)
            img = torch.stack(img)
            attention_mask = torch.stack(attention_mask)
            invalid = torch.stack(kwargs.get('invalid'))
        # self.visualize(img, pred_label.to(img))
        # if len(img.shape) == 4:
        b, c, h, w = img.shape
        # visual_ids = torch.tensor([[112153, 111769, 112177, 112201, 113207,  78422,  79446, 104519,  88598,80495,  56720, 107080, 113223, 104519, 106631, 107079],[114227, 112209, 112152, 112200,  56688,  55833, 104078, 107078,  88430,111762,  82511, 104519, 112199, 107078, 107072, 107079],[112208,  86608, 112201, 113773,  87574, 106638, 107086, 107078,  86101,107086, 107087, 107079,  61003, 104518, 107072, 107078],[ 61881,  59000, 101911, 112200, 114198,  80749,  80471, 112198, 113206,55182,  60272,  81480, 112198, 105543, 109127, 107070]]).to(input_ids)
        visual_ids = self.encode_image(img)

        if pred_label is not None:
            visual_mask_ids = self.encode_image(pred_label)
            # visual_mask_ids = viacsual_ids
            visual_mask_ids = visual_mask_ids.detach()
        else:
            visual_mask_ids = None
        input_ids, attention_mask, labels = self.prepare_input(visual_ids, input_ids, attention_mask, visual_mask_ids)

        # self.check_rec(visual_ids, visual_mask_ids, img)
        visual_ids = visual_ids.detach()

        attn_length = [torch.where(attn_m != 0)[0].max() for attn_m in attention_mask]
        max_attn = max(attn_length)
        input_ids = input_ids[:, :(max_attn+2)]
        attention_mask = attention_mask[:, :(max_attn+2)]
        labels = labels[:, :(max_attn+2)]

        for i, label in enumerate(labels):
            if attn_length[i] + 1 >= 1024:
                continue
            label[(attn_length[i] + 2):] = -100
            attention_mask[i, attn_length[i] + 1] = 1
            if invalid[i] == 1:
                label = -100
        
        logits, loss, _ = self.neck(input_ids, attention_mask=attention_mask, labels=labels)
        # print("--------", input_ids[0], input_ids[1], "--------")
        # import pdb;pdb.set_trace()
        losses = {'losses_v': loss}

        return losses


    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        # import pdb;pdb.set_trace()
        return [{'eva_dict':{}}]
       
       
    def encode_image(self, img):
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        visual_ids = visual_ids + num_text_embeddings
        visual_ids = visual_ids.flatten(1)

        return visual_ids

    def prepare_input(self, visual_ids, input_ids, attention_mask, visual_mask_ids=None):
        new_input_ids = []
        new_attention_masks = []
        new_labels = []
        visual_mask_ids = torch.zeros((len(visual_ids), 0)).to(visual_ids) if visual_mask_ids is None else visual_mask_ids
        for bz, (visual_id, visual_mask_id, input_id) in enumerate(zip(visual_ids, visual_mask_ids, input_ids)):
            if visual_mask_id.ndim != 2:
                visual_mask_id = visual_mask_id[None]
            if visual_id.ndim != 2:
                visual_id = visual_id[None]
            cur_new_input_ids = []
            cur_new_attention_masks = []
            cur_new_label = []
            bz_attention_mask = attention_mask[bz]
            image_token_indices = torch.where((input_id == IMAGE_TOKEN_INDEX) | (input_id == MASK_TOKEN_INDEX))[0]
            i_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
            m_token_indices = torch.where(input_id == MASK_TOKEN_INDEX)[0]

            cur_image_idx = 0
            cur_mask_idx = 0
            while image_token_indices.numel() > 0:
                image_token_start = image_token_indices[0]
                if image_token_start in i_token_indices:
                    cur_visual_id = visual_id[cur_image_idx]
                    cur_image_idx += 1
                else:
                    cur_visual_id = visual_mask_id[cur_mask_idx]
                    cur_mask_idx += 1

                cur_attention_mask = torch.ones_like(cur_visual_id)
                cur_label = torch.ones_like(cur_visual_id) * -100


                cur_new_input_ids.append(input_id[:image_token_start])
                cur_new_input_ids.append(cur_visual_id)
                cur_new_input_ids.append(input_id[image_token_start + 1 : image_token_start + 2])


                cur_new_attention_masks.append(bz_attention_mask[:image_token_start])
                cur_new_attention_masks.append(cur_attention_mask)
                cur_new_attention_masks.append(bz_attention_mask[image_token_start + 1 : image_token_start + 2])

                if image_token_start in m_token_indices:
                    # cur_new_label.append(torch.ones_like(bz_attention_mask[:image_token_start - 1]))
                    cur_new_label.append(input_id[:image_token_start - 1])
                    cur_new_label.append(input_id[image_token_start - 1:image_token_start])
                    cur_new_label.append(cur_visual_id)
                    cur_new_label.append(input_id[image_token_start + 1 : image_token_start + 2])
                else:
                    cur_new_label.append(torch.ones_like(bz_attention_mask[:image_token_start]) * -100)
                    cur_new_label.append(cur_label)
                    cur_new_label.append(torch.ones_like(bz_attention_mask[image_token_start + 1 : image_token_start + 2]) * -100)

                input_id = input_id[image_token_start + 2 :]
                bz_attention_mask = bz_attention_mask[image_token_start + 2 :]
                image_token_indices = torch.where((input_id == IMAGE_TOKEN_INDEX) | (input_id == MASK_TOKEN_INDEX))[0]
                i_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
                m_token_indices = torch.where(input_id == MASK_TOKEN_INDEX)[0]

            if input_id.numel() > 0:
                cur_new_input_ids.append(input_id)
                cur_new_attention_masks.append(bz_attention_mask)
                cur_new_label.append(input_id)
            # cur_new_labels.append()

            cur_new_input_ids = torch.cat(cur_new_input_ids)
            cur_new_attention_masks = torch.cat(cur_new_attention_masks)
            cur_new_label = torch.cat(cur_new_label)

            new_labels.append(cur_new_label)
            new_input_ids.append(cur_new_input_ids)
            new_attention_masks.append(cur_new_attention_masks)


        input_ids = torch.stack(new_input_ids)
        attention_mask = torch.stack(new_attention_masks)
        visual_ids = visual_ids.detach()
        labels = torch.stack(new_labels)

        return input_ids, attention_mask, labels

    