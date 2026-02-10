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
import os
from falcon.models.vbackbones.dvae_utils import ImageTokenizer, ImageTokenizerDecoder
from falcon.models.quantizer.vqgan import VQGANDecoder, VQGANEncoder
from falcon.models.quantizer.connector import ConvConnector
from falcon.registry import MODELS
from .base import BaseModel
import mmcv
import numpy as np
import cv2
import sgfmill
import clip
import copy
import random
import matplotlib.pyplot as plt
import subprocess
import imageio
import json
import torch.nn.functional as F
import torch.nn as nn
import imageio

import torch.distributed as dist
from pathlib import Path
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class IDMResnet(ResNet):
    def __init__(self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=[False, False, False],
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            in_channels=3,
            stride=1,
            idm_hidden_size=None
        ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)
            # _log_api_usage_once(self)
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        # import pdb;pdb.set_trace()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
     
        # import pdb;pdb.set_trace()
        hidden_size = 512 * block.expansion
        if idm_hidden_size == 512:
            self.pred_act_mlps = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size//2),
                nn.Linear(hidden_size//2, hidden_size//4)])
            self.pred_arm_act = nn.Linear(hidden_size//4, 6) # arm action
            self.pred_gripper_act = nn.Linear(hidden_size//4, 1) # gripper action (binary)
        else:
            self.pred_act_mlps = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size//2),
                nn.Linear(hidden_size//2, hidden_size//2)])
            self.pred_arm_act = nn.Linear(hidden_size//2, 6) # arm action
            self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)

        self.fc = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # import pdb;pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # action_embedding = x[:, :, act_query_token_start_i]
        for pred_act_mlp in self.pred_act_mlps:
            x = pred_act_mlp(x)
        arm_action_preds = self.pred_arm_act(x)  # (b, l, act_dim - 1)
        gripper_action_preds = self.pred_gripper_act(x)  # (b, l, 1)

        return arm_action_preds, gripper_action_preds

@MODELS.register_module()
class VQInverseDynamicModel(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, mode='acc', battle_with_katago=None, kata_ana=False, vq_decoder_cfg={}, max_generate_length=300, idm_hidden_size=None, work_dir='/opt/tiger', worker_index=0):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

        # import pdb;pdb.set_trace()
        self.vbackbone = MODELS.build(vbackbone)
        self.vq_num = neck['vq_num']
        self.neck = None

        self.head = None
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
        #18: [2, 2, 2, 2]
        self.IDM = IDMResnet(Bottleneck, [2, 2, 2, 2], in_channels=24, num_classes=7, idm_hidden_size=idm_hidden_size)
        self.state_loss = nn.SmoothL1Loss()
        self.gripper_loss = nn.BCEWithLogitsLoss()

    def check_rec(self, visual_ids, visual_hand_ids, img, img_hand):
        # import pdb;pdb.set_trace()
        # num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        # size = int(visual_ids.shape[-1] **0.5)
        # if visual_hand_ids.ndim == 3:
        b, t = visual_hand_ids.shape[:2]
        visual_hand_ids = visual_hand_ids.flatten(0, 1)

        rec_visual_hand_ids = visual_hand_ids.reshape(-1, 4, 4) 
        rec_visual_hand_ids = self.v_token.indices_to_codes(rec_visual_hand_ids)
        rec_visual_hand_ids = self.pre_decode(rec_visual_hand_ids)
        rec_visual_hand_ids = self.v_token_decoder(rec_visual_hand_ids)
        rec_visual_hand = torch.clamp((rec_visual_hand_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_hand = rec_visual_hand.view(b, t, 256, 256, 3)

        rec_visual_ids = visual_ids.reshape(-1, 4, 4)
        rec_visual_ids = self.v_token.indices_to_codes(rec_visual_ids)
        rec_visual_ids = self.pre_decode(rec_visual_ids)
        rec_visual_ids = self.v_token_decoder(rec_visual_ids)
        rec_visual_ids = torch.clamp((rec_visual_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_ids = rec_visual_ids.view(b, t, 256, 256, 3)
        for idx, (bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand) in enumerate(zip(rec_visual_ids, img, rec_visual_hand, img_hand)):
            import cv2
            import numpy as np
            for fi, (rec_visual_id, gt_img, hand, gt_hand) in enumerate(zip(bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand)):
                gt_hand = torch.clamp(((gt_hand+1)*127), min=0, max=255)
                gt_hand = gt_hand.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                hand = hand.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                rec_visual_id = rec_visual_id.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                gt_img = torch.clamp(((gt_img+1)*127), min=0, max=255)
                gt_img = gt_img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                show0 = np.concatenate((gt_img, gt_hand), axis=0)
                show1 = np.concatenate((rec_visual_id, hand), axis=0)
                show = np.concatenate((show0, show1), axis=1)
                cv2.imwrite(f'/opt/tiger/rec_visualize/{fi}_{idx}.jpg', show)

    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        b, t, c, h, w = img.shape 
        actions = kwargs['action'] #B, T, 7
        img_hands = kwargs.get('hand', None) 
        codes, visual_ids = self.encode_image(img.flatten(0, 1)) #B, t, 16
        hand_codes, hand_visual_ids = self.encode_image(img_hands.flatten(0, 1)) #B, t, 16
        code_shape = hand_codes.shape[1]
        codes = codes.view(b, t, code_shape, 4, 4)
        hand_codes = hand_codes.view(b, t, code_shape, 4, 4)
        visual_ids = visual_ids.view(b, t, 4, 4)
        hand_visual_ids = hand_visual_ids.view(b, t, 4, 4)
        arm_action = actions[:, 0, :6] 
        gripper_action = actions[:, 0, -1:]
        gripper_action[gripper_action == -1] = 0
        # self.check_rec(visual_ids, hand_visual_ids, img, img_hands)
        idm_input = torch.cat((codes, hand_codes), 1).flatten(1, 2) #B, 2t, 16
        arm_action_preds, gripper_action_preds = self.IDM(idm_input)
        # arm_action = pred_action[:, :6]
        # gripper_action = pred_action[:, -1:]
  
        # gripper_action[gripper_action == -1] = 0

        loss_state = self.state_loss(arm_action_preds, arm_action)
        loss_gripper = self.gripper_loss(gripper_action_preds, gripper_action)
        losses = { "loss_state": loss_state, "loss_gripper": loss_gripper}

        return losses
    
    def check_val_loss(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        losses = self.forward_train(img, input_ids, pred_label, attention_mask, **kwargs)
        # print(losses)
        return [losses]

    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        return self.check_val_loss(img, input_ids, pred_label, attention_mask, **kwargs)
    
    def encode_image(self, img):
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        codes = self.v_token.indices_to_codes(visual_ids, project_out=False)

        
        # visual_ids = visual_ids 
        # visual_ids = visual_ids.flatten(1)

        return codes, visual_ids


@MODELS.register_module()
class RGBInverseDynamicModel(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, mode='acc', battle_with_katago=None, kata_ana=False, vq_decoder_cfg={}, max_generate_length=300, resnet_layer=[2, 2, 2, 2], idm_hidden_size=None, work_dir='/opt/tiger', worker_index=0):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

        # import pdb;pdb.set_trace()
        self.vbackbone = None

        self.neck = None

        self.head = None
        self.v_token = None

        #18: [2, 2, 2, 2]
        self.IDM = IDMResnet(Bottleneck, resnet_layer, in_channels=6, stride=2, num_classes=7, idm_hidden_size=idm_hidden_size)
        # mse = F.mse_loss()
        self.state_loss = nn.SmoothL1Loss()
        self.gripper_loss = nn.BCEWithLogitsLoss()

    def check_rec(self, visual_ids, visual_hand_ids, img, img_hand):
        # import pdb;pdb.set_trace()
        # num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        # size = int(visual_ids.shape[-1] **0.5)
        # if visual_hand_ids.ndim == 3:
        b, t = visual_hand_ids.shape[:2]
        visual_hand_ids = visual_hand_ids.flatten(0, 1)

        rec_visual_hand_ids = visual_hand_ids.reshape(-1, 4, 4) 
        rec_visual_hand_ids = self.v_token.indices_to_codes(rec_visual_hand_ids)
        rec_visual_hand_ids = self.pre_decode(rec_visual_hand_ids)
        rec_visual_hand_ids = self.v_token_decoder(rec_visual_hand_ids)
        rec_visual_hand = torch.clamp((rec_visual_hand_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_hand = rec_visual_hand.view(b, t, 256, 256, 3)

        rec_visual_ids = visual_ids.reshape(-1, 4, 4)
        rec_visual_ids = self.v_token.indices_to_codes(rec_visual_ids)
        rec_visual_ids = self.pre_decode(rec_visual_ids)
        rec_visual_ids = self.v_token_decoder(rec_visual_ids)
        rec_visual_ids = torch.clamp((rec_visual_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_ids = rec_visual_ids.view(b, t, 256, 256, 3)
        for idx, (bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand) in enumerate(zip(rec_visual_ids, img, rec_visual_hand, img_hand)):
            import cv2
            import numpy as np
            for fi, (rec_visual_id, gt_img, hand, gt_hand) in enumerate(zip(bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand)):
                gt_hand = torch.clamp(((gt_hand+1)*127), min=0, max=255)
                gt_hand = gt_hand.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                hand = hand.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                rec_visual_id = rec_visual_id.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                gt_img = torch.clamp(((gt_img+1)*127), min=0, max=255)
                gt_img = gt_img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                show0 = np.concatenate((gt_img, gt_hand), axis=0)
                show1 = np.concatenate((rec_visual_id, hand), axis=0)
                show = np.concatenate((show0, show1), axis=1)
                cv2.imwrite(f'/opt/tiger/rec_visualize/{fi}_{idx}.jpg', show)

    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        b, t, c, h, w = img.shape 
        actions = kwargs['action'] #B, T, 7
        img_hands = kwargs.get('hand', None) 
        rbg_mse = F.mse_loss(img[:, 0], img[:, 1], reduce=False) #b, 3, h, w
        rgb_hand_mse = F.mse_loss(img_hands[:, 0], img_hands[:, 1], reduce=False) #b, 3, h, w

        # codes, visual_ids = self.encode_image(img.flatten(0, 1)) #B, t, 16
        # hand_codes, hand_visual_ids = self.encode_image(img_hands.flatten(0, 1)) #B, t, 16
        
        arm_action = actions[:, 0, :6] 
        gripper_action = actions[:, 0, -1:]
        gripper_action[gripper_action == -1] = 0
        # self.check_rec(visual_ids, hand_visual_ids, img, img_hands)
        idm_input = torch.cat((rbg_mse, rgb_hand_mse), 1) #B, 6, h, w
        arm_action_preds, gripper_action_preds = self.IDM(idm_input)
        

        loss_state = self.state_loss(arm_action_preds, arm_action)
        loss_gripper = self.gripper_loss(gripper_action_preds, gripper_action)
        losses = { "loss_state": loss_state, "loss_gripper": loss_gripper}

        return losses
    def act_pred(self, imgs, img_hands):
        target_size = 224
        b, t, c, h, w = imgs.shape 
        if h != target_size:
            imgs = F.interpolate(imgs.flatten(0, 1).float(), (target_size, target_size), mode="bilinear", align_corners=False).to(imgs).view(b, t, c, target_size, target_size)
            img_hands = F.interpolate(img_hands.flatten(0, 1).float(), (target_size, target_size), mode="bilinear", align_corners=False).to(imgs).view(b, t, c, target_size, target_size)
        rbg_mse = F.mse_loss(imgs[:, 0], imgs[:, 1], reduce=False) #b, 3, h, w
        rgb_hand_mse = F.mse_loss(img_hands[:, 0], img_hands[:, 1], reduce=False) #b, 3, h, w
        idm_input = torch.cat((rbg_mse, rgb_hand_mse), 1) #B, 6, h, w
        arm_action_preds, gripper_action_preds = self.IDM(idm_input)
        return arm_action_preds, gripper_action_preds

    def check_val_loss(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        losses = self.forward_train(img, input_ids, pred_label, attention_mask, **kwargs)
        # print(losses)
        return [losses]

    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        return self.check_val_loss(img, input_ids, pred_label, attention_mask, **kwargs)
    
    def encode_image(self, img):
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        codes = self.v_token.indices_to_codes(visual_ids, project_out=False)

        
        # visual_ids = visual_ids 
        # visual_ids = visual_ids.flatten(1)

        return codes, visual_ids


@MODELS.register_module()
class FeatInverseDynamicModel(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, mode='acc', battle_with_katago=None, kata_ana=False, vq_decoder_cfg={}, max_generate_length=300, idm_hidden_size=None, work_dir='/opt/tiger', worker_index=0):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

        # import pdb;pdb.set_trace()
        self.vbackbone = MODELS.build(vbackbone)
        self.vq_num = neck['vq_num']
        self.neck = None

        self.head = None
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
        #18: [2, 2, 2, 2]
        self.arm_IDM = MLP(512, 512, 6, 3)
        self.gripper_IDM = MLP(512, 512, 1, 3)

        self.state_loss = nn.SmoothL1Loss()
        self.gripper_loss = nn.BCEWithLogitsLoss()

    

    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        encode_feat = kwargs['encode_feat']
        actions = kwargs['action']
        arm_action_preds = self.arm_IDM(encode_feat[:, :, 0])
        gripper_action_preds = self.gripper_IDM(encode_feat[:, :, 0])
        arm_action = actions[:, 0, 0, :6] * 50
        gripper_action = actions[:, 0, 0, -1:] * 20
        gripper_action[gripper_action == -1] = 0
        # idm_input = torch.cat((codes, hand_codes), 1).flatten(1, 2) #B, 2t, 16
        
        # arm_action = pred_action[:, :6]
        # gripper_action = pred_action[:, -1:]
  
        # gripper_action[gripper_action == -1] = 0

        loss_state = self.state_loss(arm_action_preds, arm_action)
        loss_gripper = self.gripper_loss(gripper_action_preds, gripper_action)
        losses = { "loss_state": loss_state, "loss_gripper": loss_gripper}

        return losses
    
    def check_val_loss(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        # import pdb;pdb.set_trace()
        losses = self.forward_train(img, input_ids, pred_label, attention_mask, **kwargs)
        # print(losses)
        return [losses]

    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        return self.check_val_loss(img, input_ids, pred_label, attention_mask, **kwargs)
    
    def encode_image(self, img):
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        codes = self.v_token.indices_to_codes(visual_ids, project_out=False)

        
        # visual_ids = visual_ids 
        # visual_ids = visual_ids.flatten(1)

        return codes, visual_ids