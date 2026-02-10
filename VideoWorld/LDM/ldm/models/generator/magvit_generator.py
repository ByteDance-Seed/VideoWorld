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
from torch import Tensor
from mmengine.model import (BaseModule, normal_init, update_init_info,
                            xavier_init)
from ldm.registry import MODELS
from typing import Optional
import torch.nn.functional as F
@MODELS.register_module()
class MagVitGenerator(BaseModule):
    def __init__(self,
                 encoder,
                 quantizer,
                 decoder,
                 freeze_encoder=False,
                 use_idm=False
                 ):
        super().__init__()


        self.encoder = MODELS.build(encoder)


        self.quantizer = MODELS.build(quantizer)

        self.decoder = MODELS.build(decoder)

        if freeze_encoder:
            for n, p in self.encoder.named_parameters():
                p.requires_grad = False
                if any([x in n for x in ["act_embedding", "qformer"]]):
                    p.requires_grad = True
        self.use_idm = use_idm
        if use_idm:
            hidden_size = 512
            self.pred_act_mlps = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size//2),
                nn.Linear(hidden_size//2, hidden_size//2)])
            self.pred_arm_act = nn.Linear(hidden_size//2, 6) # arm action
            self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
        # import pdb;pdb.set_trace()
        # total_num = sum(p.numel() for p in self.decoder.parameters())

    def quantizer_image(self,video_or_images: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True, return_feat=False):
        x, cond,video_contains_first_frame, pre_encode_out = self.encoder(video_or_images, cond, video_contains_first_frame)

        codes, indice = self.quantizer(x)
        return codes,indice,cond,video_contains_first_frame, pre_encode_out, x 
        # return codes,indice,cond,video_contains_first_frame, pre_encode_out, x if return_feat else None


    def decoder_image(self,codes,cond, video_contains_first_frame, video_or_images=None, pre_encode_out=None):
        pred_image = self.decoder(codes,cond, video_contains_first_frame, video_or_images, pre_encode_out)
        return pred_image
    def forward(self,video_or_images: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True, return_feat=False):
        codes,indice,cond,video_contains_first_frame, pre_encode_out, encode_feat = self.quantizer_image(video_or_images, cond, video_contains_first_frame, return_feat)
        recon_video = self.decoder_image(codes,cond, video_contains_first_frame, video_or_images, pre_encode_out)
        if self.use_idm:
            act_feat = encode_feat.squeeze() #B, 512
            action_embedding = act_feat
            # import pdb;pdb.set_trace()
            for i, pred_act_mlp in enumerate(self.pred_act_mlps):
                action_embedding = pred_act_mlp(action_embedding)
                action_embedding = F.relu(action_embedding) if i==0 else action_embedding
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, l, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, l, 1)
            return recon_video,codes,indice, encode_feat, (arm_action_preds, gripper_action_preds) 
            
        return recon_video,codes,indice, encode_feat, None
        # if return_feat:
        #     return recon_video,codes,indice, encode_feat
        # else:
        #     return recon_video,codes,indice, None






