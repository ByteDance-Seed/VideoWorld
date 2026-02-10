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

from falcon.registry import MODELS
from ..utils.accuracy import accuracy


@MODELS.register_module()
class ITGHead(BaseModule):
    def __init__(
            self, text_weight=1.0, ignore_index=-100, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.ulm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

        self.text_weight = text_weight

    def forward(self, text_pred, text_label, ):
        text_label = text_label.contiguous().view(-1)
        cap_pred = text_pred.contiguous()
        tg_loss = self.ulm_loss(
            cap_pred.view(-1, cap_pred.size(-1)), text_label)

        losses = {}
        labels_a = text_label[text_label != self.ignore_index]
        fusion_a = cap_pred.view(
            -1, cap_pred.size(-1))[text_label != self.ignore_index]
        losses['acc'] = accuracy(fusion_a, labels_a)
        losses['loss_text'] = tg_loss * self.text_weight


        return losses


@MODELS.register_module()
class ITGMixHead(BaseModule):
    def __init__(
            self, text_weight=1.0, ignore_index=-100, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.lm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.vlm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

        self.text_weight = text_weight

    def forward(self, v_pred, label, text_pred, text_label):
        text_label = text_label.contiguous().view(-1)
        cap_pred = text_pred[:, :-1].contiguous()
        tg_loss = self.lm_loss(
            cap_pred.view(-1, cap_pred.size(-1)), text_label)

        losses = {}
        labels_a = text_label[text_label != self.ignore_index]
        fusion_a = cap_pred.view(
            -1, cap_pred.size(-1))[text_label != self.ignore_index]
        # losses['acc'] = accuracy(fusion_a, labels_a)
        losses['loss_text'] = tg_loss * self.text_weight

        v_label = label.contiguous().view(-1)
        v_pred = v_pred.contiguous()
        v_loss = self.vlm_loss(
            v_pred.view(-1, v_pred.size(-1)), v_label)

        v_loss = torch.nan_to_num(v_loss, nan=0.0)
        labels_a = v_label[v_label != self.ignore_index]
        fusion_a = v_pred.view(
            -1, v_pred.size(-1))[v_label != self.ignore_index]
        # losses['acc_v'] = accuracy(fusion_a, labels_a)
        losses['loss_v'] = v_loss

        return losses
