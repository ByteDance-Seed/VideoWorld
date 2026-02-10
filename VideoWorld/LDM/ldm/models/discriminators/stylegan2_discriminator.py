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

from typing import Optional

import mmengine
import numpy as np

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner.amp import autocast
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch import Tensor
from .stylegan2_utils import ConvDownLayer,ResBlock,ModMBStddevLayer,EqualLinearActModule

from ldm.registry import MODELS



@MODELS.register_module('StyleGANv2Discriminator')
@MODELS.register_module()
class StyleGAN2Discriminator(BaseModule):
    def __init__(self,
                 in_size,
                 img_channels=3,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 out_fp32=True,
                 convert_input_fp32=True,
                 input_bgr2rgb=False,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg=init_cfg)
        self.num_fp16_scale = num_fp16_scales
        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32
        self.out_fp32 = out_fp32

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))
        in_channels = channels[in_size]

        _use_fp16 = num_fp16_scales > 0 or fp16_enabled
        convs = [
            ConvDownLayer(
                img_channels, channels[in_size], 1, fp16_enabled=_use_fp16)
        ]

        for i in range(log_size,2,-1):
            out_channel = channels[2 ** (i - 1)]

            # add fp16 training for higher resolutions
            _use_fp16 = (log_size - i) < num_fp16_scales or fp16_enabled

            convs.append(
                ResBlock(
                    in_channels,
                    out_channel,
                    blur_kernel,
                    fp16_enabled=_use_fp16,
                    convert_input_fp32=convert_input_fp32))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)
        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(
            in_channels + 1, channels[4], 3, fp16_enabled=fp16_enabled)

        final_linear_out_channels = 1

        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                channels[4] * 4 * 4,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], final_linear_out_channels),
        )

        self.input_bgr2rgb = input_bgr2rgb
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmengine.print_log(f'Load pretrained model from {ckpt_path}')

    def forward(self, x: Tensor, label: Optional[Tensor] = None):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            label (torch.Tensor, optional): The conditional input feed to
                mapping layer. Defaults to None.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        # This setting was used to finetune on converted weights
        if self.input_bgr2rgb:
            x = x[:, [2, 1, 0], ...]

        # convs has own fp-16 controller, do not wrap here
        x = self.convs(x)
        x = self.mbstd_layer(x)

        fp16_enabled = (
                self.final_conv.fp16_enabled or not self.convert_input_fp32)

        with autocast(enabled=fp16_enabled):
            if not fp16_enabled:
                x = x.to(torch.float32)
            x = self.final_conv(x)
            x = x.view(x.shape[0], -1)
            x = self.final_linear(x)
        return x


@MODELS.register_module()
class ADAStyleGAN2Discriminator(StyleGAN2Discriminator):

    def __init__(self, in_size, *args, data_aug=None, **kwargs):
        """StyleGANv2 Discriminator with adaptive augmentation.

        Args:
            in_size (int): The input size of images.
            data_aug (dict, optional): Config for data
                augmentation. Defaults to None.
        """
        super().__init__(in_size, *args, **kwargs)
        self.with_ada = data_aug is not None and data_aug != dict()
        if self.with_ada:
            self.ada_aug = MODELS.build(data_aug)
            self.ada_aug.requires_grad = False
        self.log_size = int(np.log2(in_size))

    def forward(self, x):
        """Forward function."""
        if self.with_ada:
            x = self.ada_aug.aug_pipeline(x)
        return super().forward(x)
