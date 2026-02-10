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
from mmengine.model import (BaseModule, normal_init, update_init_info,
                            xavier_init)
from ldm.registry import MODELS
from ..utils import get_module_device


@MODELS.register_module()
class VQGenerator(BaseModule):
    def __init__(self,
                 encoder,
                 post_encoder,
                 quantizer,
                 pre_decoder,
                 decoder):
        super().__init__()


        self.encoder = MODELS.build(encoder)

        self.post_encoder = MODELS.build(post_encoder)

        self.quantizer = MODELS.build(quantizer)

        self.pre_decoder = MODELS.build(pre_decoder)

        self.decoder = MODELS.build(decoder)



    def quantizer_image(self,image):
        z = self.post_encoder(self.encoder(image))

        codes,indice = self.quantizer(z)
        return codes,indice


    def decoder_image(self,codes):
        pred_image = self.decoder(self.pre_decoder(codes))
        return pred_image


    def forward(self,image):
        # import pdb;pdb.set_trace()
        codes,indice = self.quantizer_image(image)
        pred_image = self.decoder_image(codes)
        return pred_image,codes,indice




@MODELS.register_module()
class VQLatentActionGenerator(BaseModule):
    def __init__(self,
                 encoder,
                 post_encoder,
                 quantizer,
                 pre_decoder,
                 decoder,
                 z_encoder=None,
                 z_post_encoder=None,
                 z_quantizer=None,
                 z_pre_decoder=None,
                 z_decoder=None):
        super().__init__()


        self.encoder = MODELS.build(encoder)

        self.post_encoder = MODELS.build(post_encoder)

        self.quantizer = MODELS.build(quantizer)

        self.pre_decoder = MODELS.build(pre_decoder)

        self.decoder = MODELS.build(decoder)

        if z_encoder is not None:
            self.z_encoder = MODELS.build(z_encoder)
            for p in self.z_encoder.parameters():
                p.requires_grad = False
        else:
            self.z_encoder = None
        
        if z_post_encoder is not None:
            self.post_z_encoder = MODELS.build(z_post_encoder)
            for p in self.post_z_encoder.parameters():
                p.requires_grad = False
        else:
            self.post_z_encoder = None

        if z_quantizer is not None:
            self.z_quantizer = MODELS.build(z_quantizer)
            for p in self.z_quantizer.parameters():
                p.requires_grad = False
        else:
            self.z_quantizer = None

      
    
        # self.pre_z_decoder = MODELS.build(z_pre_decoder) if z_pre_decoder is not None else None
        # self.z_decoder = MODELS.build(z_decoder) if z_decoder is not None else None
        
       

    def quantizer_image(self,image):
        z = self.post_encoder(self.encoder(image))

        codes,indice = self.quantizer(z)
        return codes,indice


    def decoder_image(self,codes, image):
        pred_image = self.decoder(self.pre_decoder(codes), image=image)
        return pred_image

    def produce_z(self, image):
    
        z = self.post_z_encoder(self.z_encoder(image))
        codes, indice = self.z_quantizer(z)
        # import pdb;pdb.set_trace()
        # pred_image = self.z_decoder(self.pre_z_decoder(codes))
        # import cv2
        # import numpy as np
        # cv2.imwrite('/opt/tiger/mmagicinit/test.jpg', torch.clamp((pred_image[0] + 1) * 127, min=0, max=255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
        # cv2.imwrite('/opt/tiger/mmagicinit/test_true.jpg', torch.clamp((image[0] + 1) * 127, min=0, max=255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
        return codes, indice

    def forward(self,image):
        # self.produce_z(image)
        if self.z_encoder is not None:
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                b, _, t, _, _ = image.shape
                image = image.permute(0, 2, 1, 3, 4).flatten(0, 1)  #bt, c, h, w
                z_codes, z_indice = self.produce_z(image) #z_codes: b*t, 256, h, w
                _, c, h, w = z_codes.shape
                image = z_codes.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        # import pdb;pdb.set_trace()
        codes,indice = self.quantizer_image(image)
        # import pdb;pdb.set_trace()
        pred_image = self.decoder_image(codes, image)
        # import pdb;pdb.set_trace()
        return pred_image,codes,indice

