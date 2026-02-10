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
import torch.nn.functional as F
from dall_e import load_model, map_pixels, unmap_pixels
from mmengine.model import BaseModule


class ImageTokenizer(BaseModule):

    def __init__(self,
                 down=False,
                 encoder_path='work_dirs/init/dall-e/encoder.pkl'):
        super().__init__()
        self.dalle_encoder = load_model(
            encoder_path, device=torch.device('cpu')).eval()
        self.dalle_encoder.requires_grad_(False)
        self.down = down

        # self.std_buffer = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer(
            'std_buffer',
            torch.tensor([0.26862954, 0.26130258,
                          0.27577711]).reshape(1, 3, 1, 1))
        self.register_buffer(
            'mean_buffer',
            torch.tensor([0.48145466, 0.4578275,
                          0.40821073]).reshape(1, 3, 1, 1))

        # self.token_embedding = nn.Embedding(8192, embedding)

    def init_weights(self):
        pass


    def unnormalize(self, imgs):
        # imgs *= self.std_buffer
        # imgs += self.mean_buffer
        imgs = imgs / 255.0
        imgs = F.max_pool2d(imgs, 2, 2)
        if self.down:
            imgs = F.max_pool2d(imgs, 2, 2)
        return imgs

    def forward(self, imgs):
        with torch.no_grad():
            imgs = self.unnormalize(imgs)
            imgs = map_pixels(imgs)
            z_logits = self.dalle_encoder(imgs)
            z = torch.argmax(z_logits, axis=1)

        z = z.flatten(1)
        # z_token = self.token_embedding(z)
        return z


class ImageTokenizerDecoder(nn.Module):
    def __init__(self,
                 output_size=224,
                 decoder_path='work_dirs/init/dall-e/decoder.pkl'):
        super().__init__()
        self.dalle_decoder = load_model(
            decoder_path, device=torch.device('cpu')).eval()
        self.dalle_decoder.requires_grad_(False)

        self.output_size = (output_size, output_size)

    def forward(self, z):
        x_stats = self.dalle_decoder(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = torch.nn.functional.interpolate(x_rec, size=self.output_size, mode='bicubic')
        return x_rec * 255.0
