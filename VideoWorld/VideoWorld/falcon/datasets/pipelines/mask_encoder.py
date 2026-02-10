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
from mmcv.transforms import BaseTransform

from falcon.models.vbackbones.dvae_utils import ImageTokenizer
from falcon.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DVAEMaskEncoder(BaseTransform):

    def __init__(self,
                 down_size=False,
                 ):
        self.down_size = down_size
        self.mask_token = ImageTokenizer(down=down_size)

    def transform(self, results):

        mask_gt = results.get('pred_label', None)  # (w,h,3)
        if mask_gt is not None:
            mask_gt = torch.tensor(mask_gt).unsqueeze(dim=0).permute(0, 3, 1, 2)
            visual_mask_ids = self.mask_token(mask_gt)
            visual_mask_ids = visual_mask_ids.squeeze(dim=0).tolist()
            visual_mask_ids = str(visual_mask_ids)
            results['mask_ids'] = visual_mask_ids
        else:
            results['mask_ids'] = None

        return results
