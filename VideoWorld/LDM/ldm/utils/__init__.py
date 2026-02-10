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

from .cli import modify_args
from .img_utils import (all_to_tensor, can_convert_to_image, get_box_info,
                        reorder_image, tensor2img, to_numpy)
from .io_utils import MMAGIC_CACHE_DIR, download_from_url
# TODO replace with engine's API
from .logger import print_colored_log
from .sampler import get_sampler
from .setup_env import register_all_modules, try_import
from .trans_utils import (add_gaussian_noise, adjust_gamma, bbox2mask,
                          brush_stroke_mask, get_irregular_mask, make_coord,
                          random_bbox, random_choose_unknown)
from .typing import ConfigType, ForwardInputs, LabelVar, NoiseVar, SampleList
from .tar_dataloader import go_image_tar_decoder
__all__ = [
    'modify_args', 'print_colored_log', 'register_all_modules', 'try_import',
    'ForwardInputs', 'SampleList', 'NoiseVar', 'ConfigType', 'LabelVar',
    'MMAGIC_CACHE_DIR', 'download_from_url', 'get_sampler', 'tensor2img',
    'random_choose_unknown', 'add_gaussian_noise', 'adjust_gamma',
    'make_coord', 'bbox2mask', 'brush_stroke_mask', 'get_irregular_mask',
    'random_bbox', 'reorder_image', 'to_numpy', 'get_box_info',
    'can_convert_to_image', 'all_to_tensor'
]
