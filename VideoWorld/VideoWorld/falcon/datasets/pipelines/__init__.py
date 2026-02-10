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
from mmcv.transforms import CenterCrop, LoadImageFromFile, RandomFlip, RandomResize, Resize

from falcon.registry import TRANSFORMS
from .auto_augment import (AutoAugment, AutoContrast, BaseAugTransform,
                           Brightness, ColorTransform, Contrast, Cutout,
                           Equalize, GaussianBlur, Invert, Posterize,
                           RandAugment, Rotate, Sharpness, Shear, Solarize,
                           SolarizeAdd, Translate)
from .formatting import Collect, ToNumpy, ToPIL, ToTensor, Transpose
from .loading import ImageReader, TransferData, LoadImageNetTextFromFile, LoadImageNetFromFile, LoadImageTextFromlmdb, \
    LoadVGTextFromFile, LoadiNaturaFromFile, LoadImageNetINalistFromFile, LoadImageClsTextFromlmdb, LoadCOCOFromFile, \
    LoadNYUDepthMask, LoadLlavaPre, LoadVISMask, LoadADE20KPseudoVideoMask
from .mask_encoder import DVAEMaskEncoder
from .processing import Normalize, RandomCrop, RandomResizedCrop, RandomErasing, MMResize, VideoMMResize, PseudoVideoRotate
from .prompt_gen import MaskPromptSelect, MaskPromptSelectADE20k, MaskPromptSelectNYU, MaskPromptSelectVIS
from .tokenizer import AutoTextTokenizer, LlamaTokenizer, LlamaTokenizerPrompt, BertTextTokenizer, TokenizerforGoImage, TokenizerforCALVIN, TokenizerforCALVINEnvVal
from .video_transform import *
for t in (CenterCrop, LoadImageFromFile, RandomFlip, RandomResize, Resize):
    TRANSFORMS.register_module(module=t)

__all__ = [
    'ToPIL', 'ToNumpy', 'Transpose', 'Collect', 'RandomCrop', 'RandomErasing',
    'RandomResizedCrop', 'CenterCrop', 'LoadImageFromFile', 'Normalize', 'MMResize', 'LoadLlavaPre',
    'AutoAugment', 'AutoContrast', 'Brightness', 'ColorTransform', 'Contrast', 'Cutout',
    'Equalize', 'RandAugment', 'MaskPromptSelect', 'MaskPromptSelectADE20k', 'MaskPromptSelectNYU',
    'RandomFlip', 'Resize', 'AutoTextTokenizer', 'LlamaTokenizer',
    'RandomResize', 'ToTensor', 'ImageReader', 'TransferData', 'LoadNYUDepthMask', 'VideoMMResize',
    'LlamaTokenizerPrompt', 'LoadImageNetFromFile', 'LoadImageNetTextFromFile',
    'LoadImageTextFromlmdb', 'LoadVGTextFromFile', 'LoadiNaturaFromFile', 'LoadImageNetINalistFromFile',
    'LoadImageClsTextFromlmdb', 'LoadCOCOFromFile', 'BertTextTokenizer', 'DVAEMaskEncoder',
    'LoadVISMask', 'MaskPromptSelectVIS', 'PseudoVideoRotate', 'LoadADE20KPseudoVideoMask', 
]
