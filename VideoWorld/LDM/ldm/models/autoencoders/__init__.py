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
from .vqgan import *
from .l2vqgan import VQGANEncoderL2, VQGANDecoderL2
from .magvit import Magvitencoder,Magvitdecoder
from .magvit_v2 import (MagvitV2decoder,MagvitV2encoder,MagvitV2Adadecoder,MagvitV2Gencoder,MagvitV2Gdecoder,
                        MagvitV2AdaAttdecoder, MagvitV2Attencoder)
from .magvit_v2_2d import MagvitV2encoder2D, MagvitV2Adadecoder2D
from .magvit_v2_ldm import MagvitV2LAencoder, MagvitV2LAAdadecoder

__all__=[
    'VQGANEncoder', 'VQGANDecoder', 'VQGANAdaDecoder','Magvitencoder', 'Magvitdecoder','MagvitV2decoder','MagvitV2encoder',
    'MagvitV2Adadecoder', 'MagvitV2Gencoder', 'MagvitV2Gdecoder', 'VQGANEncoderL2', 'VQGANDecoderL2',
    'MagvitV2Attencoder', 'MagvitV2AdaAttdecoder','MagvitV2encoder2D', 'MagvitV2Adadecoder2D'
]