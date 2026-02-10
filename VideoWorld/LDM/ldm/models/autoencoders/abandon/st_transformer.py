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
from .st_transformer_utils import *
from ldm.registry import MODELS
from torch.nn import Module, ModuleList

import torch
import torch.nn.functional as F

from torch import nn, einsum, Tensor
import numpy as np
from typing import  Union, Tuple, Optional, List


def get_module(name : str):
    if name == 'space_attn':
        return SpatialAttention
    elif name == 'time_attn':
        return TemporalAttention
    elif name == 'space-time_attn':
        return SpaceTimeAttention
    # * Image modules
    elif name == 'blur_pool':
        return BlurPooling2d
    elif name == 'space_downsample':
        return SpaceDownsample
    elif name == 'image-residual':
        return ImageResidualBlock
    # * Video modules
    elif name == 'video-residual':
        return VideoResidualBlock
    elif name == 'causal-conv3d':
        return CausalConv3d
    elif name == 'causal-conv3d-transpose':
        return CausalConvTranspose3d
    elif name == 'depth2space_upsample':
        return DepthToSpaceUpsample
    elif name == 'depth2time_upsample':
        return DepthToTimeUpsample
    elif name == 'depth2spacetime_upsample':
        return DepthToSpaceTimeUpsample
    elif name == 'spacetime_downsample':
        return SpaceTimeDownsample
    # * Norm modules
    elif name == 'group_norm':
        return nn.GroupNorm
    elif name == 'adaptive_group_norm':
        return AdaptiveGroupNorm
    # * Activation modules
    elif name == 'gelu':
        return nn.GELU
    elif name == 'relu':
        return nn.ReLU
    elif name == 'leaky_relu':
        return nn.LeakyReLU
    elif name == 'silu':
        return nn.SiLU
    else:
        raise ValueError(f'Unknown module name: {name}')
    

def parse_blueprint(
    blueprint,
) -> Tuple[nn.ModuleList, List[bool]]:
    # Parse the blueprint
    layers = []
    ext_kw = []
    
    for desc in blueprint:            
        if isinstance(desc, str): desc = (desc, {})
        
        name, kwargs = default(desc, (None, {}))
        ext_kw.extend(
            [kwargs.pop('has_ext', False)] * kwargs.get('n_rep', 1)
        )
        layers.extend(
            [
                get_module(name)(**kwargs)
                for _ in range(kwargs.pop('n_rep', 1))
                if exists(name) and exists(kwargs)
            ]
        )
        
    return nn.ModuleList(layers), ext_kw


@MODELS.register_module()
class STTransformerEncoder(Module):
