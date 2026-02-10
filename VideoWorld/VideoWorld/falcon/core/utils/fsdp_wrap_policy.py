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
import torch.nn as nn
from mmengine.registry import FUNCTIONS
from torch.distributed.fsdp.wrap import _module_wrap_policy
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from falcon.models.lbackbones.utils import CLIPEncoderLayer
from falcon.models.vbackbones.beit import BEiTTransformerEncoderLayer
from falcon.models.vbackbones.mae_vit import TransformerEncoderLayer
from falcon.models.vbackbones.simple_visual_embed import CBlock


@FUNCTIONS.register_module()
def llama_auto_wrap_policy(module: nn.Module, recurse: bool,
                           nonwrapped_numel: int):
    name_list = ['LlamaDecoderLayer', 'CLIPEncoder', 'TransformerEncoderLayer', 'BEiTTransformerEncoderLayer']
    transformer_cls_to_wrap = set()
    for name in name_list:
        if name == LlamaDecoderLayer.__name__:
            transformer_cls_to_wrap.add(LlamaDecoderLayer)
        elif name == CLIPEncoderLayer.__name__:
            transformer_cls_to_wrap.add(CLIPEncoderLayer)
        elif name == TransformerEncoderLayer.__name__:
            transformer_cls_to_wrap.add(TransformerEncoderLayer)
        elif name == BEiTTransformerEncoderLayer.__name__:
            transformer_cls_to_wrap.add(BEiTTransformerEncoderLayer)
        elif name == CBlock.__name__:
            transformer_cls_to_wrap.add(CBlock)

    return _module_wrap_policy(module, recurse, nonwrapped_numel,
                               transformer_cls_to_wrap)

@FUNCTIONS.register_module()
def gpt2_auto_wrap_policy(module: nn.Module, recurse: bool,
                           nonwrapped_numel: int):
    # print("_______________in policy______")
    name_list = ['GPT2Block', 'CLIPEncoder', 'TransformerEncoderLayer', 'BEiTTransformerEncoderLayer']
    transformer_cls_to_wrap = set()
    for name in name_list:
        if name == GPT2Block.__name__:
            # print('___wrapped___')
            transformer_cls_to_wrap.add(GPT2Block)
        # elif name == CLIPEncoderLayer.__name__:
        #     transformer_cls_to_wrap.add(CLIPEncoderLayer)
        # elif name == TransformerEncoderLayer.__name__:
        #     transformer_cls_to_wrap.add(TransformerEncoderLayer)
        # elif name == BEiTTransformerEncoderLayer.__name__:
        #     transformer_cls_to_wrap.add(BEiTTransformerEncoderLayer)
        # elif name == CBlock.__name__:
        #     transformer_cls_to_wrap.add(CBlock)

    return _module_wrap_policy(module, recurse, nonwrapped_numel,
                               transformer_cls_to_wrap)

@FUNCTIONS.register_module()
def stablelm_auto_wrap_policy(module: nn.Module, recurse: bool,
                              nonwrapped_numel: int):
    name_list = ['GPTNeoXLayer', 'CLIPEncoder']
    transformer_cls_to_wrap = set()
    for name in name_list:
        if name == GPTNeoXLayer.__name__:
            transformer_cls_to_wrap.add(GPTNeoXLayer)
        elif name == CLIPEncoderLayer.__name__:
            transformer_cls_to_wrap.add(CLIPEncoderLayer)

    return _module_wrap_policy(module, recurse, nonwrapped_numel,
                               transformer_cls_to_wrap)
