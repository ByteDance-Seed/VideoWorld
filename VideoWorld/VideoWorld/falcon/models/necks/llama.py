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
import transformers
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.model import BaseModule
from transformers.models.llama import LlamaForCausalLM
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
from falcon.registry import MODELS

@MODELS.register_module()
class LlamaGenModel(BaseModule):
    def __init__(self, pretrain_path, vq_num, sepcial_token_num, use_text=True, init_cfg=None, use_focal_loss=False):
        super().__init__(init_cfg=init_cfg)
        # import pdb;pdb.set_trace()
        # if init_cfg is not None and init_cfg['type'] == 'Pretrained':
        # pretrain_path = init_cfg.checkpoint
        # model_args = {
        #     'max_position_embeddings': max_position_embeddings,
        #     'ignore_mismatched_sizes':True,
        #     # 'gradient_checkpointing': gradient_checkpointing,
        # }
        # self.max_position_embeddings = max_position_embeddings
        self.vq_num = vq_num
        self.sepcial_token_num = sepcial_token_num
        # import pdb;pdb.set_trace()
        self.llm = LlamaForCausalLM.from_pretrained(pretrain_path) 
        if use_text:
            self.llm.resize_token_embeddings(self.llm.get_input_embeddings().num_embeddings + sepcial_token_num + vq_num)
        else:
            self.llm.resize_token_embeddings(sepcial_token_num + vq_num)
        # self.resize_pe_embeddings(max_position_embeddings)

    def generate(self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_length: int = 512,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            eos_token_id=50256,
            index=None):
        # import pdb;pdb.set_trace()
        # max_length = self.max_position_embeddings
        # print(input_ids)
        return self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, eos_token_id=eos_token_id)
        # return self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=1, eos_token_id=eos_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        inputs_embeds=None, 
        return_dict=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # import pdb;pdb.set_trace()
        output = self.llm(input_ids, attention_mask=attention_mask, labels=labels, inputs_embeds=inputs_embeds, return_dict=return_dict, output_hidden_states=return_dict)
        if return_dict:
            return output['logits'], output['loss'], output['hidden_states'][-1]
        return output['logits'], output['loss']

if __name__ == '__main__':
    model = LlamaGenModel('/mnt/bn/panxuran/MyTinyLlama', 64000, 3)
    import pdb;pdb.set_trace()