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
import random
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from transformers import AutoTokenizer, GPT2Tokenizer

from falcon.registry import TRANSFORMS

DEFAULT_FRAME_INTERVAL = "<next>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_MASK_TOKEN = "<mask>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_MASK_START_TOKEN = "<mask_start>"
DEFAULT_MASK_END_TOKEN = "<mask_end>"
DEFAULT_ACT_START_TOKEN = "<act_start>"
DEFAULT_ACT_END_TOKEN = "<act_end>"
DEFAULT_ACT_TOKEN = "<act>"
class ImagePrompt:

    @property
    def prefix_size(self):
        return len(self.prefix)

    @property
    def postfix_size(self):
        return len(self.postfix)

    def sample(self):
        pre = self.prefix[random.randrange(self.prefix_size)]
        post = self.postfix[random.randrange(self.postfix_size)]
        return pre, post

    def set_max_length(self, tokenizer):
        self.pre_max_length = max([
            tokenizer(x, return_tensors='pt')['input_ids'].shape[-1]
            for x in self.prefix
        ])
        self.post_max_length = max([
            tokenizer(x, return_tensors='pt')['input_ids'].shape[-1]
            for x in self.postfix
        ])


@dataclass
class CaptaionPrompt(ImagePrompt):
    """Image prompts for image captaining.

    An example has the following format: <s> {prefix} <i> {image tokens} </i>
    {postfix} {caption} </s>
    """

    prefix = ['The image is given by:']
    postfix = ['Write a sentence to describe the image.']


def build_img_prompt(data_type):
    if data_type == 'caption':
        return CaptaionPrompt()
    else:
        raise NotImplementedError(f'Data type `{data_type}` not supported.')


@TRANSFORMS.register_module()
class AutoTextTokenizer(BaseTransform):

    def __init__(self,
                 pretrained='bert-base-uncased',
                 input_text='caption',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=48,
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length


        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, **kwargs)

    def transform(self, results):
        text = results[self.input_text]
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True)
        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)

        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


@TRANSFORMS.register_module()
class BertTextTokenizer(BaseTransform):

    def __init__(self,
                 pretrained='bert-base-uncased',
                 input_text='text',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=32,
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length

        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, **kwargs)

    def transform(self, results):
        text = results[self.input_text]
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True)
        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)

        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)

        img_postfix = results['task']

        img_pre = self.tokenizer(img_postfix, max_length=4, add_special_tokens=False, return_tensors='pt',
                                truncation=True)
        results['img_postfix'] = img_pre['input_ids'].squeeze(dim=0)
        results['img_postfix_mask'] = img_pre['attention_mask'].squeeze(dim=0)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


@TRANSFORMS.register_module()
class LlamaTokenizer(BaseTransform):

    def __init__(self,
                 pretrained='bert-base-uncased',
                 input_text='caption',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=48,
                 add_eos_token=True,
                 use_img_prompt=True,
                 data_type='caption',
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length
        self.add_eos_token = add_eos_token
        self.use_img_prompt = use_img_prompt


        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            model_max_length=max_length,
            padding_side='right',
            use_fast=False,
            **kwargs)
        import pdb;pdb.set_trace()
        """
        The default `unk` token in `Tokenizer` is ''.
        Any new special tokens mush be added before the `unk_token` is modified.
        If a new special token is added, and the vocabulary is extended,
        the world embedding and prediction layer need to be extended accordingly.
        Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py#L384-L447
        """
        DEFAULT_PAD_TOKEN = '</s>'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'

        self.tokenizer.add_special_tokens({
            'eos_token': DEFAULT_EOS_TOKEN,
            'bos_token': DEFAULT_BOS_TOKEN,
            'pad_token': DEFAULT_PAD_TOKEN,
        })

        # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
        self.tokenize = partial(
            self.tokenizer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            add_special_tokens=False)
        if self.use_img_prompt:
            self.img_prompt = build_img_prompt(data_type)
            self.img_prompt.set_max_length(
                partial(self.tokenizer, add_special_tokens=False))

    def transform(self, results):
        
        text = results[self.input_text]
        if not self.use_img_prompt:
            text = self.tokenizer.bos_token + text
        text = text + self.tokenizer.eos_token if self.add_eos_token else text
        inputs = self.tokenize(
            text, max_length=self.tokenizer.model_max_length)

        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)

        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)

        if self.use_img_prompt:
            pre, post = self.img_prompt.sample()
            img_pre = self.tokenize(
                self.tokenizer.bos_token + pre,
                max_length=self.img_prompt.pre_max_length + 1)
            img_post = self.tokenize(
                post, max_length=self.img_prompt.post_max_length)
            results['img_prefix'] = img_pre[self.token_ids].squeeze(dim=0)
            results['img_postfix'] = img_post[self.token_ids].squeeze(dim=0)
            results['img_prefix_mask'] = img_pre[self.input_mask].squeeze(
                dim=0)
            results['img_postfix_mask'] = img_post[self.input_mask].squeeze(
                dim=0)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


@TRANSFORMS.register_module()
class LlamaTokenizerPrompt(BaseTransform):

    def __init__(self,
                 pretrained='bert-base-uncased',
                 input_text='caption',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=48,
                 prefix_length=64,
                 add_eos_token=True,
                 use_img_prompt=True,
                 data_type='caption',
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length
        self.add_eos_token = add_eos_token
        self.use_img_prompt = use_img_prompt


        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            model_max_length=max_length,
            padding_side='right',
            use_fast=False,
            **kwargs)

        self.prefix_length = prefix_length
        """
        The default `unk` token in `Tokenizer` is ''.
        Any new special tokens mush be added before the `unk_token` is modified.
        If a new special token is added, and the vocabulary is extended,
        the world embedding and prediction layer need to be extended accordingly.
        Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py#L384-L447
        """
        DEFAULT_PAD_TOKEN = '</s>'
        DEFAULT_EOS_TOKEN = '</s>'
        DEFAULT_BOS_TOKEN = '<s>'

        self.tokenizer.add_special_tokens({
            'eos_token': DEFAULT_EOS_TOKEN,
            'bos_token': DEFAULT_BOS_TOKEN,
            'pad_token': DEFAULT_PAD_TOKEN,
        })

        # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
        self.tokenize = partial(
            self.tokenizer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            add_special_tokens=False)

        self.prefix_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n"
        self.suffix_prompt = "\n\n### Response: "

        self.prompt_tokenize = AutoTokenizer.from_pretrained(
            pretrained,
            model_max_length=max_length,
            padding_side='left',
            use_fast=False,
            **kwargs)
        self.prompt_tokenize.add_special_tokens({
            'eos_token': DEFAULT_EOS_TOKEN,
            'bos_token': DEFAULT_BOS_TOKEN,
            'pad_token': DEFAULT_PAD_TOKEN,
        })

    def transform(self, results):
        task = results.get("task", 'Write a sentence to describe the image.')
        prefix = self.prefix_prompt.format(task)
        suffix = self.suffix_prompt

        text = results[self.input_text]

        prefix = self.tokenizer.bos_token + prefix

        text = text + self.tokenizer.eos_token
        inputs = self.tokenize(
            text, max_length=self.tokenizer.model_max_length)

        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)

        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)

        img_pre = self.tokenize(prefix, max_length=self.prefix_length)
        img_post = self.prompt_tokenize(suffix,
                                        return_tensors='pt',
                                        padding='max_length',
                                        truncation=True,
                                        max_length=8,
                                        add_special_tokens=False)

        results['img_prefix'] = img_pre[self.token_ids].squeeze(dim=0)
        results['img_postfix'] = img_post[self.token_ids].squeeze(dim=0)
        results['img_prefix_mask'] = img_pre[self.input_mask].squeeze(dim=0)
        results['img_postfix_mask'] = img_post[self.input_mask].squeeze(dim=0)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str



def locate_new_piece_using_morphology(img0, img1, corner_x=12, corner_y=12, cell_size=29):
    import cv2
    # 计算两幅图像的差异
    # import pdb;pdb.set_trace()
    diff = cv2.absdiff(img1, img0)
    # 转换为灰度图

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray[gray<90] = 0
    gray[gray>220] = 0
    # cv2.imwrite('/opt/tiger/gary.jpg', gray)
    # 应用阈值
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('/opt/tiger/thresh.jpg', thresh)
    # 形态学变换，如膨胀操作
    kernel = np.ones((2,2),np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erode, kernel, iterations=2)
    # cv2.imwrite('/opt/tiger/dilation.jpg', dilation)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    grid_x = round((cx - corner_x) / cell_size)
    grid_y = round((cy - corner_y) / cell_size)

    return grid_x, grid_y

@TRANSFORMS.register_module()
class TokenizerforGoImage(BaseTransform):

    def __init__(self,
                 pretrained='./work_dirs/init/gpt2-medium',
                 input_text='prompt',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=100,
                 test_mode=False,
                 padding_side='left',
                 board_size=9,
                 sub_board_size=0,
                 offset=None,
                 is_battle=False,
                 pred_image=True,
                 is_llama=False,
                 use_action=False,
                 use_la_action=False,
                 la_action_num=729,
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length
        self.test_mode = test_mode
        self.padding_side = padding_side
        self.board_size = board_size
        self.sub_board_size = sub_board_size
        self.offset = offset
        self.is_battle = is_battle
        self.pred_image = pred_image
        # import pdb;pdb.set_trace()
        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        if is_llama:
            self.tokenizer = AutoTokenizer.from_pretrained("./work_dirs/init/Llama-300m", model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained('./work_dirs/init/gpt2_medium', model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
        self.tokenizer.model_max_length = max_length
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN], special_tokens=True)
        if pred_image:
            self.tokenizer.add_tokens([DEFAULT_MASK_TOKEN], special_tokens=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
        self.tokenize = partial(self.tokenizer,return_tensors='pt',padding='max_length',truncation=True,add_special_tokens=False)
        self.use_action = use_action
       
        if use_action:
            self.tokenizer.add_tokens([DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_TOKEN, DEFAULT_ACT_END_TOKEN], special_tokens=True)
            
    def transform(self, results):
        # import pdb;pdb.set_trace() 
        pos = -1 if self.padding_side == 'left' else 0
        data_mode = results['data_mode']
        roles = ['b', 'w']
        text = []
        ignore_text = []
        offset_x = 0
        offset_y = 0
        if self.sub_board_size > 0:
            offset_x = random.randint(0, 10) if self.offset == None else self.offset[0]
            offset_y = random.randint(0, 10) if self.offset == None else self.offset[1]
        
        if self.use_action:
            if results.get('gt_action', None) is None and not self.test_mode and self.use_action:
                move_ids = torch.tensor(results.get('gt_action'))[None]
        
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_ACT_START_TOKEN + DEFAULT_ACT_TOKEN + DEFAULT_ACT_END_TOKEN + DEFAULT_IM_START_TOKEN + DEFAULT_MASK_TOKEN + DEFAULT_IM_END_TOKEN if not self.test_mode else  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_ACT_START_TOKEN 
        else:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_IM_START_TOKEN + DEFAULT_MASK_TOKEN + DEFAULT_IM_END_TOKEN if not self.test_mode else  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_IM_START_TOKEN 
       
        MASK_TOKEN_INDEX = self.tokenize(DEFAULT_MASK_TOKEN).input_ids[0][0].item()  
        IMAGE_TOKEN_INDEX = self.tokenize(DEFAULT_IMAGE_TOKEN).input_ids[0][0].item()
        ACT_TOKEN_INDEX = self.tokenize(DEFAULT_ACT_TOKEN).input_ids[0][0].item()
        inputs = self.tokenize(text)
        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)
        token_ids[token_ids==IMAGE_TOKEN_INDEX] = -200
        token_ids[token_ids==ACT_TOKEN_INDEX] = -400
        
        if not self.test_mode:
            act_token_pos = torch.where(token_ids[0]==-400)[0][0].item()
            token_ids = torch.cat([token_ids[:, :act_token_pos], move_ids+len(self.tokenizer), token_ids[:, act_token_pos+1:]], dim=1)
            input_mask = torch.cat([input_mask[:, :act_token_pos], torch.ones_like(move_ids), input_mask[:, act_token_pos+1:]], dim=1)
        if self.pred_image:
            token_ids[token_ids==MASK_TOKEN_INDEX] = -300
        
        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)
        
     
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


@TRANSFORMS.register_module()
class TokenizerforCALVIN(BaseTransform):

    def __init__(self,
                 pretrained='./work_dirs/init/gpt2-medium',
                 input_text='prompt',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=100,
                 test_mode=False,
                 padding_side='left',
                 pred_image=True,
                 use_action=False,
                 use_state=False,
                 pred_image_num=5,
                 use_lang_embed=False,
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length
        self.test_mode = test_mode
        self.padding_side = padding_side
        self.pred_image = pred_image
        self.pred_image_num = pred_image_num
        self.use_lang_embed = use_lang_embed
        # import pdb;pdb.set_trace()
        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained("./work_dirs/Llama_tokenizer", model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
      
        self.tokenizer.model_max_length = max_length
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN], special_tokens=True)
        if pred_image:
            self.tokenizer.add_tokens([DEFAULT_MASK_TOKEN], special_tokens=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
        self.tokenize = partial(self.tokenizer,return_tensors='pt',padding='max_length',truncation=True,add_special_tokens=False)
        self.use_action = use_action
        
        self.use_state = use_state
        
        self.tokenizer.add_tokens([DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_TOKEN,  DEFAULT_ACT_END_TOKEN], special_tokens=True)
            
        
    def transform(self, results):
        
        pos = -1 if self.padding_side == 'left' else 0

        roles = ['b', 'w']
        text = results[self.input_text]
        ignore_text = []
        offset_x = 0
        offset_y = 0
     
        _move = DEFAULT_ACT_TOKEN
        
        text = '' if self.use_lang_embed else  text + ' '
        
        pred_tokens = ''.join([DEFAULT_ACT_START_TOKEN + _move + DEFAULT_ACT_END_TOKEN + DEFAULT_IM_START_TOKEN + DEFAULT_MASK_TOKEN + DEFAULT_IM_END_TOKEN] * self.pred_image_num)
        text = text + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + pred_tokens if not self.test_mode else  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_IM_START_TOKEN 
        
        MASK_TOKEN_INDEX = self.tokenize(DEFAULT_MASK_TOKEN).input_ids[0][0].item()  
        IMAGE_TOKEN_INDEX = self.tokenize(DEFAULT_IMAGE_TOKEN).input_ids[0][0].item()
        ACT_TOKEN_INDEX = self.tokenize(DEFAULT_ACT_TOKEN).input_ids[0][0].item()
        inputs = self.tokenize(text)
        token_ids = inputs.get(self.token_ids, None)
        input_mask = inputs.get(self.input_mask, None)
        token_ids[token_ids==IMAGE_TOKEN_INDEX] = -200
        
        if self.pred_image:
            token_ids[token_ids==MASK_TOKEN_INDEX] = -300
        
        results[self.token_ids] = token_ids.squeeze(dim=0)
        results[self.input_mask] = input_mask.squeeze(dim=0)
        results['action_idx'] = ACT_TOKEN_INDEX
 
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


@TRANSFORMS.register_module()
class TokenizerforCALVINEnvVal(BaseTransform):

    def __init__(self,
                 pretrained='./work_dirs/init/gpt2-medium',
                 input_text='prompt',
                 token_ids='input_ids',
                 input_mask='attention_mask',
                 max_length=100,
                 test_mode=False,
                 padding_side='left',
                 pred_image=True,
                 use_action=False,
                 use_state=False,
                 pred_image_num=5,
                 test_with_act=False,
                 use_act_start_end=True,
                 use_lang_embed=False,
                 **kwargs):
        self.input_text = input_text
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.pretrained = pretrained
        self.max_length = max_length
        self.test_mode = test_mode
        self.padding_side = padding_side
        self.pred_image = pred_image
        self.pred_image_num = pred_image_num
        self.test_with_act = test_with_act
        self.use_act_start_end = use_act_start_end
        self.use_lang_embed = use_lang_embed
        assert isinstance(
            pretrained, str
        ), f'Autokenizer must has pretrained models, but get {pretrained}'
        self.tokenizer = AutoTokenizer.from_pretrained("./work_dirs/init/Llama-300m", model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
      
        self.tokenizer.model_max_length = max_length
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN], special_tokens=True)
        if pred_image:
            self.tokenizer.add_tokens([DEFAULT_MASK_TOKEN], special_tokens=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
        self.tokenize = partial(self.tokenizer,return_tensors='pt',padding='max_length',truncation=True,add_special_tokens=False)
        self.use_action = use_action
        
        self.use_state = use_state
        
    
        self.tokenizer.add_tokens([DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_TOKEN,  DEFAULT_ACT_END_TOKEN], special_tokens=True)
            

    def transform(self, results):
 

        pos = -1 if self.padding_side == 'left' else 0
        task_seqs = results[self.input_text]
        

        MASK_TOKEN_INDEX = self.tokenize(DEFAULT_MASK_TOKEN).input_ids[0][0].item()  
        IMAGE_TOKEN_INDEX = self.tokenize(DEFAULT_IMAGE_TOKEN).input_ids[0][0].item()
        ACT_TOKEN_INDEX = self.tokenize(DEFAULT_ACT_TOKEN).input_ids[0][0].item()

        seq_token_ids = []
        seq_input_mask = []
        if isinstance(results[self.input_text], tuple):
            for text in task_seqs:
                text = '' if self.use_lang_embed else  text + ' '
                if not self.use_act_start_end:
                    text = text + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                else:
                    if self.test_with_act:
                        text = text + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_ACT_START_TOKEN + DEFAULT_ACT_TOKEN + DEFAULT_ACT_END_TOKEN
                    else:
                        text = text + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + DEFAULT_ACT_START_TOKEN 
                inputs = self.tokenize(text)
                token_ids = inputs.get(self.token_ids, None)
                input_mask = inputs.get(self.input_mask, None)
                token_ids[token_ids==IMAGE_TOKEN_INDEX] = -200
                seq_token_ids.append(token_ids.squeeze(dim=0))
                seq_input_mask.append(input_mask.squeeze(dim=0))
        
        results[self.token_ids] = torch.stack(seq_token_ids)
        results[self.input_mask] = torch.stack(seq_input_mask)

        results['action_idx'] = ACT_TOKEN_INDEX
       
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pretrained={self.pretrained}')
        return repr_str


# @TRANSFORMS.register_module()
# class GPT2TokenizerforTicTacToe(BaseTransform):

#     def __init__(self,
#                  pretrained='./work_dirs/init/gpt2-medium',
#                  input_text='prompt',
#                  token_ids='input_ids',
#                  input_mask='attention_mask',
#                  max_length=100,
#                  add_eos_token=False,
#                  test_mode=False,
#                  padding_side='left',
#                  is_mamba=False,
#                  selected_step_num = 100,
#                  **kwargs):
#         self.input_text = input_text
#         self.token_ids = token_ids
#         self.input_mask = input_mask
#         self.pretrained = pretrained
#         self.max_length = max_length
#         self.add_eos_token = add_eos_token
#         self.test_mode = test_mode
#         self.padding_side = padding_side
#         self.is_mamba = is_mamba
#         self.selected_step_num = selected_step_num
#         assert isinstance(
#             pretrained, str
#         ), f'Autokenizer must has pretrained models, but get {pretrained}'
#         if is_mamba:
#             self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
#         else:
#             self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', model_max_length=max_length, padding_side=padding_side, use_fast=False, **kwargs)
#         self.tokenizer.model_max_length = max_length
#         self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_FRAME_INTERVAL], special_tokens=True)

#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
#         self.tokenize = partial(self.tokenizer,return_tensors='pt',padding='max_length',truncation=True,add_special_tokens=False)

#     def transform(self, results):
        
#         # {"filename": "00000000_04126375_X_O", "win_shape": "O", "start_shape": "X", "steps": "04126375"}   
#         pos = -1 if self.padding_side == 'left' else 0
#         # IMAGE_TOKEN_INDEX = self.tokenize(DEFAULT_IMAGE_TOKEN).input_ids[0][pos].item()
#         # MASK_TOKEN_INDEX = self.tokenize(DEFAULT_MASK_TOKEN).input_ids[0][pos].item()
#         moves = results['moves']
#         winner = results['winner']
#         assert ('X' in winner or 'O' in winner or 'Draw' in winner), winner
#         if 'Draw' in winner:
#             winner = 'D'

#         # winner = 'B' if 'B' in winner else 'W'
#         text = []
#         shape =  ['X', 'O']
#         start_shape = results['start_shape']
#         shape_index = 0 if start_shape == 'X' else 1
#         for mi, move in enumerate(moves):
#             if self.test_mode and mi >= self.selected_step_num:
#                 break
#             _move = str(move) + ' ' + shape[shape_index % 2]
#             if mi == 0:
#                 winner = shape[shape_index % 2]
#             shape_index += 1
#             text.append(_move)

            

#         text = DEFAULT_FRAME_INTERVAL.join(text)
#         text = winner + ' ' + DEFAULT_IM_START_TOKEN + ' ' + text
#         if not self.test_mode:
#             text = text + DEFAULT_IM_END_TOKEN
#         elif self.selected_step_num == 0:
#             text = text
#         else:
#             text = text + DEFAULT_FRAME_INTERVAL
            
#         inputs = self.tokenize(text)
#         token_ids = inputs.get(self.token_ids, None)
#         input_mask = inputs.get(self.input_mask, None)

#         results[self.token_ids] = token_ids.squeeze(dim=0)
#         results[self.input_mask] = input_mask.squeeze(dim=0)
            
#         import pdb;pdb.set_trace()
#         return results

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'pretrained={self.pretrained}')
#         return repr_str


# @TRANSFORMS.register_module()
# class LlamaTokenizerforMixMask(BaseTransform):

#     def __init__(self,
#                  pretrained='./tokenizer/llama/open_3B_v2',
#                  input_text='prompt',
#                  token_ids='input_ids',
#                  input_mask='attention_mask',
#                  max_length=48,
#                  add_eos_token=False,
#                  **kwargs):
#         self.input_text = input_text
#         self.token_ids = token_ids
#         self.input_mask = input_mask
#         self.pretrained = pretrained
#         self.max_length = max_length
#         self.add_eos_token = add_eos_token

#         assert isinstance(
#             pretrained, str
#         ), f'Autokenizer must has pretrained models, but get {pretrained}'
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             pretrained,
#             model_max_length=max_length,
#             padding_side='left',
#             use_fast=False,
#             **kwargs)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         # setting `add_special_tokens=False` to disable bos token, so that we can process it manually
#         self.tokenize = partial(
#             self.tokenizer,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             add_special_tokens=False)

        

#     def transform(self, results):

#         # text = results[self.input_text]
#         #
#         # inputs = self.tokenize(text)
#         #
#         # token_ids = inputs.get(self.token_ids, None)
#         # input_mask = inputs.get(self.input_mask, None)
#         #
#         # results[self.token_ids] = token_ids.squeeze(dim=0)
#         # results[self.input_mask] = input_mask.squeeze(dim=0)
#         if results.get(self.input_text, None):
#             text = results[self.input_text]
#             inputs = self.tokenize(text)
#             token_ids = inputs.get(self.token_ids, None)
#             input_mask = inputs.get(self.input_mask, None)
#             results['prefix_ids'] = token_ids.squeeze(dim=0)
#             results['prefix_mask'] = input_mask.squeeze(dim=0)
#             tmp = torch.ones((1, 32)) * -100
#             tmp = tmp.to(torch.long)

#             results['text_label'] = tmp.squeeze(dim=0)
#         else:
#             prefix = results['human']
#             suffix = results['gpt']
#             prefix = self.tokenize(prefix, max_length=16)
#             suffix = self.tokenize(suffix, max_length=32)
#             prefix_tokens_ids = prefix.get(self.token_ids, None)
#             prefix_tokens_mask = prefix.get(self.input_mask, None)

#             suffix_tokens_ids = suffix.get(self.token_ids, None)
#             suffix_tokens_mask = suffix.get(self.input_mask, None)

#             prefix_ids = torch.cat([prefix_tokens_ids, suffix_tokens_ids], dim=1)
#             prefix_mask = torch.cat([prefix_tokens_mask, suffix_tokens_mask], dim=1)
#             assert prefix_ids.size(1) == 48

#             results['prefix_ids'] = prefix_ids.squeeze(dim=0)
#             results['prefix_mask'] = prefix_mask.squeeze(dim=0)

#             index_mask = 1 - suffix_tokens_mask
#             label = suffix_tokens_ids
#             label = label.masked_fill(index_mask.to(torch.bool), -100)

#             results['text_label'] = label.squeeze(dim=0)

#         return results

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'pretrained={self.pretrained}')
#         return repr_str