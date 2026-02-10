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
import json
import os.path
import os.path as osp
from typing import Sequence
import numpy as np
import random
from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import mmcv
import re
from falcon.registry import DATASETS, TRANSFORMS
from sgfmill import sgf

from typing import List, Any, Optional
import multiprocessing as mp
import pickle
import numpy as np
import torch
import copy


class NumpySerializedList():
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)

@DATASETS.register_module()
class GoImageDataset(Dataset):

    def __init__(
        self,
        ann_file: str,
        data_root: str = '',
        # prefix='',
        pipeline: Sequence = (),
        test_mode: bool = False,
        level_list = [],
        pred_image=False
    ):
        # self.data_root = data_root + '/val' if test_mode else data_root +'/train'
        self.data_root = data_root
        self.data_infors = self.load_annotations(ann_file)
        self.test_mode = test_mode
        self.re_find_level = re_find_level
        self.pred_image = pred_image
        
        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)
    

    def load_annotations(self, data_ann):
        # import pdb;pdb.set_trace()
        
        with open(data_ann,'r') as f:
            infos = json.load(f)
        idx_battles = {}
        battles = NumpySerializedList(infos['battles'])
        for bi, battle in enumerate(battles):
            idx_battles[battle['idx']] = bi
        self.idx_battles = idx_battles
        return battles

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        item = copy.deepcopy(self.data_infors[idx])
        item['data_mode'] = 'go'
        item['level'] = '9d'
        if len(item['extra_moves']) == 0:
            item['extra_moves'] = None
        idx = item['idx']
        _idx = "%09d" % idx
        file_path = os.path.join(self.data_root, _idx)
        move_names = os.listdir(file_path)
        move_names = [move_name for move_name in move_names if 'extra' not in move_names]
        move_names = sorted(move_names, key=lambda x: int(re.search(r'\d+', x).group()))
        move_names = [file_path+'/'+move_name for move_name in move_names]
        start_move = int(move_names[0].replace('.png', ''))
        black_move_start = start_move % 2 == 0
        if black_move_start and start_move != 0:
            move_names = move_names[1:]
            black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
        elif black_move_start and start_move == 0:
            empty = ["/opt/tiger/PointVIS/falcon/data/go_dataset_size9/empty_board.png"]
            empty.extend(move_names)
            move_names = empty
            black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
        else:
            black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
        
        sample_black_idx= random.sample(range(black_move_idx), 1)[0]
        name2 = move_names[sample_black_idx] if not os.path.exist(move_names[sample_black_idx].replace(_idx+'/', _idx+'/extra_')) else move_names[sample_black_idx].replace(_idx+'/', _idx+'/extra_')
        name1 = move_names[sample_black_idx-1]
        sample_image_paths = [name1, name2]
        if self.test_mode and self.pred_image:
            img = mmcv.imread('/opt/tiger/PointVIS/falcon/data/go_dataset_size9/empty_board.png', channel_order='rgb', backend='pillow')
            item['img'] = img
            item = self.pipeline(item)
            return item
        


        gt_move = moves[sample_start]
        '''
        import cv2
        img0 = mmcv.imread(file_path+'/'+str(sample_start)+'.png', channel_order='rgb', backend='pillow')
        img1 = mmcv.imread(file_path+'/'+str(sample_start+1)+'.png', channel_order='rgb', backend='pillow')
        cv2.imwrite('/opt/tiger/go0.jpg', img0)
        cv2.imwrite('/opt/tiger/go1.jpg', img1)
        '''
        item['gt_move'] = gt_move
        item['image_paths'] = sample_image_paths

        img = mmcv.imread(sample_image_paths[0], channel_order='rgb', backend='pillow')
        if self.pred_image:
            pred_label = mmcv.imread(sample_image_paths[1], channel_order='rgb', backend='pillow')
            pred_label = mmcv.imresize(
                pred_label,
                (256, 256),
                interpolation='bilinear',
                backend='pillow')
            item['pred_label'] = pred_label
        item['img'] = img
        
        # item['extra_moves'] = extra_move
        # imgs = []
        
        item = self.pipeline(item)
        return item


@DATASETS.register_module()
class GoImageValDataset(Dataset):

    def __init__(
        self,
        # prefix='',
        pipeline: Sequence = (),
        test_mode: bool = False,
        level_list = [],
        pred_image=False,
        length=1,
        level='18k',
        data_mode='go'
    ):
        # self.data_root = data_root + '/val' if test_mode else data_root +'/train'

        self.length = length
        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.level = level
        self.data_mode = data_mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        item = {}
        item['data_mode'] = self.data_mode
        item['level'] = '9d'
        item['extra_moves'] = None
        item['katrain_level'] = self.level
        # if self.test_mode and self.pred_image:
        img = mmcv.imread('./empty_board.png', channel_order='rgb', backend='pillow')
        img = mmcv.imresize(
                img,
                (256, 256),
                interpolation='bilinear',
                backend='pillow')

        item['img'] = img
        item = self.pipeline(item)
        return item
        


@DATASETS.register_module()
class GoImageAccDataset(Dataset):

    def __init__(
        self,
        data_root='',
        pipeline: Sequence = (),
        test_mode: bool = False,
        level_list = [],
        pred_image=False,
        length=1,
        level='18k',
        data_mode='go',
        
    ):
        # self.data_root = data_root + '/val' if test_mode else data_root +'/train'
        
        self.length = length
        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.level = level
        self.data_mode = data_mode
        file_paths = []
        for battle_file in os.listdir(data_root)[:1000]:
            _idx = battle_file
            battle_file_path = os.path.join(data_root, battle_file)
            all_move_names = os.listdir(battle_file_path)
            extra_move_names = [move_name for move_name in all_move_names if 'extra' in move_name]
            move_names = [move_name.replace('i_', '') for move_name in all_move_names if 'extra' not in move_name]
            move_names = sorted(move_names, key=lambda x: int(re.search(r'\d+', x).group()))
            start_move = int(move_names[0].replace('.png', ''))
            black_move_start = start_move % 2 == 0
            if black_move_start and start_move != 0:
                move_names = move_names[1:]
                black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
            elif black_move_start and start_move == 0:
                empty = ["./data/go_dataset_size9/empty_board.png"]
                empty.extend(move_names)
                move_names = empty
                black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
            else:
                black_move_idx = [i for i in range(len(move_names)) if i % 2 == 1]
            if len(black_move_idx) == 0:
                continue
            move_names = [battle_file_path+'/'+move_name for move_name in move_names]
            for black_move in black_move_idx:
                try:
                    name2 = move_names[black_move] if 'extra_'+move_names[black_move].split('/')[-1] not in extra_move_names else move_names[black_move].replace(_idx+'/', _idx+'/extra_')       
                    name1 = move_names[black_move-1]
                except:
                    # import pdb;pdb.set_trace()
                    print(2)
                    continue

                file_paths.append((name1, name2))
        self.data_infos = file_paths[:length]
        # import pdb;pdb.set_trace()
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        # import pdb;pdb.set_trace()
        item = {}
        item['data_mode'] = 'acc'
        item['level'] = '9d'
        item['extra_moves'] = None
        item['katrain_level'] = self.level
        # if self.test_mode and self.pred_image:
        name1, name2 = copy.deepcopy(self.data_infos[idx])
        img = mmcv.imread(name1, channel_order='rgb', backend='pillow')
        img = mmcv.imresize(
                img,
                (256, 256),
                interpolation='bilinear',
                backend='pillow')

        item['img'] = img

        pred_label = mmcv.imread(name2, channel_order='rgb', backend='pillow')
        pred_label = mmcv.imresize(
            pred_label,
            (256, 256),
            interpolation='bilinear',
            backend='pillow')

        item['pred_label'] = pred_label

        item = self.pipeline(item)
        return item
    
  
  

if __name__ == '__main__':
    img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)
    # dataset settings
    aux_info = ['input_ids', 'attention_mask', 'pred_label', 'invalid']
    test_aux_info = ['input_ids', 'attention_mask', 'level', 'pred_label', 'data_mode', 'katrain_level']
    test_to_tensor = ['input_ids', 'attention_mask', 'pred_label']

    test_pipeline = [

    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='GPT2TokenizerforChessImage',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        game_type='short',
        board_size=19,
        sub_board_size=9,
        offset=(0, 0, 0, 160),
        test_mode=True,
        pred_image=True,
        is_llama=True,
    ),
    # dict(
    #     type='LlamaTokenizerforMask',
    #     pretrained='./tokenizer/llama/open_3B_v2/',
    #     input_text='prompt',
    # ),
    dict(type='Collect', keys=['img', *test_aux_info]),
    dict(type='ToTensor', keys=['img', *test_to_tensor]),
]
        
    dataset = GoImageValDataset(data_root='/opt/tiger/GO_val_image',  pipeline=test_pipeline, length=1000)
    # dataset = CALVINDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/training", clip_length=2, interval_range=1, pipeline=train_pipeline, calvin_scene='D')
    # dataset = CALVINValLossDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/validation", clip_length=10, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], interval_range=[1, 1], pipeline=train_pipeline)
    # dataset = CALVINValDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/training", pipeline=test_pipeline, calvin_scene='D', show=True)
    for data in dataset:
        pass


