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
import os.path as osp
from typing import Sequence
import random
from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import mmcv
from ldm.registry import DATASETS, TRANSFORMS
import os
import re
import json
@DATASETS.register_module()
class GoImage(Dataset):

    def __init__(
        self,
        ann_file: str,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        clip_length: int = 1
    ):
        self.data_root = data_root
        self.data_infors = self.load_annotations(ann_file)
        self.clip_length = clip_length
        self.test = test_mode

        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def load_annotations(self, data_ann):
        data = json.load(open(data_ann))
        return data

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        # import pdb;pdb.set_trace()
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        imgs = []
        if '/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image' in item['image0']:
            img0 = mmcv.imread(item['image0'].replace('/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image', self.data_root))[:, :, ::-1]
            img1 = mmcv.imread(item['image1'].replace('/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image', self.data_root))[:, :, ::-1]
        else:
            img0 = mmcv.imread(os.path.join(self.data_root + item['image0']))[:, :, ::-1]
            img1 = mmcv.imread(os.path.join(self.data_root + item['image1']))[:, :, ::-1]
        # if len(item['filenames']) < self.clip_length:
        #     item['filenames'] = ['/opt/tiger/mmagicinit/ldm/data/go_dataset_size9/empty_board.png', item['filenames'][0]]
        # else:
        #     sample_start = random.sample([i for i in range(len(item['filenames'])-self.clip_length+1)], k=1)[0]
        #     item['filenames'] = item['filenames'][sample_start:sample_start+self.clip_length]
        item['prefix'] = ''
        item['imgs'] = [img0, img1]
        # item['filename'] = item['filenames']
        item = self.pipeline(item)
        return item



@DATASETS.register_module()
class GoImageMF(Dataset):

    def __init__(
        self,
        # ann_file: str,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        clip_length: int = 1,
        interval=1,
        sample_num=-1
    ):
        # import pdb;pdb.set_trace()
        self.data_root = data_root
        infos = []
        file_list = os.listdir(data_root)
        with open(f'./data/go_dataset_size9/go_eval_info.json', 'r') as f:
            battles = json.load(f)
        battles = battles['battles']
        idx_to_battle = {}
        for battle in battles:
            idx_to_battle["%09d" % battle['idx']] = battle
        num = 0
        for file in file_list:
            battle = idx_to_battle[file]
            moves = battle['moves']
            captured = battle['captured']
            # types = battle[f'type_size{str(interval * 2 - 1)}']
            types = battle['type_size3']
            file_path = os.path.join(data_root, file)
            names = os.listdir(file_path)
            names = [name.replace('i_', '') for name in names]
            ignore_move_names = [name.replace('i_', '') for name in os.listdir(file_path) if 'i_' in name]
            move_names = sorted(names, key=lambda x: int(re.search(r'\d+', x).group()))
            start_move = int(move_names[0].replace('.png', ''))
            black_move_start = start_move % 2 == 0
            if black_move_start:
                move_names = move_names[1:]
                # types = types[1:]
            
            for i in range(0, len(move_names)-(interval * 2), 2):
                item = {}
                _name0 = move_names[i] if move_names[i] not in ignore_move_names else 'i_' + move_names[i] 
                next_idx = i+(interval*2)-1
                _name1 = move_names[next_idx] if move_names[next_idx] not in ignore_move_names else 'i_' + move_names[next_idx] 
                item['idx'] = battle['idx']
                item['clip_start_end_id'] = [_name0, _name1]
                item['image0'] = os.path.join(file_path, _name0)
                item['image1'] = os.path.join(file_path, _name1)
                idx0 = int(_name0[:-4].replace('i_', ''))
                idx1 = int(_name1[:-4].replace('i_', ''))
                item['action'] = []
                item['capture'] = []
                
                for mi in range(idx0+1, idx1+1):
                    item['action'].append(moves[mi])
                    item['capture'].append(captured[mi])
                infos.append(item)
        print(num)
        if sample_num > 0:
            infos = infos[:sample_num]
        self.data_infors = infos
        self.clip_length = clip_length
        self.test = test_mode

        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def load_annotations(self, data_ann):
        data = json.load(open(data_ann))
        return data

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        # import pdb;pdb.set_trace()
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        imgs = []
        # if '/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image' in item['image0']:
        #     img0 = mmcv.imread(item['image0'].replace('/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image', self.data_root))[:, :, ::-1]
        #     img1 = mmcv.imread(item['image1'].replace('/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image', self.data_root))[:, :, ::-1]
        # else:
        img0 = mmcv.imread(item['image0'])[:, :, ::-1]
        img1 = mmcv.imread(item['image1'])[:, :, ::-1]
        # if len(item['filenames']) < self.clip_length:
        #     item['filenames'] = ['/opt/tiger/mmagicinit/ldm/data/go_dataset_size9/empty_board.png', item['filenames'][0]]
        # else:
        #     sample_start = random.sample([i for i in range(len(item['filenames'])-self.clip_length+1)], k=1)[0]
        #     item['filenames'] = item['filenames'][sample_start:sample_start+self.clip_length]
        item['prefix'] = ''
        item['imgs'] = [img0, img1]
        # item['filename'] = item['filenames']
        item = self.pipeline(item)
        # import pdb;pdb.set_trace()
        return item

if __name__ == '__main__':
    img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)
    aux_info = ['input_ids', 'attention_mask', 'state', 'action', 'action_idx']
    train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(type='SampleFrames', clip_len=17, frame_interval=2, num_clips=1),
    # dict(type='DecordDecode'),
    # dict(type='CenterCropLongEdgeVideo'),
    # dict(type='ResizeVideo', scale=(128, 128),keep_ratio=False),
    dict(type='PackVideoInputs')
]

    test_pipeline = [
    dict(type='ResizeVideo', scale=(128, 128), keep_ratio=False),
    dict(type='PackVideoInputs', meta_keys=['clip_start_end_id', 'idx'])
]

    dataset = GoImageMF(data_root="./data/go_dataset_size9/go_eval", interval=1, sample_num=50000,pipeline=test_pipeline)
    # dataset = CALVINVideoValDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/training", clip_length=5, interval=5, pipeline=train_pipeline)

    for data in dataset:
        pass
