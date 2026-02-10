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

from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from ldm.registry import DATASETS, TRANSFORMS
import numpy as np
import copy
import mmcv
@DATASETS.register_module()
class CALVINDataset(Dataset):

    def __init__(
        self,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        obs_type='rgb_static',
        clip_length=1,
        interval_range=[1, 20],
        use_hand=True,
        extra_data_path=[]
    ):
        self.data_root = data_root
        self.clip_length = clip_length
        self.interval_range = interval_range
        scene_info_path = osp.join(data_root, 'scene_info.npy')
        lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        
        if len(extra_data_path) > 0:
            extra_anns = []
            for _extra_data_path in extra_data_path:
                extra_txt = _extra_data_path + '/gr1_rollout.txt'
                extra_infos = open(extra_txt, 'r').readlines()
                for info in extra_infos:
                    start, end, task = info.split(' ')[:3]
                    task_lang = task
                    need_slide = '' if 'part2' in _extra_data_path else 'need_slide'
                    annotations.append(((int(start), int(end)), task_lang, _extra_data_path, need_slide))

        random.shuffle(annotations)        
        self.data_infors = annotations
        self.obs_type = obs_type
        self.use_hand = use_hand
        # self.data_infors = self.load_annotations(osp.join(data_root, ann_file))

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
        item = {}
        ann = copy.copy(self.data_infors[idx])
        start_index, end_index = ann[0][0], ann[0][1]
        if self.clip_length == 1:
            sample_frame = random.sample(list(range(start_index, end_index+1)), 1)[0]
            frame_path = f"{self.data_root}/episode_{sample_frame:07d}.npz"
            t = np.load(frame_path, allow_pickle=True)
            img = t[self.obs_type]
            hand = t['rgb_gripper']
            if self.use_hand:
                score = random.random()
                item['img'] = img if score > 0.5 else hand
            else:
                item['img'] = img
            item['original_shape'] = img.shape[:2]
            item['img_shape'] = img.shape[:2]
           
        else:
            max_try = 10
            try_num = 0
            while try_num < max_try:
                interval = random.sample(list(range(self.interval_range[0], self.interval_range[1]+1)), 1)[0]
                select_len = interval * (self.clip_length-1) + 1
                if select_len + start_index > end_index:
                    interval = random.sample(list(range(self.interval_range[0], self.interval_range[1]+1)), 1)[0]
                    try_num += 1
                else:
                    break
            if try_num >= max_try:
                interval = 1
            
            try:
                sample_start = random.sample(list(range(start_index, end_index+1-(interval*(self.clip_length-1)))), 1)[0]
            except:
                print("__________", start_index, end_index, interval, "__________")
            # print(random.random(), sample_start, interval, ann[1])
            imgs = []
            hands = []
            actions = []
            for i in range(self.clip_length):
                index = sample_start + (i * interval)
                frame_path = f"{self.data_root}/episode_{index:07d}.npz"
                t = np.load(frame_path, allow_pickle=True)
                img = t[self.obs_type]
                
                action = t['rel_actions']
                
                # img = mmcv.imresize(img, (256, 256), interpolation='bilinear', backend='pillow')
                if self.use_hand:
                    hand = t['rgb_gripper']
                    hand = mmcv.imresize(hand, (256, 256), interpolation='bilinear', backend='pillow')
                    hand = (hand / 127.5) - 1.0
                    hands.append(hand)

                imgs.append(img)
                actions.append(action)
            if self.use_hand:
                hand = np.stack(hands)
                hand = np.transpose(hand, (0, 3, 1, 2))
                item['hand'] = hand
            item['imgs'] = np.stack(imgs)
            # item['state'] = np.stack(states)
            item['action'] = np.stack(actions)
            item['action'][:, :3] *= 50
            item['action'][:, 3:6] *= 20
            item['original_shape'] = imgs[0].shape[:2]
            item['img_shape'] = imgs[0].shape[:2]
        
        item['gt_label'] = 0
        item['prefix'] = ''
        item['data_root'] = self.data_root
        item = self.pipeline(item)
        return item



@DATASETS.register_module()
class CALVINValDataset(Dataset):

    def __init__(
        self,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        obs_type='rgb_static',
        use_hand=True
    ):
        self.data_root = data_root

        scene_info_path = osp.join(data_root, 'scene_info.npy')
        lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        annotations = annotations

        data_paths = []
        is_hand = []
        for ann in annotations:
            index = ann[0]
            start, end = index[0], index[1]
            for i in range(start, end+1):
                path = f"{self.data_root}/episode_{i:07d}.npz"
                data_paths.append(path)
                is_hand.append(False)
                if use_hand:
                    data_paths.append(path)
                    is_hand.append(True)
        self.data_infors = list(zip(data_paths, is_hand))
        self.obs_type = obs_type
        self.use_hand = use_hand
            
        # self.data_infors = self.load_annotations(osp.join(data_root, ann_file))

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
        item = {}
        frame_path, is_hand = copy.copy(self.data_infors[idx])
        t = np.load(frame_path, allow_pickle=True)
        img = t[self.obs_type] if not is_hand else t['rgb_gripper']
        item['img'] = img
        # import cv2
        # cv2.imwrite('/opt/tiger/mmagicinit/test.jpg', item['img'][:, :, ::-1])
        item['prefix'] = ''
        item['data_root'] = self.data_root
        item['original_shape'] = img.shape[:2]
        item['img_shape'] = img.shape[:2]
        item['gt_label'] = 0
        item = self.pipeline(item)
        
        return item


@DATASETS.register_module()
class CALVINVideoValDataset(Dataset):

    def __init__(
        self,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        obs_type='rgb_static',
        clip_length=2,
        interval = 10,
        max_interval = False,
    ):
        self.data_root = data_root
        self.clip_length = clip_length
        scene_info_path = osp.join(data_root, 'scene_info.npy')
        lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        # length = len(annotations) // 3
        annotations = annotations[:2]
        data_paths = []
        langs = []
        clip_start_end_ids = []
        
        for ann in annotations:
            lang = ann[1]
            index = ann[0]
            
            start, end = index[0], index[1]
            interval = 1
            for i in range(start, end+2-clip_length):
                clip_path = []
                for j in range(clip_length):
                    ep_idx = i+j
                    path = f"{self.data_root}/episode_{ep_idx:07d}.npz"
                    clip_path.append(path)
                data_paths.append(clip_path)
                clip_start_end_ids.append((i, i+clip_length-1))
                langs.append(ann[1])
            
        self.data_infors = data_paths
        self.clip_start_end_ids = clip_start_end_ids
        self.langs = langs
        self.obs_type = obs_type
        
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
        item = {}
        imgs = []
        states = []
        index = []
        rel_actions = []
        clip_path = copy.copy(self.data_infors[idx])
        lang = self.langs[idx]
        for frame_path in clip_path:
            index.append(int(frame_path.split('_')[-1].split('.')[0]))
            t = np.load(frame_path, allow_pickle=True)
            img = t[self.obs_type]
            state = t['robot_obs']
            rel_action = t['rel_actions']
            rel_actions.append(rel_action)
            imgs.append(img)
            states.append(state)
        action_list = []
        rel_action_list = []
        
            
        # import pdb; pdb.set_trace()
        item['action'] = np.stack(rel_actions)
        item['action'][:, :3] *= 50
        item['action'][:, 3:6] *= 20
        item['prompt'] = lang
        item['rel_pos'] = action_list
        item['imgs'] = imgs
        item['prefix'] = ''
        item['data_root'] = self.data_root
        item['original_shape'] = img.shape[:2]
        item['img_shape'] = img.shape[:2]
        item['gt_label'] = 0
        item['states'] = np.stack(states)
        item['clip_start_end_id'] = self.clip_start_end_ids[idx]
        item = self.pipeline(item)
        return item
    


if __name__ == '__main__':
    img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)
    aux_info = ['input_ids', 'attention_mask', 'state', 'action', 'action_idx']
    # train_pipeline = [
    # # dict(type='LoadImageNetFromFile'),
    # dict(type='CenterCropLongEdge'),
    # dict(type='Resize', scale=(256, 256), backend='pillow'),
    # dict(type='PackInputs')
    # ]
    train_pipeline = [
    # dict(type='LoadImageNetFromFile'),
    dict(type='CenterCropLongEdge'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='PackInputs')
]

    test_pipeline = [
    dict(type='ResizeVideo', scale=(128, 128), keep_ratio=False),
    dict(type='PackVideoInputs',  data_keys=['gt_img', 'states'], meta_keys=['clip_start_end_id'])
]
    dataset = CALVINDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/training", pipeline=train_pipeline, extra_data_path=['/mnt/bn/panxuran/calvin/task_D_extra', '/mnt/bn/panxuran/calvin/task_D_extra_alltask', '/mnt/bn/panxuran/calvin/task_D_extra_alltask_part2'])
    # dataset = CALVINValDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/validation", pipeline=train_pipeline)
    # dataset = CALVINVideoValDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/training", clip_length=10, interval=1, pipeline=test_pipeline, for_stage2=True)

    for data in dataset:
        pass
