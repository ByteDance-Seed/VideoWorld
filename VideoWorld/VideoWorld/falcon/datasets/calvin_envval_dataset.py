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
import mmcv
import torch.distributed as dist
from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from falcon.registry import DATASETS, TRANSFORMS
import numpy as np
import copy
import torch
from pathlib import Path
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from omegaconf import OmegaConf
import hydra
EP_LEN = 360

CALVIN_ROOT = '../calvin'


@DATASETS.register_module()
class CALVINEnvValDataset(Dataset):

    def __init__(
        self,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        obs_type='rgb_static',
        state_type='robot_obs',
        action_type='rel_actions',
        clip_length=1,
        use_hand=False,
        num_sequences = 1000
        # interval_range=[5, 15]
    ):
        self.data_root = data_root
        self.clip_length = clip_length
        # self.interval_range = interval_range
        # scene_info_path = osp.join(data_root, 'scene_info.npy')
        # lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        # annotations = np.load(lang_info_path, allow_pickle=True).item()
        # annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        
        # self.data_infors = annotations[:-800] 
        self.obs_type = obs_type
        self.state_type = state_type
        self.action_type = action_type
        self.test = test_mode
        observation_space = {
            'rgb_obs': ['rgb_static', 'rgb_gripper'], 
            'depth_obs': [], 
            'state_obs': ['robot_obs'], 
            'actions': ['rel_actions'], 
            'language': ['language']}
        try:
            current_rank = dist.get_rank()
        except:
            current_rank = 0
        # import pdb;pdb.set_trace()
        # self.env = self.make_env(data_root, observation_space, current_rank)

        conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable.yaml")
        eval_dir = get_log_dir('/opt/tiger/PointVIS')

        lang_info_path = osp.join(data_root, 'validation', 'lang_annotations/auto_lang_ann.npy')
        lang_embed_path = osp.join(data_root, 'validation', 'lang_annotations/embeddings.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        lang_embed = np.load(lang_embed_path, allow_pickle=True).item()
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in lang_embed.items()}
        # import pdb;pdb.set_trace()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["emb"])) 
        self.data_infors = annotations
        # self.data_infors_lang_embeddings = {ann[1]: ann[2] for ann in annotations}
        self.eval_sequences = get_sequences(num_sequences)
        self.val_annotations = val_annotations

        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def make_env(self, dataset_path, observation_space, device_id):
        val_folder = Path(dataset_path) / "validation"
        from falcon.utils.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
        device = torch.device('cuda', device_id)
        env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
        return env
    

    def load_annotations(self, data_ann):
        data = json.load(open(data_ann))
        return data

    def __len__(self):
        return len(self.eval_sequences)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        # import pdb;pdb.set_trace()
        initial_state, eval_sequence = self.eval_sequences[idx]
        eval_text = [self.val_annotations[seq][0] for seq in eval_sequence]
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        # lang_embed = [self.lang_embeddings[seq][0] for seq in eval_text]
        
        # Visualize
        # self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        # obs = self.env.get_obs()
        # import cv2
        # cv2.imwrite('/opt/tiger/test.jpg', obs['rgb_obs']['rgb_static'][:, :, ::-1])


        item = {}
        # img = obs['rgb_obs']['rgb_static']
        img = np.zeros((200, 200, 3))
        item['img'] = img
        item['state'] = robot_obs
        item['scene'] = scene_obs
        item['original_shape'] = img.shape[:2]
        item['img_shape'] = img.shape[:2]
        # item['lang_emb'] = np.concatenate(lang_embed)
        item['eval_sequence'] = eval_sequence
        item['initial_state'] = initial_state
        
        item['gt_label'] = 0
        item['prompt'] = eval_sequence
        item['data_root'] = self.data_root
        item = self.pipeline(item)
        
        # import cv2
        # print(ann[1])
        # for fi, img in enumerate(item['img']):
        #     img = img.permute(1, 2, 0)
        #     img = torch.clamp(((img+1)*127), min=0, max=255).numpy().astype(np.uint8)
        #     cv2.imwrite(f'/opt/tiger/visualize/{fi}.jpg', img[:, :, ::-1])
        return item





if __name__ == '__main__':
    img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)
    test_aux_info = ['input_ids', 'attention_mask', 'scene', 'state', 'eval_sequence', 'initial_state']
    test_to_tensor = ['input_ids', 'attention_mask', 'scene', 'state']
    train_pipeline = [
    # dict(type='LoadADE20KMask', prefix="images"),
    # dict(type='Resize', scale=(256, 256), backend='pillow'),
    # dict(type='MaskPromptSelectADE20k', only_mask=True),
    # dict(type='RandomResizedCrop', scale=2, crop_ratio_range=(0.8, 1.0)),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='GPT2TokenizerforCALVINEnvVal',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        pred_image=True,
        test_mode=True
    ),
    # dict(
    #     type='LlamaTokenizerforMask',
    #     pretrained='./tokenizer/llama/open_3B_v2/',
    #     input_text='prompt',
    # ),
    dict(type='Collect', keys=['img', *test_aux_info]),
    dict(type='ToTensor', keys=['img', *test_to_tensor]),
]

    # dataset = CALVINEnvValDataset(data_root="/mnt/bn/panxuran/calvin/calvin_debug_dataset/",  pipeline=train_pipeline)
    dataset = CALVINEnvValDataset(data_root="/mnt/bn/panxuran/calvin/task_ABCD_D/",  pipeline=train_pipeline)

    for data in dataset:
        pass
