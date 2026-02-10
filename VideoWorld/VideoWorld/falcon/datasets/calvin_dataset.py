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
from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
from falcon.registry import DATASETS, TRANSFORMS
import numpy as np
import copy
import torch
import imageio
import cv2
import os
import random
import pybullet 
target_lang =  ["grasp the drawer handle, then open it",
               "grasp the drawer handle and open it",
               "grasp the handle of the drawer, then open it",
               "grasp the handle of the drawer and open it",
               "open the drawer",
               "go open the drawer",
               "pull the handle of the drawer",
               "pull the drawer",
               "open the cabinet drawer",
                "grasp the drawer handle, then close it",
               "grasp the drawer handle and close it",
               "grasp the handle of the drawer, then close it",
               "grasp the handle of the drawer and close it",
               "close the drawer",
               "go close the drawer",
               "push the handle of the drawer",
               "push the drawer",
               "close the cabinet drawer"]

task_lang_dict = {
# rotation
"rotate_red_block_right": ["grasp the red block, then rotate it right",
                         "grasp the red block, then turn it right",
                         "grasp the red block and rotate it right",
                         "grasp the red block and turn it right",
                         "take the red block and rotate it right",
                         "take the red block and turn it right",
                         "rotate right the red block",
                         "rotate the red block 90 degrees to the right",
                         "rotate the red block to the right",
                         "rotate the red block towards the right",
                         "turn the red block right"],
"rotate_red_block_left": ["grasp the red block, then rotate it left",
                         "grasp the red block, then turn it left",
                         "grasp the red block and rotate it left",
                         "grasp the red block and turn it left",
                         "take the red block and rotate it left",
                         "take the red block and turn it left",
                         "rotate the red block 90 degrees to the left",
                         "rotate left the red block",
                         "rotate the red block to the left",
                         "rotate the red block towards the left",
                         "turn the red block left"],
'rotate_blue_block_right': ["grasp the blue block, then rotate it right",
                         "grasp the blue block, then turn it right",
                         "grasp the blue block and rotate it right",
                         "grasp the blue block and turn it right",
                         "take the blue block and rotate it right",
                         "take the blue block and turn it right",
                         "rotate the blue block 90 degrees to the right",
                         "rotate right the blue block",
                         "rotate the blue block to the right",
                         "rotate the blue block towards the right",
                         "turn the blue block right"],
'rotate_blue_block_left': ["grasp the blue block, then rotate it left",
                         "grasp the blue block, then turn it left",
                         "grasp the blue block and rotate it left",
                         "grasp the blue block and turn it left",
                         "take the blue block and rotate it left",
                         "take the blue block and turn it left",
                         "rotate the blue block 90 degrees to the left",
                         "rotate left the blue block",
                         "rotate the blue block to the left",
                         "rotate the blue block towards the left",
                         "turn the blue block left"],
'rotate_pink_block_right': ["grasp the pink block, then rotate it right",
                         "grasp the pink block, then turn it right",
                         "grasp the pink block and rotate it right",
                         "grasp the pink block and turn it right",
                         "take the pink block and rotate it right",
                         "take the pink block and turn it right",
                         "rotate the pink block 90 degrees to the right",
                         "rotate right the pink block",
                         "rotate the pink block to the right",
                         "rotate the pink block towards the right",
                         "turn the pink block right"],
'rotate_pink_block_left': ["grasp the pink block, then rotate it left",
                         "grasp the pink block, then turn it left",
                         "grasp the pink block and rotate it left",
                         "grasp the pink block and turn it left",
                         "take the pink block and rotate it left",
                         "take the pink block and turn it left",
                         "rotate the pink block 90 degrees to the left",
                         "rotate left the pink block",
                         "rotate the pink block to the left",
                         "rotate the pink block towards the left",
                         "turn the pink block left"],

# sliding
'push_red_block_right': ["push the red block towards the right",
                       "push right the red block",
                       "push the red block to the right",
                       "go push the red block to the right",
                       "slide the red block towards the right",
                       "slide right the red block",
                       "slide the red block to the right",
                       "sweep the red block to the right",
                       "go slide the red block to the right"],
'push_red_block_left': ["push the red block towards the left",
                       "push left the red block",
                       "push the red block to the left",
                       "go push the red block to the left",
                       "slide the red block towards the left",
                       "slide left the red block",
                       "slide the red block to the left",
                       "sweep the red block to the left",
                       "go slide the red block to the left"],
'push_blue_block_right': ["push the blue block towards the right",
                       "push right the blue block",
                       "push the blue block to the right",
                       "go push the blue block to the right",
                       "slide the blue block towards the right",
                       "slide right the blue block",
                       "slide the blue block to the right",
                       "sweep the blue block to the right",
                       "go slide the blue block to the right"],
'push_blue_block_left': ["push the blue block towards the left",
                       "push left the blue block",
                       "push the blue block to the left",
                       "go push the blue block to the left",
                       "slide the blue block towards the left",
                       "slide left the blue block",
                       "slide the blue block to the left",
                       "sweep the blue block to the left",
                       "go slide the blue block to the left"],
'push_pink_block_right': ["push the pink block towards the right",
                       "push right the pink block",
                       "push the pink block to the right",
                       "go push the pink block to the right",
                       "slide the pink block towards the right",
                       "slide right the pink block",
                       "slide the pink block to the right",
                       "sweep the pink block to the right",
                       "go slide the pink block to the right"],
'push_pink_block_left': ["push the pink block towards the left",
                       "push left the pink block",
                       "push the pink block to the left",
                       "go push the pink block to the left",
                       "slide the pink block towards the left",
                       "slide left the pink block",
                       "slide the pink block to the left",
                       "sweep the pink block to the left",
                       "go slide the pink block to the left"],
# open/close
'move_slider_left': [ "grasp the door handle, then slide the door to the left",
                    "grasp the door handle, then move the door to the left",
                    "grasp the door handle and slide the door to the left",
                    "grasp the door handle and move the door to the left",
                    "move the door all the way to the left",
                    "slide the door all the way to the left",
                    "move the door to the left",
                    "slide the door to the left",
                    "push the door to the left",
                    "move the door to the left side",
                    "slide the door to the left side",
                    "push the door to the left side",
                    "slide the door to the left, then let it go",
                    "move the door all the way to the left and let go",
                    "move the sliding door to the left",
                    "push the sliding door to the left"],
'move_slider_right': [ "grasp the door handle, then slide the door to the right",
                    "grasp the door handle, then move the door to the right",
                    "grasp the door handle and slide the door to the right",
                    "grasp the door handle and move the door to the right",
                    "move the door all the way to the right",
                    "slide the door all the way to the right",
                    "move the door to the right",
                    "slide the door to the right",
                    "push the door to the right",
                    "move the door to the right side",
                    "slide the door to the right side",
                    "push the door to the right side",
                    "slide the door to the right, then let it go",
                    "move the door all the way to the right and let go",
                    "move the sliding door to the right",
                    "push the sliding door to the right"],
'open_drawer': [ "grasp the drawer handle, then open it",
               "grasp the drawer handle and open it",
               "grasp the handle of the drawer, then open it",
               "grasp the handle of the drawer and open it",
               "open the drawer",
               "go open the drawer",
               "pull the handle of the drawer",
               "pull the drawer",
               "open the cabinet drawer"],
'close_drawer': [ "grasp the drawer handle, then close it",
               "grasp the drawer handle and close it",
               "grasp the handle of the drawer, then close it",
               "grasp the handle of the drawer and close it",
               "close the drawer",
               "go close the drawer",
               "push the handle of the drawer",
               "push the drawer",
               "close the cabinet drawer"],
# lifting
'lift_red_block_table': ["lift the red block from the table",
                       "pick up the red block on the table",
                       "pick up the red block from the table",
                       "lift the red block",
                       "pick up the red block",
                       "lift the red block up",
                       "grasp the red block on the table and lift it up",
                       "grasp the red block and lift it up",
                       "grasp the red block on the table, then lift it up",
                       "grasp the red block, then lift it up"],
'lift_blue_block_table': ["lift the blue block from the table",
                       "pick up the blue block on the table",
                       "pick up the blue block from the table",
                       "lift the blue block",
                       "pick up the blue block",
                       "lift the blue block up",
                       "grasp the blue block on the table and lift it up",
                       "grasp the blue block and lift it up",
                       "grasp the blue block on the table, then lift it up",
                       "grasp the blue block, then lift it up"],
'lift_pink_block_table': ["lift the pink block from the table",
                       "pick up the pink block on the table",
                       "pick up the pink block from the table",
                       "lift the pink block",
                       "pick up the pink block",
                       "lift the pink block up",
                       "grasp the pink block on the table and lift it up",
                       "grasp the pink block and lift it up",
                       "grasp the pink block on the table, then lift it up",
                       "grasp the pink block, then lift it up"],

'lift_red_block_slider': [ "pick up the red block from the shelf",
                         "pick up the red block from the sliding cabinet",
                         "pick up the red block in the sliding cabinet",
                         "grasp the red block lying on the shelf",
                         "grasp the red block lying in the cabinet",
                         "grasp the red block lying in the sliding cabinet",
                         "grasp the red block lying in the slider",
                         "lift the red block lying on the shelf",
                         "lift the red block lying in the cabinet",
                         "lift the red block lying in the sliding cabinet",
                         "lift the red block lying in the slider",
                         "in the slider pick up the red block",
                         "in the cabinet pick up the red block",
                         "in the slider grasp the red block",
                         "in the cabinet grasp the red block",
                         "in the sliding cabinet grasp the red block",
                         "lift the red block on the shelf"],
'lift_blue_block_slider': [ "pick up the blue block from the shelf",
                         "pick up the blue block from the sliding cabinet",
                         "pick up the blue block in the sliding cabinet",
                         "grasp the blue block lying on the shelf",
                         "grasp the blue block lying in the cabinet",
                         "grasp the blue block lying in the sliding cabinet",
                         "grasp the blue block lying in the slider",
                         "lift the blue block lying on the shelf",
                         "lift the blue block lying in the cabinet",
                         "lift the blue block lying in the sliding cabinet",
                         "lift the blue block lying in the slider",
                         "in the slider pick up the blue block",
                         "in the cabinet pick up the blue block",
                         "in the slider grasp the blue block",
                         "in the cabinet grasp the blue block",
                         "in the sliding cabinet grasp the blue block",
                         "lift the blue block on the shelf"],
'lift_pink_block_slider': ["pick up the pink block from the shelf",
                         "pick up the pink block from the sliding cabinet",
                         "pick up the pink block in the sliding cabinet",
                         "grasp the pink block lying on the shelf",
                         "grasp the pink block lying in the cabinet",
                         "grasp the pink block lying in the sliding cabinet",
                         "grasp the pink block lying in the slider",
                         "lift the pink block lying on the shelf",
                         "lift the pink block lying in the cabinet",
                         "lift the pink block lying in the sliding cabinet",
                         "lift the pink block lying in the slider",
                         "in the slider pick up the pink block",
                         "in the cabinet pick up the pink block",
                         "in the slider grasp the pink block",
                         "in the cabinet grasp the pink block",
                         "in the sliding cabinet grasp the pink block",
                         "lift the pink block on the shelf"],

'lift_red_block_drawer': ["grasp the red block from the drawer",
                        "grasp the red block lying in the drawer",
                        "grasp the red block in the drawer",
                        "pick up the red block lying in the drawer",
                        "pick up the red block from the drawer",
                        "pick up the red block in the drawer",
                        "go towards the red block in the drawer and pick it up",
                        "go towards the red block in the drawer and grasp it",
                        "go towards the red block in the drawer and lift it",
                        "lift the red block in the drawer",
                        "lift the red block lying in the drawer"],
'lift_blue_block_drawer': ["grasp the blue block from the drawer",
                        "grasp the blue block lying in the drawer",
                        "grasp the blue block in the drawer",
                        "pick up the blue block lying in the drawer",
                        "pick up the blue block from the drawer",
                        "pick up the blue block in the drawer",
                        "go towards the blue block in the drawer and pick it up",
                        "go towards the blue block in the drawer and grasp it",
                        "go towards the blue block in the drawer and lift it",
                        "lift the blue block in the drawer",
                        "lift the blue block lying in the drawer"],
'lift_pink_block_drawer': ["grasp the pink block from the drawer",
                        "grasp the pink block lying in the drawer",
                        "grasp the pink block in the drawer",
                        "pick up the pink block lying in the drawer",
                        "pick up the pink block from the drawer",
                        "pick up the pink block in the drawer",
                        "go towards the pink block in the drawer and pick it up",
                        "go towards the pink block in the drawer and grasp it",
                        "go towards the pink block in the drawer and lift it",
                        "lift the pink block in the drawer",
                        "lift the pink block lying in the drawer"],

'place_in_slider': [ "place in slider",
                   "put it in the slider",
                   "place the block in the sliding cabinet",
                   "place the object in the sliding cabinet",
                   "place the grasped object in the sliding cabinet",
                   "put the block in the sliding cabinet",
                   "put the object in the sliding cabinet",
                   "put the grasped object in the sliding cabinet",
                   "place the block in the cabinet",
                   "place the object in the cabinet",
                   "place the grasped object in the cabinet",
                   "put the block in the cabinet",
                   "put the object in the cabinet",
                   "put the grasped object in the cabinet",
                   "place the block in the slider",
                   "place the object in the slider",
                   "place the grasped object in the slider",
                   "put the block in the slider",
                   "put the object in the slider",
                   "put the grasped object in the slider"],

'place_in_drawer': [ "place the block in the drawer",
                   "place the object in the drawer",
                   "place the grasped object in the drawer",
                   "put the block in the drawer",
                   "put the object in the drawer",
                   "put the grasped object in the drawer",
                   "store the block in the drawer",
                   "store the object in the drawer",
                   "store the grasped object in the drawer",
                   "move to the drawer and place the object",
                   "go towards the drawer and place the object",
                   "move to the drawer, then place the object",
                   "move to the drawer and store the object",
                   "go towards the drawer and store the object",
                   "move to the drawer, then store the object"],

'push_into_drawer': ["push the object into the drawer",
                   "push the block into the drawer",
                   "slide the object into the drawer",
                   "slide the block into the drawer",
                   "sweep the object into the drawer",
                   "sweep the block into the drawer",
                   "push the object that it falls into the drawer"],

'stack_block': ["stack blocks on top of each other",
              "stack the blocks",
              "stack the object on top of another object",
              "place the block on top of another block",
              "place the grasped block on top of another block",
              "put the grasped block on top of a block",
              "put the block on top of another block",
              "stack the block on top of another block"],
'unstack_block': ["collapse the stacked blocks",
                "take off the stacked block",
                "unstack the blocks",
                "go to the tower of blocks and take off the top one",
                "remove a block from the stack",
                "take off the block that is on top of the other one",
                "remove the top block"],

'turn_on_lightbulb': ["turn on the light bulb",
                    "turn on the yellow light",
                    "turn on the yellow lamp",
                    "move up the switch",
                    "push the switch upwards",
                    "slide up the switch",
                    "move the light switch to turn on the light bulb",
                    "toggle the light switch to turn on the light bulb",
                    "move the light switch to turn on the yellow light",
                    "toggle the light switch to turn on the yellow light"],

'turn_off_lightbulb': ["turn off the light bulb",
                    "turn off the yellow light",
                    "turn off the yellow lamp",
                    "move down the switch",
                    "push the switch downwards",
                    "slide down the switch",
                    "move the light switch to turn off the light bulb",
                    "toggle the light switch to turn off the light bulb",
                    "move the light switch to turn off the yellow light",
                    "toggle the light switch to turn off the yellow light"],

'turn_on_led': ["turn on the led light",
              "turn on the led",
              "turn on the led lamp",
              "turn on the green light",
              "turn on the green lamp",
              "push down the button to turn on the led light",
              "push down the button to turn on the led",
              "push down the button to turn on the green light",
              "push the button to turn on the led light",
              "push the button to turn on the led",
              "push the button to turn on the green light",
              "toggle the button to turn on the led light",
              "toggle the button to turn on the led",
              "toggle the button to turn on the green light"],

'turn_off_led': ["turn off the led light",
              "turn off the led",
              "turn off the led lamp",
              "turn off the green light",
              "turn off the green lamp",
              "push down the button to turn off the led light",
              "push down the button to turn off the led",
              "push down the button to turn off the green light",
              "push the button to turn off the led light",
              "push the button to turn off the led",
              "push the button to turn off the green light",
              "toggle the button to turn off the led light",
              "toggle the button to turn off the led",
              "toggle the button to turn off the green light"]

}


def normalize(img, mean, std, to_bgr=False):
    # import pdb;pdb.set_trace()
    # print(img.shape)
    img = img.copy().astype(np.float32)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1. / np.float64(std.reshape(1, -1))
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.subtract(img, mean)
    img = cv2.multiply(img, stdinv)
    return img

def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    if gripper[0] < -1:
        gripper[0] = -1
    if gripper[0] > 1:
        gripper[0] = 1
    return np.concatenate([rel_pos, rel_orn, gripper])
    
@DATASETS.register_module()
class CALVINDataset(Dataset):

    def __init__(
        self,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
        obs_type='rgb_static',
        state_type='robot_obs',
        action_type='rel_actions',
        clip_length=1,
        interval_range=[5, 15],
        multiply_coefficient=True,
        use_hand=False,
        check_val_loss=False,
        calvin_scene = 'all',
        size=256,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        la_path=None,
        all_len=False,
    ):
        self.data_root = data_root
        self.clip_length = clip_length
        self.interval_range = interval_range
        self.multiply_coefficient = multiply_coefficient
        self.size= size
        self.mean, self.std = mean, std
        self.la_path = la_path
        self.all_len = all_len

        scene_info_path = osp.join(data_root, 'scene_info.npy')
        lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        scene_info = np.load(scene_info_path, allow_pickle=True).item()
        # import pdb;pdb.set_trace()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["emb"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        new_annotations = []
        if calvin_scene == 'D':
            for ann in annotations:
                index = ann[0]
                if index[1] > 611098:
                    continue
                
                if ann[1] in target_lang:
                    new_annotations.append(ann)
            annotations = new_annotations

        
        self.data_infors = annotations 
        self.obs_type = obs_type
        self.state_type = state_type
        self.action_type = action_type
        self.test = test_mode
        self.use_hand = use_hand
        self.check_val_loss = check_val_loss
        
        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

        # load la action annotation data if needed
        if self.la_path is not None:
            self.la_anno = torch.load(self.la_path)

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
        print(ann[1])
        ep_len = end_index - start_index + 1
        if self.clip_length == 1:
            sample_frame = random.sample(list(range(start_index, end_index+1)), 1)[0]
            frame_path = f"{self.data_root}/episode_{sample_frame:07d}.npz"
            t = np.load(frame_path, allow_pickle=True)
            img = t[self.obs_type]
            item['img'] = img
            item['original_shape'] = img.shape[:2]
            item['img_shape'] = img.shape[:2]
            
        else:
            # import pdb;pdb.set_trace()
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
            if try_num > max_try:
                interval = 1
            sample_start = random.sample(list(range(start_index, end_index+1-(interval*(self.clip_length-1)))), 1)[0]

            imgs = []
            hands = []
            states = []
            actions = []
            la_actions = []
            size = self.size
            if self.all_len:
                sample_start = start_index
                sample_clip_length = end_index - start_index
            else:
                sample_clip_length = self.clip_length
            for i in range(sample_clip_length):
                index = sample_start + (i * interval)
                frame_path = f"{self.data_root}/episode_{index:07d}.npz"
                t = np.load(frame_path, allow_pickle=True)
                img = t[self.obs_type]
                state = np.concatenate((t[self.state_type][:6], t[self.state_type][-1:]))
                action = t[self.action_type]
                if self.la_path is not None and i != sample_clip_length-1:
                    la_action = self.la_anno[index]
                    la_actions.append(la_action['la_action'])
                img = mmcv.imresize(img, (size, size), interpolation='bilinear', backend='pillow')
                if self.use_hand:
                    hand = t['rgb_gripper']
                    hand = mmcv.imresize(hand, (size, size), interpolation='bilinear', backend='pillow')
                    hand = hand / 255.
                    hand = normalize(hand, np.array(self.mean), np.array(self.std))
                    # hand = (hand / 127.5) - 1.0
                    hands.append(hand)

                imgs.append(img)
                states.append(state)
                actions.append(action)

            item['img'] = np.stack(imgs) 
            item['state'] = np.stack(states)
            item['action'] = np.stack(actions)
            if self.la_path is not None:
                item['la_action'] = np.stack(la_actions)
            item['original_shape'] = imgs[0].shape[:2]
            item['img_shape'] = imgs[0].shape[:2]
            if self.use_hand:
                hand = np.stack(hands)
                hand = np.transpose(hand, (0, 3, 1, 2))
                item['hand'] = hand

        if self.multiply_coefficient:
            item['action'][:, :3] *= 50
            item['action'][:, 3:6] *= 20

        item['gt_label'] = 0
        item['prompt'] = ann[1]
        item['lang_emb'] = ann[2]
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
        check_val_loss=False,
        multiply_coefficient=True,
        calvin_scene='all',
        show=False
    ):
        self.data_root = data_root
        self.show = show
        scene_info_path = osp.join(data_root, 'scene_info.npy')
        lang_info_path = osp.join(data_root, 'lang_annotations/auto_lang_ann.npy')
        annotations = np.load(lang_info_path, allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["emb"])) #((np.int64(1401659), np.int64(1401723)), 'move the door all the way to the right')
        new_annotations = []
        # import pdb;pdb.set_trace()
        if calvin_scene == 'D':
            for ann in annotations:
                index = ann[0]
                if index[1] > 611098:
                    continue
                new_annotations.append(ann)
            annotations = new_annotations
        self.data_infors = annotations
        self.check_val_loss = check_val_loss
        self.obs_type = obs_type
        self.multiply_coefficient = multiply_coefficient
        # self.data_infors = self.load_annotations(osp.join(data_root, ann_file))

        self.test = test_mode

        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.vis_root = '/opt/tiger/check_train_set/GT'
        os.system(f'mkdir -p {self.vis_root}')

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
        index = ann[0]
        data_paths = []
        if self.show:
            vis_path = f"{self.vis_root}/{ann[1].replace(' ', '-')}"
            os.system(f'mkdir -p {vis_path}')
            subtask_idx = len(os.listdir(vis_path))
            subtask_vis_path = f"{self.vis_root}/{ann[1].replace(' ', '-')}/{subtask_idx}"
            os.system(f'mkdir -p {subtask_vis_path}')
            for i in range(start_index, end_index+1):
                path = f"{self.data_root}/episode_{i:07d}.npz"
                data_paths.append(path)
                t = np.load(path, allow_pickle=True)
                img = t[self.obs_type]
                cv2.imwrite(f'{subtask_vis_path}/{i}.jpg', img[:, :, ::-1])
            
            with imageio.get_writer(uri=f'{subtask_vis_path}/{start_index}.gif', mode='I', fps=10) as writer:
                target_file = subtask_vis_path
                for i in range(start_index, end_index+1):
                    writer.append_data(imageio.imread(f'{target_file}/{i}.jpg'))
                
        frame_path = data_paths[0]
        t = np.load(frame_path, allow_pickle=True)
        img = t[self.obs_type]
        state = np.concatenate((t['robot_obs'][:6], t['robot_obs'][-1:]))
        action = t['rel_actions']
        item['img'] = img
        item['scene'] = t['scene_obs']
        item['state'] = t['robot_obs']
        item['prefix'] = ''
        item['data_root'] = self.data_root
        item['original_shape'] = img.shape[:2]
        item['img_shape'] = img.shape[:2]
        item['gt_label'] = 0
        item['lang_emb'] = ann[2]
        item['eval_sequence'] = [ann[1]]
        item['prompt'] = (ann[1], )
        item['episode_infos'] = data_paths
        if self.check_val_loss:
            item['check_val_loss'] = True
        # item['initial_state'] = initial_state
        item = self.pipeline(item)
        return item


