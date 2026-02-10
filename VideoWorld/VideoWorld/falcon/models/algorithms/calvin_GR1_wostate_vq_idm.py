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
import torch
import os
from falcon.models.quantizer.vqgan import VQGANDecoder
from falcon.models.quantizer.connector import ConvConnector
from falcon.registry import MODELS
from .base import BaseModel
import mmcv
import numpy as np
import cv2
import sgfmill
import clip
import copy
import hydra
IMAGE_TOKEN_INDEX = -200
MASK_TOKEN_INDEX = -300
ACT_TOKEN_INDEX = -400
import random
import matplotlib.pyplot as plt
import subprocess
import imageio
import json
import torch.nn.functional as F
import torch.nn as nn


import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
CALVIN_ROOT = '../calvin'
EP_LEN = 360
NUM_SEQUENCES = 1000

def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from falcon.utils.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env

def visualize(imgs, masks, preds=None, rec_masks=None):
    mean=torch.tensor([0.5, 0.5, 0.5]).to(imgs)
    std=torch.tensor([0.5, 0.5, 0.5]).to(imgs)
    import numpy as np
    import cv2
    # import pdb;pdb.set_trace()
    import copy
    _masks = copy.deepcopy(masks)
    for idx, (_img, mask) in enumerate(zip(imgs, _masks)):
        _img = _img.permute(1, 2, 0)
        _img = (_img * std + mean) * 255
        _img = _img.cpu().numpy().astype(np.uint8)
        cv2.imwrite('/opt/tiger/visualize1/image_{}.jpg'.format(idx), _img[:,:,::-1])
        if mask.shape[0] == 3:
            mask = mask.permute(1, 2, 0)

        mask = mask.cpu().numpy().astype(np.uint8)
        cv2.imwrite('/opt/tiger/visualize1/mask_{}.jpg'.format(idx), mask[:,:,::-1])

    if rec_masks is not None:
        _rec_masks = copy.deepcopy(rec_masks)
        for idx, rec_mask in enumerate(_rec_masks):
            if rec_mask.shape[0] == 3:
                rec_mask = rec_mask.permute(1, 2, 0)
            rec_mask = rec_mask.cpu().numpy().astype(np.uint8)
            cv2.imwrite('/opt/tiger/visualize/rec_mask_{}.jpg'.format(idx), rec_mask[:,:,::-1])

    if preds is not None:
        _preds = copy.deepcopy(preds)
        for idx, pred in enumerate(_preds):
            if pred.shape[0] == 3:
                pred = pred.permute(1, 2, 0)
            pred = pred.cpu().numpy().astype(np.uint8)
            cv2.imwrite('/opt/tiger/visualize1/pred_{}.jpg'.format(idx), pred[:,:,::-1])

@MODELS.register_module()
class VideoWorldRobotics(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, pre_encode_lang=False, sup_actions=True, use_clip_lang=False, 
                 test_wo_generate=False, seq_length=1, use_time_embedding=True, training_target=['act_pred', 'fwd_pred', 'fwd_pred_hand'], vq_decoder_cfg={}, 
                 max_generate_length=300, max_new_tokens=100, work_dir='/opt/tiger', worker_index=0, use_img_start=False, use_la_action=False, la_act_scope=1,
                 ):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

        # import pdb;pdb.set_trace()
        self.vbackbone = MODELS.build(vbackbone)
        self.vq_num = neck['vq_num']
        self.neck = MODELS.build(neck)

        self.head = MODELS.build(head)
        self.v_token = MODELS.build(quantizer)

        self.post_encode = ConvConnector(in_channels=256, out_channels=256)
        self.pre_decode = ConvConnector(in_channels=256, out_channels=256)
        self.v_token_decoder = VQGANDecoder(**vq_decoder_cfg)
        self.pred_image = pred_image
        self.pred_action = pred_action
        self.work_dir = work_dir
        self.max_generate_length = max_generate_length
        self.max_new_tokens = max_new_tokens
        self.test_wo_generate = test_wo_generate
        self.use_clip_lang = use_clip_lang
        self.seq_length = seq_length
        self.use_time_embedding = use_time_embedding
        self.use_img_start = use_img_start
        self.use_la_action = use_la_action
        state_dict = torch.load(vbackbone['init_cfg']['checkpoint'])['state_dict']
        encoder_state_dict = {}
        decoder_state_dict = {}
        post_encode_state_dict = {}
        pre_decode_state_dict = {}
        quantizer_state_dict = {}
        # import pdb;pdb.set_trace()
        for k in state_dict:
            if 'generator.encoder' in k:
                encoder_state_dict[k[len('generator.encoder.'):]] = state_dict[k]
            if 'generator.decoder' in k:
                key = k[len('generator.encoder.'):]
                decoder_state_dict[key] = state_dict[k]
            if 'generator.pre_decoder' in k:
                pre_decode_state_dict[k[len('generator.pre_decoder.'):]] = state_dict[k]
            if 'generator.post_encoder' in k:
                post_encode_state_dict[k[len('generator.post_encoder.'):]] = state_dict[k]
            if 'generator.quantizer' in k:
                quantizer_state_dict[k[len('generator.quantizer.'):]] = state_dict[k]
        # import pdb;pdb.set_trace()
        self.vbackbone.load_state_dict(encoder_state_dict)
        self.v_token_decoder.load_state_dict(decoder_state_dict)
        self.v_token.load_state_dict(quantizer_state_dict)
        self.post_encode.load_state_dict(post_encode_state_dict)
        self.pre_decode.load_state_dict(pre_decode_state_dict)

        for p in self.v_token.parameters():
            p.requires_grad = False
        for p in self.vbackbone.parameters():
            p.requires_grad = False
        for p in self.v_token_decoder.parameters():
            p.requires_grad = False
        for p in self.post_encode.parameters():
            p.requires_grad = False
        for p in self.pre_decode.parameters():
            p.requires_grad = False

        # if use_clip_lang:
        self.model_clip, _ = clip.load('ViT-B/32', device='cuda')
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False
        self.tokenizer = clip.tokenize

        variant = {"embed_dim": 384, "n_layer": 12, "n_head": 12, "activation_function": "relu", "dropout": 0.1, "n_positions": 1024, "device": "cuda", "resampler_depth": 3, "resampler_dim_head": 128, "resampler_heads": 4, "resampler_num_media_embeds": 1, "resampler_num_latents": 9, "seq_len": 10, "action_mode": "ee_rel_pose", "act_dim": 7, "state_dim": 7, "use_hand_rgb": True, "clip_backbone": "ViT-B/32", "act_pred": True, "fwd_pred": True, "fwd_pred_hand": True, "fwd_pred_next_n": 3, "without_norm_pix_loss": False, "img_feat_dim": 768, "patch_feat_dim": 768, "lang_feat_dim": 512}
        resampler_params = dict()
        resampler_params['depth'] = variant['resampler_depth']
        resampler_params['dim_head'] = variant['resampler_dim_head']
        resampler_params['heads'] = variant['resampler_heads']
        resampler_params['num_latents'] = variant['resampler_num_latents']
        resampler_params['num_media_embeds'] = variant['resampler_num_media_embeds']
        variant['resampler_params'] = resampler_params
        state_dim=variant['state_dim']
        act_dim=variant['act_dim']
        hidden_size= 1024
        sequence_length=variant['seq_len']
        training_target=training_target
        img_feat_dim=variant['img_feat_dim']
        lang_feat_dim=variant['lang_feat_dim']
        patch_feat_dim=variant['patch_feat_dim']
        resampler_params=variant['resampler_params']
        without_norm_pix_loss=variant['without_norm_pix_loss']
        use_hand_rgb=variant['use_hand_rgb']
        n_layer=variant['n_layer']
        n_head=variant['n_head']
        n_inner=4*variant['embed_dim']
        activation_function=variant['activation_function']
        n_positions=variant['n_positions']
        resid_pdrop=variant['dropout']
        attn_pdrop=variant['dropout']

        self.state_dim, self.act_dim, self.hidden_size, self.sequence_length, self.img_feat_dim, self.lang_feat_dim, self.patch_feat_dim, self.use_hand_rgb  = state_dim, act_dim, hidden_size, sequence_length, img_feat_dim, lang_feat_dim, patch_feat_dim, use_hand_rgb

        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        
        if self.use_la_action:
            self.act_pred = False

        self.without_norm_pixel_loss = False

        # Embedding functions for states
        # self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        # self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
        # self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)

        # Relative timestep embedding
        if use_time_embedding:
            self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)


    
        if self.act_pred:
            # Action prediction
            self.action_queries = nn.Embedding(1, hidden_size) # arm + gripper
            self.pred_act_mlps = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size//2),
                nn.Linear(hidden_size//2, hidden_size//2)])
            self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
            self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
            self.state_loss = nn.SmoothL1Loss()
            self.gripper_loss = nn.BCEWithLogitsLoss()

        # self.rgb_loss = nn.MSELoss()
        # import pdb;pdb.set_trace()

        self.la_act_scope = la_act_scope
        
        #IDM
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.act_embed_fuse = nn.Linear(self.la_act_scope+1, 1, bias=False)
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)

        data_root = "./data/calvin/task_ABCD_D/"
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
        self.eval_env = make_env(data_root, observation_space, current_rank)
        conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        self.val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        os.system('mkdir -p ./visualize_calvin')
        
    def check_rec(self, visual_ids, visual_hand_ids, img, img_hand, is_rollout=False, ep_idx=0):
        
        b, t = visual_hand_ids.shape[:2]
        visual_hand_ids = visual_hand_ids.flatten(0, 1)

        rec_visual_hand_ids = visual_hand_ids.reshape(-1, 4, 4)
        rec_visual_hand_ids = self.v_token.indices_to_codes(rec_visual_hand_ids)
        rec_visual_hand_ids = self.pre_decode(rec_visual_hand_ids)
        rec_visual_hand_ids = self.v_token_decoder(rec_visual_hand_ids)
        rec_visual_hand = torch.clamp((rec_visual_hand_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_hand = rec_visual_hand.view(b, t, 256, 256, 3)

        rec_visual_ids = visual_ids.reshape(-1, 4, 4)
        rec_visual_ids = self.v_token.indices_to_codes(rec_visual_ids)
        rec_visual_ids = self.pre_decode(rec_visual_ids)
        rec_visual_ids = self.v_token_decoder(rec_visual_ids)
        rec_visual_ids = torch.clamp((rec_visual_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        rec_visual_ids = rec_visual_ids.view(b, t, 256, 256, 3)
        for idx, (bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand) in enumerate(zip(rec_visual_ids, img, rec_visual_hand, img_hand)):
            import cv2
            import numpy as np
            for fi, (rec_visual_id, gt_img, hand, gt_hand) in enumerate(zip(bz_rec_visual_id, bz_gt_img, bz_hand, bz_gt_hand)):
                gt_hand = torch.clamp(((gt_hand+1)*127), min=0, max=255)
                gt_hand = gt_hand.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                hand = hand.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                rec_visual_id = rec_visual_id.cpu().numpy()[:, :, ::-1].astype(np.uint8)
                gt_img = torch.clamp(((gt_img+1)*127), min=0, max=255)
                gt_img = gt_img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
                show0 = np.concatenate((gt_img, gt_hand), axis=0)
                show1 = np.concatenate((rec_visual_id, hand), axis=0)
                show = np.concatenate((show0, show1), axis=1)
                if is_rollout and fi == (len(bz_rec_visual_id)-1):
                    os.system(f'mkdir /opt/tiger/rec_visualize/{ep_idx}')
                    frame_length = len(os.listdir(f'/opt/tiger/rec_visualize/{ep_idx}'))
                    cv2.imwrite(f'/opt/tiger/rec_visualize/{ep_idx}/{frame_length}.jpg', show)
                

    def de_normalize(self, img, mean, std):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = 0 - np.float64(mean.reshape(1, -1))
        stdinv = np.float64(std.reshape(1, -1))
        img = cv2.multiply(img, stdinv)
        img = cv2.subtract(img, mean)
        img = img * 255.
        img = np.clip(img, a_min=0, a_max=255)
        img = img.astype(np.uint8)
        return img

    def normalize(self, img, mean, std):
        img = img.copy().astype(np.float32) / 255.
        assert img.dtype != np.uint8
        mean = mean = np.float64(mean.reshape(1, -1))
        stdinv = 1. / np.float64(std.reshape(1, -1))
        img = cv2.subtract(img, mean)
        img = cv2.multiply(img, stdinv)
        return img

    def visualize(self, img, hand):
        # import pdb;pdb.set_trace()
        for bi, (b_img, b_hand) in enumerate(zip(img, hand)):
            for fi, (t_img, t_hand) in enumerate(zip(b_img, b_hand)):
                _img = self.de_normalize(t_img.permute(1, 2, 0).cpu().numpy(), mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
                _hand = self.de_normalize(t_hand.permute(1, 2, 0).cpu().numpy(), mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
                show = np.concatenate((_img, _hand))
                cv2.imwrite(f'/opt/tiger/train_vis/{bi}_{fi}.jpg', show[:, :, ::-1])

    def embed_func(self):
        return self.neck.llm.model.embed_tokens

    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        episode_infos = kwargs.get('episode_infos', None)
        check_val_loss = kwargs.get('check_val_loss', None)
        return self.rollout_pred_rgb(img, input_ids, pred_label, attention_mask, index, **kwargs)
    
    def angle_between_angles(self, a, b):
        diff = b - a
        return (diff + torch.pi) % (2 * torch.pi) - torch.pi

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0
        self.la_indice_list = []
    
    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
        states = kwargs['state'] #B, T, 7
        actions = kwargs['action'] #B, T, 7
        action_idx = kwargs['action_idx'][0][0]
        la_actions = kwargs.get('la_action', None) #B, T-1
        lang_emb = kwargs.get('lang_emb', None) #B, T, 384
        img_hands = kwargs.get('hand', None)
        # arm_state = states[:, :, :6]
        # arm_action = actions[:, :, :6]
        # gripper_state = states[:, :, -1:]
        # gripper_action = actions[:, :, -1:]
        # gripper_state[gripper_state == -1] = 0
        # gripper_action[gripper_action == -1] = 0

        # gripper_state = gripper_state[:, :, 0].long()
        # gripper_state = F.one_hot(gripper_state, num_classes=2).float() #B, T, 2

        langs = kwargs.pop('prompt')
        lang_emb = []
        for lang in langs:
            tokenized_lang = self.tokenizer(lang).to('cuda')
            emb = self.model_clip.encode_text(tokenized_lang)
            lang_emb.append(emb)
        lang_embeddings = torch.stack(lang_emb) #B, 1, 512
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)

        rgb = img
        hand_rgb = img_hands
        # self.visualize(rgb, hand_rgb)
        batch_size, sequence_length, c, h, w = img.shape
        attention_mask = torch.ones(batch_size, sequence_length).long().to(rgb.device)
        label = torch.ones(batch_size, sequence_length).long().to(rgb.device) * -100

        visual_ids = self.encode_image(img) #B, l, 16
        visual_ids_hand = self.encode_image(img_hands) #B, l, 16
       
        if self.use_img_start:
            img_start_ids = torch.ones(batch_size, sequence_length, 1).long().to(visual_ids.device) * 64000
            img_end_ids = torch.ones(batch_size, sequence_length, 1).long().to(visual_ids.device) * 64001
            visual_ids = torch.cat([img_start_ids, visual_ids, visual_ids_hand, img_end_ids], dim=2)
        else:
            visual_ids = torch.cat([visual_ids, visual_ids_hand], dim=2)
        # concat la_actions into visual ids
        # import pdb;pdb.set_trace()
        if self.use_la_action:
            la_actions = la_actions + 64002
            pad_len = 1 if la_actions.shape[1] < visual_ids.shape[1] else 0
            if la_actions.ndim == 3:
                la_actions = torch.cat([la_actions, torch.zeros(batch_size, pad_len, la_actions.shape[-1]).long().to(visual_ids.device)], dim=1)
            else:
                la_actions = torch.cat([la_actions, torch.zeros(batch_size, pad_len).long().to(visual_ids.device)], dim=1).unsqueeze(-1) 
            visual_ids = torch.cat([visual_ids, la_actions], dim=2)

        obs_embeddings = self.embed_func()(visual_ids) #b, l, 16, h
        lang_embeddings = lang_embeddings.view(batch_size, 1, 1, -1).repeat(1, sequence_length, 1, 1)
       
        stacked_inputs = torch.cat((lang_embeddings, obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        

        if self.act_pred:
            action_queries = self.action_queries.weight  # (1, h)
            action_queries = action_queries.view(1, 1, 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, l, 1, h)
            stacked_inputs = torch.cat((stacked_inputs, action_queries), dim=2)  # (b, l, n_tokens, h)

        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 0
        n_patch_tokens = 0
        n_obs_tokens = 17 if self.use_img_start else 16
        n_hand_patch_tokens = 0
        n_hand_obs_tokens = 17 if self.use_img_start else 16
        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
        n_act_pred_tokens = 1
        if self.act_pred or self.use_la_action:
            act_query_token_start_i = n_tokens
            n_tokens += self.la_act_scope
        
        obs_tokens_start_i = n_lang_tokens + n_state_tokens + n_patch_tokens
        # Layer norm
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
        # stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask
        stacked_label = label.view(batch_size, sequence_length, 1)
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        if self.use_hand_rgb:
            if not self.use_la_action:
                stacked_label = stacked_label.repeat(1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens)
                stacked_attention_mask = stacked_attention_mask.repeat(1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens)
                stacked_label[:, :, obs_tokens_start_i:obs_tokens_start_i+(n_obs_tokens+n_hand_obs_tokens)] = visual_ids
            else:
                stacked_label = stacked_label.repeat(1, 1, n_tokens)
                stacked_attention_mask = stacked_attention_mask.repeat(1, 1, n_tokens)
                stacked_attention_mask[:, -1:, -1:] = 0
                stacked_label[:, :, obs_tokens_start_i:] = visual_ids
                stacked_label[:, -1:, -1:] = -100
        if self.act_pred:
            act_query_label = torch.ones((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long).cuda() * -100
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long).cuda()
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
            stacked_label = torch.cat((stacked_label, act_query_label),dim=2)

        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)
        stacked_label = stacked_label.reshape(batch_size, n_tokens * sequence_length)
  
        logits, loss, hidden_state = self.neck(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask, labels=stacked_label, return_dict=True)
        x = hidden_state.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

        losses = { "loss_v": loss}

        return losses


    def rollout_pred_rgb(self, img, seq_input_ids, pred_label=None, seq_attention_mask=None, index=None, **kwargs):
        # import pdb;pdb.set_trace()
        import cv2
        scene = kwargs.pop('scene')
        robot_obs = kwargs.pop('state')
        eval_sequence = kwargs.pop('eval_sequence')
        seq_lang_emb = kwargs.get('lang_emb', None) #B, T, 384
        action_idx = kwargs.pop('action_idx')[0].item()
        init_robot_obs = robot_obs
        init_scene_obs = scene
        
        self.eval_env.reset(robot_obs=robot_obs.cpu().numpy()[0], scene_obs=scene.cpu().numpy()[0])
        episode_infos = kwargs.get('episode_infos', None)
        success_counter = 0
        vis_root = "./visualize_calvin"
        sequence_idx = len(os.listdir(vis_root))
        vis_path = f"{vis_root}/{sequence_idx}"
        os.system(f'mkdir -p {vis_path}')
        # ep_idx = len(os.listdir('/opt/tiger/rec_visualize'))
        la_indice = torch.zeros([0, self.la_act_scope]).to('cuda').long()
        # import pdb;pdb.set_trace()
        for subtask_i, (subtask, input_ids, attention_mask) in enumerate(zip(eval_sequence, seq_input_ids[0], seq_attention_mask[0])):
            subtask_vis_path = f"{vis_path}/{subtask[0].replace(' ', '-')}"
            # if subtask[0].replace(' ', '-') == 'close_drawer':
            #     break
            os.system(f'mkdir -p {subtask_vis_path}')
            self.reset()
            seq_len = 9
            obs = self.eval_env.get_obs()
            start_info = self.eval_env.get_info()
            subtask_success = False
            cv2.imwrite(f'{subtask_vis_path}/0.jpg', obs['rgb_obs']['rgb_static'][:, :, ::-1])
            try:
                print(f'-----------{subtask}: {self.val_annotations[subtask[0]]}-----------')
            except:
                print(f'-----------{subtask}-----------')
            device = input_ids.device
            for step in range(EP_LEN):
                rgb = mmcv.imresize(obs['rgb_obs']['rgb_static'], (256, 256))
                rgb = self.normalize(rgb, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]))
                hand_rgb = mmcv.imresize(obs['rgb_obs']['rgb_gripper'], (256, 256))
                hand_rgb = self.normalize(hand_rgb, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]))
                rgb = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float).to(device)
                hand_rgb = torch.from_numpy(hand_rgb).permute(2, 0, 1).to(torch.float).to(device)
                self.rgb_list.append(rgb)
                self.hand_rgb_list.append(hand_rgb)
                self.la_indice_list.append(la_indice)
                state = obs['robot_obs']
                arm_state = state[:6]
                gripper_state = state[-1]
                state = torch.from_numpy(np.hstack([arm_state, gripper_state])).to(torch.float).to(device)
                self.state_list.append(state)

                buffer_len = len(self.rgb_list)
                if buffer_len > seq_len:
                    self.rgb_list.pop(0)
                    self.hand_rgb_list.pop(0)
                    self.state_list.pop(0)
                    self.la_indice_list.pop(1)
                    buffer_len = len(self.rgb_list)
                # Static RGB
                c, h, w = rgb.shape
                rgb_data = torch.zeros((1, buffer_len, c, h, w))
                rgb_tensor = torch.stack(self.rgb_list, dim=0)  # (l, c, h, w)
                rgb_data[0, :buffer_len] = rgb_tensor
                rgb_data = rgb_data.to(device)
                # Hand RGB
                c, h, w = hand_rgb.shape
                hand_rgb_data = torch.zeros((1, buffer_len, c, h, w))
                hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0)  # (l, c, h, w)
                hand_rgb_data[0, :buffer_len] = hand_rgb_tensor
                hand_rgb_data = hand_rgb_data.to(device)

                # Attention mask
                attention_mask = torch.zeros(1, buffer_len).long().to(device)
                attention_mask[0, :buffer_len] = 1

                lang = self.val_annotations[subtask[0]][0]
                tokenized_text = self.tokenizer(lang).to(device)
                emb = self.model_clip.encode_text(tokenized_text)
                lang_embeddings = torch.stack([emb]) #B, 1, 512
                lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization
                lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)

                batch_size, sequence_length, c, h, w = rgb_data.shape

                visual_ids = self.encode_image(rgb_data) #B, l, 16
                visual_ids_hand = self.encode_image(hand_rgb_data) #B, l, 16
                # self.check_rec(visual_ids, visual_ids_hand, rgb_data, hand_rgb_data, True, ep_idx)
                if self.use_img_start:
                    img_start_ids = torch.ones(batch_size, sequence_length, 1).long().to(visual_ids.device) * 64000
                    img_end_ids = torch.ones(batch_size, sequence_length, 1).long().to(visual_ids.device) * 64001
                    visual_ids = torch.cat([img_start_ids, visual_ids, visual_ids_hand, img_end_ids], dim=2)
                else:
                    visual_ids = torch.cat([visual_ids, visual_ids_hand], dim=2)
                # import pdb;pdb.set_trace()
                if self.use_la_action:
                    # import pdb;pdb.set_trace()
                    la_indice = torch.cat(self.la_indice_list) + 64002
                    la_indice = la_indice.unsqueeze(0)
                    la_indice = torch.cat([la_indice, torch.zeros(1, 1, self.la_act_scope).long().to(visual_ids.device)], dim=1)
                    visual_ids = torch.cat([visual_ids, la_indice], dim=2)

                obs_embeddings = self.embed_func()(visual_ids) #b, l, 16, h

                lang_embeddings = lang_embeddings.view(batch_size, 1, 1, -1).repeat(1, sequence_length, 1, 1)
                lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
               
                stacked_inputs = torch.cat(
                        (lang_embeddings,
                        obs_embeddings), dim=2)  # (b, l, n_tokens, h)
            
                # Number of tokens
                n_lang_tokens = 1
                n_state_tokens = 0
                n_patch_tokens = 0
                n_obs_tokens = 17 if self.use_img_start else 16
                n_hand_patch_tokens = 0
                n_hand_obs_tokens = 17 if self.use_img_start else 16
                n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
                if self.use_hand_rgb:
                    n_tokens += n_hand_obs_tokens
                    n_tokens += n_hand_patch_tokens
                n_act_pred_tokens = 1
                if self.act_pred or self.use_la_action:
                    act_query_token_start_i = n_tokens
                    n_tokens += self.la_act_scope
                obs_tokens_start_i = n_lang_tokens + n_state_tokens + n_patch_tokens

                # Layer norm
                stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
      
                stacked_attention_mask = torch.ones_like(stacked_inputs[:, :, 0])
                
                if self.use_la_action:
                    stacked_inputs = stacked_inputs[:, :-self.la_act_scope]
                    stacked_attention_mask = stacked_attention_mask[:, :-self.la_act_scope]

                # import pdb;pdb.set_trace()
                # if self.la_act_pred:
                gen_kwargs = {'output_hidden_states': True, 'return_dict_in_generate': True}
                outputs = self.generate(stacked_inputs, stacked_attention_mask, **gen_kwargs)
                act_token_start = stacked_inputs.shape[1]
                output_ids = outputs.sequences
                output_hidden_states = torch.cat([outputs.hidden_states[i][-1] for i in range(len(outputs.hidden_states))], dim=1)
                action_embedding = output_hidden_states[:, act_token_start-1:act_token_start+self.la_act_scope]
                la_indice = output_ids[:, :self.la_act_scope] - 64002
                neg_ind = la_indice < 0
                la_indice[neg_ind] = 0
                print("la_indice:", la_indice)

                # IDM
                for i, pred_act_mlp in enumerate(self.pred_act_mlps):
                    action_embedding = pred_act_mlp(action_embedding)
                action_embedding = self.act_embed_fuse(action_embedding.permute(0, 2, 1)).squeeze(-1)
                arm_action_preds = self.pred_arm_act(action_embedding)  # (b, l, act_dim - 1)
                gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, l, 1)


                # arm_action_preds = prediction['arm_action_preds']  # (1, l, act_dim - 1)
                arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (l, act_dim - 1)
                # arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]
                # gripper_action_preds = prediction['gripper_action_preds']  # (1, l, 1)
                gripper_action_preds = gripper_action_preds.flatten()  # (l, )
                # gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]
                # Use the last action
                arm_action_pred = arm_action_preds[-1]  # (act_dim - 1, )
                arm_action_pred[:3] *= 0.02
                arm_action_pred[3:6] *= 0.05
                gripper_action_pred = gripper_action_preds[-1:]  # (1, )
                gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
                gripper_action_pred = gripper_action_pred > 0.5
                gripper_action_pred = gripper_action_pred.int().float()
                gripper_action_pred = gripper_action_pred * 2.0 - 1.0
                action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=0)  # (act_dim,)
                action_pred = action_pred.detach().cpu()

                # print(arm_action_preds, F.sigmoid(gripper_action_preds[0, 0]))
                obs, _, _, current_info = self.eval_env.step(action_pred)
                # print(obs['robot_obs'], obs['scene_obs'])
                robot_obs = torch.from_numpy(obs['robot_obs'])[None].to(robot_obs)
                cv2.imwrite(f'{subtask_vis_path}/{step+1}.jpg', obs['rgb_obs']['rgb_static'][:, :, ::-1])
                
                # import pdb;pdb.set_trace()
                current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask[0]})
                if len(current_task_info) > 0:
                    subtask_success = True
                    success_counter += 1
                    break
        
            if not subtask_success:
                break
        print(success_counter)
        record = [{'success_counter': success_counter}]
        
        return record

    def encode_image(self, img, dtype=None, device=None):
        if not isinstance(img, torch.Tensor):
            img = mmcv.imresize(img, (256, 256))
            img = torch.from_numpy(img).to(device).to(dtype)
            img = (img / 127.5) - 1.0
            img = img.permute(2, 0, 1)[None]
        flag=False
        if img.ndim == 5:
            flag = True
            b, t, _, _, _= img.shape
            img = img.flatten(0, 1)
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        # num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        # visual_ids = visual_ids + num_text_embeddings
        visual_ids = visual_ids.flatten(1)
        if flag:
            visual_ids = visual_ids.view(b, t, -1)
        return visual_ids
    def generate_image_just_forward(self, input_ids, attention_mask, size, action_idx, input_embed=None, t_num=1, **kwargs):
        _, _, hidden_state = self.neck(input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True) if not self.pre_encode_lang else self.neck(inputs_embeds=input_embed, attention_mask=attention_mask, labels=input_ids, return_dict=True)
        act_token_mask = input_ids[:, 1:] == action_idx
        act_token_mask = torch.cat([act_token_mask, torch.zeros((act_token_mask.shape[0], 1)).bool().cuda()], dim=1,)
        action_embedding = hidden_state[act_token_mask].view(1, t_num, -1)

        for pred_act_mlp in self.pred_act_mlps:
            action_embedding = pred_act_mlp(action_embedding)
        arm_action_preds = self.pred_arm_act(action_embedding)  # (b, l, act_dim - 1)
        gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, l, 1)

        return None, arm_action_preds, gripper_action_preds


    def generate_image(self, input_ids, attention_mask, size, action_idx, **kwargs):

        outputs = self.generate(input_ids, attention_mask,  **kwargs)
        output_ids = outputs.sequences
        output_hidden_states = torch.cat([outputs.hidden_states[i][-1] for i in range(len(outputs.hidden_states))], dim=1)

        act_token_mask = output_ids[:, 1:] == action_idx
        # act_token_mask = torch.cat([act_token_mask, torch.zeros((act_token_mask.shape[0], 1)).bool().cuda()], dim=1,)
        action_num = act_token_mask.sum()

        action_embeddings = output_hidden_states[act_token_mask].view(1, action_num, -1)
        for pred_act_mlp in self.pred_act_mlps:
            action_embeddings = pred_act_mlp(action_embeddings)
        arm_action_preds = self.pred_arm_act(action_embeddings)  # (b, l, act_dim - 1)
        gripper_action_preds = self.pred_gripper_act(action_embeddings)  # (b, l, 1)

        #
        text_length = input_ids.size(1)
        img_length = int(size**0.5)
        image_preds = []
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        action_num = 1
        for n in range(action_num):
            image_pred = output_ids[:, text_length + 3 + ((size + 5) * n):text_length + 3 + size + ((size + 5) * n)]
            image_pred = image_pred.reshape(-1, img_length, img_length) - num_text_embeddings
            image_pred = torch.clamp(image_pred, min=0)
            image_pred_code = self.v_token.indices_to_codes(image_pred)
            image_pred = self.pre_decode(image_pred_code)
            image_pred = self.v_token_decoder(image_pred)
            image_pred = torch.clamp((image_pred + 1) * 127, min=0, max=255)
            image_preds.append(image_pred)
        image_preds = torch.cat(image_preds, dim=0).permute(0, 2, 3, 1).to(torch.uint8)


        return image_preds, arm_action_preds, gripper_action_preds
    def embed_func(self):
        return self.neck.llm.model.embed_tokens



    def generate(self, inputs_embeds, attention_mask=None, **generate_kwargs):
        # generate_kwargs['image_embeds'] = visual_full
        generate_kwargs['bos_token_id'] = 64001
        # import pdb;pdb.set_trace()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        text_pred = self.neck.generate(
            input_ids=None,
            attention_mask=attention_mask,
            max_length=self.max_generate_length,
            max_new_tokens=self.max_new_tokens,
            inputs_embeds=inputs_embeds,
            **generate_kwargs
        )
        return text_pred

