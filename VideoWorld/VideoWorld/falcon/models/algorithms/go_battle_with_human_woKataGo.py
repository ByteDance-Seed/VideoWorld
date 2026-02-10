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
import torch.nn.functional as F
import os
from falcon.models.vbackbones.dvae_utils import ImageTokenizer, ImageTokenizerDecoder
from falcon.models.quantizer.vqgan import VQGANDecoder, VQGANEncoder
from falcon.models.quantizer.connector import ConvConnector
from falcon.registry import MODELS
from .base import BaseModel
from ..utils import KataGo_Ana, Katrain_bot
import numpy as np
import cv2
import sgfmill
import mmcv
import copy
IMAGE_TOKEN_INDEX = -200
MASK_TOKEN_INDEX = -300
import random
import matplotlib.pyplot as plt
import subprocess
import imageio
import json
def send_command_and_get_response(katago_process, command):
    katago_process.stdin.write(command + '\n')  # 发送命令
    katago_process.stdin.flush()  # 确保命令被发送到子进程
    response = ''
    while True:
        line = katago_process.stdout.readline().strip()
        response += line + '\n'
        if line == '':
            break  # 等待直到读取到空行，表示响应结束
    return response

def locate_new_piece_using_morphology(img0, img1, corner_x=12, corner_y=12, cell_size=29):

    # 计算两幅图像的差异
    # import pdb;pdb.set_trace()
    diff = cv2.absdiff(img1, img0)
    # 转换为灰度图

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray[gray<90] = 0
    gray[gray>220] = 0
    # 应用阈值
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 形态学变换，如膨胀操作
    kernel = np.ones((2,2),np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erode, kernel, iterations=2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    grid_x = round((cx - corner_x) / cell_size)
    grid_y = round((cy - corner_y) / cell_size)

    return grid_x, grid_y

def render_next_state(moves, caps, save_path=None):
    name = str(int((random.random() + random.random() + random.random() + random.random()) * 1000000)) + '.png'
    save_path = './visualize_temp/{}'.format(name) if save_path is None else save_path
    board_size = 9
    fig, ax = plt.subplots(figsize=(2.56, 2.56), facecolor='orange')
    ax.set_facecolor('orange')
    # Draw the grid

    # import pdb;pdb.set_trace()
    # Draw stones

    for i in range(board_size):
        ax.plot([i, i], [0, board_size-1], color='k', zorder=1, linewidth=1, antialiased=True)
        ax.plot([0, board_size-1], [i, i], color='k', zorder=1, linewidth=1, antialiased=True)
    ax.set_aspect('equal', adjustable='box')
    # bbox = Bbox.from_bounds(0, 0, 128, 128)
    ax.axis('off')
    ax.set_facecolor('orange')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  
    record_dot = {}
    color_dict = {'b': 'black', 'w': 'white'}
    for mi, (move, cap) in enumerate(zip(moves, caps)):
        if move[0] is None:
            continue
        color, position = move
        position = list(position)
        position[0] = 8 - position[0]
        dot = ax.scatter(*position[::-1], s=380, c=color_dict[color], zorder=2, antialiased=False)
        record_dot[str(position[0])+str(position[1])] = [dot, color]
        if cap is not None:
            for cap_pos in cap:
                cap_pos = list(cap_pos)
                cap_pos[0] = 8 - cap_pos[0]
                record_dot[str(cap_pos[0])+str(cap_pos[1])][0].remove()
        plt.axis('tight')
        # plt.savefig('/opt/tiger/tic_test/{}.png'.format(mi), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)

    img = mmcv.imread(save_path, channel_order='rgb', backend='pillow')
    img = mmcv.imresize(
        img,
        (256, 256),
        interpolation='bilinear',
        backend='pillow')
    img = (img / 127) - 1
    img = torch.tensor(img.transpose(2, 0, 1))
    img = torch.clamp(img, min=-1, max=1)
    return img


@MODELS.register_module()
class VideoWorldGoBattleVSHumanwoKataGo(BaseModel):

    def __init__(self, vbackbone, neck, head, quantizer, init_cfg=None, pred_image=True, pred_action=False, mode='acc', battle_with_katago=None, kata_ana=False, vq_decoder_cfg={}, max_generate_length=300, work_dir='/opt/tiger', worker_index=0):
        super().__init__(
            vbackbone=vbackbone, neck=neck, head=head, init_cfg=init_cfg)

   
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
    

        state_dict = torch.load(vbackbone['init_cfg']['checkpoint'])['state_dict']
        encoder_state_dict = {}
        decoder_state_dict = {}
        post_encode_state_dict = {}
        pre_decode_state_dict = {}
        quantizer_state_dict = {}
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

        self.mode = mode
        self.board_size = 19
        self.sub_board_size = 9
        self.kata_ana = kata_ana
        self.katago_process = None
        

        self.worker_index = f'{torch.cuda.current_device()}_{worker_index}'
        os.system(f'mkdir ./visualize/')
        os.system(f'mkdir ./visualize_temp/')
        
    def check_rec(self, visual_ids, visual_mask_ids, img):
        # import pdb;pdb.set_trace()
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        size = int(visual_ids.shape[-1] **0.5)
        rec_visual_mask_ids = visual_mask_ids.reshape(-1, size, size) - num_text_embeddings
        rec_visual_mask_ids = self.v_token.indices_to_codes(rec_visual_mask_ids)
        rec_visual_mask_ids = self.pre_decode(rec_visual_mask_ids)
        rec_visual_mask_ids = self.v_token_decoder(rec_visual_mask_ids)
        rec_visual_mask = torch.clamp((rec_visual_mask_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)

        rec_visual_ids = visual_ids.reshape(-1, size, size) - num_text_embeddings
        rec_visual_ids = self.v_token.indices_to_codes(rec_visual_ids)
        rec_visual_ids = self.pre_decode(rec_visual_ids)
        rec_visual_ids = self.v_token_decoder(rec_visual_ids)
        rec_visual_ids = torch.clamp((rec_visual_ids + 1) * 127, min=0, max=255).permute(0, 2, 3, 1)
        for idx, (rec_visual_id, gt_img) in enumerate(zip(rec_visual_ids, img)):
            import cv2
            import numpy as np
            rec_visual_id = rec_visual_id.cpu().numpy()[:, :, ::-1].astype(np.uint8)
            gt_img = gt_img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8)
            show = np.concatenate((gt_img, rec_visual_id), axis=0)
            cv2.imwrite(f'/opt/tiger/rec_visualize/test_{idx}.jpg', show)

    def forward_train(self, img, input_ids, pred_label=None, attention_mask=None, **kwargs):
       
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
            pred_label = torch.stack(pred_label)
            img = torch.stack(img)
            attention_mask = torch.stack(attention_mask)
            invalid = torch.stack(kwargs.get('invalid'))
        
        b, c, h, w = img.shape
        
        visual_ids = self.encode_image(img)

        if pred_label is not None:
            visual_mask_ids = self.encode_image(pred_label)
            visual_mask_ids = visual_mask_ids.detach()
        else:
            visual_mask_ids = None
        input_ids, attention_mask, labels = self.prepare_input(visual_ids, input_ids, attention_mask, visual_mask_ids)

        # self.check_rec(visual_ids, visual_mask_ids, img)
        visual_ids = visual_ids.detach()

        attn_length = [torch.where(attn_m != 0)[0].max() for attn_m in attention_mask]
        max_attn = max(attn_length)
        input_ids = input_ids[:, :(max_attn+2)]
        attention_mask = attention_mask[:, :(max_attn+2)]
        labels = labels[:, :(max_attn+2)]

        for i, label in enumerate(labels):
            if attn_length[i] + 1 >= 1024:
                continue
            label[(attn_length[i] + 2):] = -100
            attention_mask[i, attn_length[i] + 1] = 1
            if invalid[i] == 1:
                label = -100
        
        logits, loss, _ = self.neck(input_ids, attention_mask=attention_mask, labels=labels)
        # print("--------", input_ids[0], input_ids[1], "--------")
        losses = {'losses_v': loss}

        return losses


    def forward_test(self, img, input_ids, pred_label=None, attention_mask=None, index=None, **kwargs):
        return self.image_gen_battle(img, input_ids, pred_label, attention_mask, index, **kwargs)
       

    def send_ai_command(self, command):
        if self.katago_process:
            res = send_command_and_get_response(self.katago_process, command).strip().replace(' ', '').replace('=', '')
        else:
            if command =='final_score':
                res = self.katrain_process.game.current_node.format_score()
            else:
                res = self.katrain_process.query(command)
        return res

    def image_gen_battle(self, img, input_ids, pred_label=None, attention_mask=None, index=0, **kwargs):
        print("\n -------------------------- \nLet's start the game! You will play as White, and VideoWorld will play as Black. VideoWorld will generate a new board state based on your move and store it at the path './visualize'. \n -------------------------- \n")
        kwargs['eos_token_id'] = 50256
        cur_iter = len(os.listdir(f'./visualize/'))
        kwargs.pop('data_mode')
        kwargs.pop('level')
        kwargs.pop('katrain_level')
        
        eval_dict = {
            "prior_ratio_kata_to_katago": [],
            "prior_ratio_pred_to_katago": [],
            "prior_ratio_kata_to_katago_list": [],
            "prior_ratio_pred_to_katago_list": [],
            "winrate_list": [],
            'results': -1,
            'score': 0,
            "katrain_level": '5d',
            "data_mode": "go_battle",
            "katago_ana_results": [],
            "error_move_num": 0,
            "error_color_num": 0,
            "score_list": []
        }

        max_retry = 10
        moves = []
        board = sgfmill.boards.Board(self.sub_board_size)
        displayboard = board.copy()
        step_num = 0
        caps = []
        prior_ratio_pred_to_katago = []
        prior_ratio_kata_to_katago = []
        score_list = []
        ana_res = None
        manual_open = True
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
            img = torch.stack(img)
            attention_mask = torch.stack(attention_mask)
        while True:
            # generte next move image
            visual_ids = self.encode_image(img)
            _input_ids, _attention_mask, _ = self.prepare_input(visual_ids, input_ids, attention_mask)
            attn_length = [torch.where(attn_m != 0)[0].max() for attn_m in _attention_mask]
            max_attn = max(attn_length)
            _input_ids = _input_ids[:, :(max_attn+1)]
            _attention_mask = _attention_mask[:, :(max_attn+1)]
            try_num = 1
            while(try_num <= max_retry):
                outputs, action_pred = self.generate_image(_input_ids, _attention_mask, visual_ids.shape[-1], **kwargs)
                outputs = outputs[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                _img = torch.clamp((img[0] + 1) * 127, min=0, max=255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                try:
                    if step_num == 0:
                        pos = 'D6'
                        pred_x, pred_y = 5, 3
                    else:
                        pred_y, pred_x = locate_new_piece_using_morphology(_img, outputs) #this pred_x is actually 8 - row
                        pred_x = 8 - pred_x
                        num_pos = pred_x * 9 + pred_y
                        col = chr(pred_y + 65) if pred_y < 8 else chr(pred_y + 66)
                        row = str(pred_x + 1)
                        pos = col + row
                    try:
                        _, captured = displayboard.play(pred_x, pred_y, 'b')
                    except:
                        print("Error, retry num {}".format(try_num))
                        try_num += 1
                        continue
                    print("B_move:", pos)
                    moves.append(('b', (pred_x, pred_y)))
                    # _, captured = displayboard.play(pred_x, pred_y, 'b')
                    caps.append(captured)

                    #Kata Ana
                    if ana_res is not None:
                        pred_row_major_pos = (self.sub_board_size - pred_x - 1) * self.sub_board_size + pred_y
                        step_prior = ana_res['policy']
                        max_prior = max(step_prior)
                        pred_prior = step_prior[pred_row_major_pos]
                        prior_ratio_pred_to_katago.append(pred_prior / max_prior)
                    break
                except:
                    print("Error, retry num {}".format(try_num))
                    try_num += 1

            os.system(f' mkdir -p ./visualize/{cur_iter}/{torch.cuda.current_device()}')
            cv2.imwrite(f'./visualize/{cur_iter}/{torch.cuda.current_device()}/{step_num}.jpg', outputs[:,:,::-1])
          
            step_num += 1
            if try_num > max_retry:
                break

            try:
                quite = False
                while True:
                    kata_pos = input('Please enter your move (e.g. D5, E6) or quite (q):')
                    if kata_pos == 'q':
                        quite = True
                        break
                    try:
                        col = kata_pos[0]
                        row = int(kata_pos[1:]) - 1
                        col = ord(col) - 65 if col != 'J' else ord(col) - 66
                        _, captured = displayboard.play(row, col, 'w')
                        break
                    except:
                        print("Error, please enter again")
                        continue
                if quite:
                    break
                moves.append(('w', (row, col)))
                caps.append(captured)
                img = render_next_state(moves, caps).to(img)[None]

            except:
                break
       
        b_count = sum([displayboard.board[i].count('b') for i in range(9)])
        w_count = sum([displayboard.board[i].count('w') for i in range(9)])
        if w_count > b_count:
            eval_dict['results'] = 0
            print('Currently, the write (You) are winning')
        elif w_count < b_count:
            eval_dict['results'] = 1
            print('Currently, the black (VideoWorld) are winning')
        else:
            print('Draw')
        if eval_dict['results'] != -1:
            eval_dict['score'] = 0
       
        eval_dict['moves'] = moves
        record = {'eval_dict': eval_dict}
        
        with imageio.get_writer(uri=f'./visualize/{cur_iter}/{torch.cuda.current_device()}/battle.gif', mode='I', fps=1) as writer:
            target_file = f'./visualize/{cur_iter}/{torch.cuda.current_device()}'
            for i in range(len(os.listdir(target_file))):
                writer.append_data(imageio.imread(f'{target_file}/{i}.jpg'))
        
        return [record]
    
    def encode_image(self, img):
        visual_full = self.vbackbone(img)
        visual_full = self.post_encode(visual_full)
        _, visual_ids = self.v_token(visual_full)
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        visual_ids = visual_ids + num_text_embeddings
        visual_ids = visual_ids.flatten(1)

        return visual_ids

    def generate_image(self, input_ids, attention_mask, size, **kwargs):
        kwargs['max_new_tokens'] = 25
        outputs = self.generate(input_ids, attention_mask, **kwargs)
        text_length = input_ids.size(1)
        if self.pred_action:
            action_pred = outputs[:, text_length:text_length + 3]
            image_pred = outputs[:, text_length + 5:text_length + 5 + size]
            action_pred = action_pred - 32004
            pred_x, pred_y = action_pred % 9, action_pred // 9
        else:
            action_pred = None
            image_pred = outputs[:, text_length:text_length + size]
        image_pred = torch.cat([image_pred, torch.ones([len(image_pred), size-image_pred.shape[1]]).to(image_pred) * 50260], dim=1) if image_pred.shape[1] < size else image_pred
        num_text_embeddings = self.neck.llm.get_input_embeddings().num_embeddings - self.neck.vq_num
        img_length = int(size**0.5)
        image_pred = image_pred.reshape(-1, img_length, img_length) - num_text_embeddings
        image_pred = torch.clamp(image_pred, min=0)
        image_pred_code = self.v_token.indices_to_codes(image_pred)
        image_pred = self.pre_decode(image_pred_code)
        outputs = self.v_token_decoder(image_pred)
        outputs = torch.clamp((outputs + 1) * 127, min=0, max=255)

        return outputs, action_pred

    def prepare_input(self, visual_ids, input_ids, attention_mask, visual_mask_ids=None):
        new_input_ids = []
        new_attention_masks = []
        new_labels = []
        visual_mask_ids = torch.zeros((len(visual_ids), 0)).to(visual_ids) if visual_mask_ids is None else visual_mask_ids
        for bz, (visual_id, visual_mask_id, input_id) in enumerate(zip(visual_ids, visual_mask_ids, input_ids)):
            if visual_mask_id.ndim != 2:
                visual_mask_id = visual_mask_id[None]
            if visual_id.ndim != 2:
                visual_id = visual_id[None]
            cur_new_input_ids = []
            cur_new_attention_masks = []
            cur_new_label = []
            bz_attention_mask = attention_mask[bz]
            image_token_indices = torch.where((input_id == IMAGE_TOKEN_INDEX) | (input_id == MASK_TOKEN_INDEX))[0]
            i_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
            m_token_indices = torch.where(input_id == MASK_TOKEN_INDEX)[0]

            cur_image_idx = 0
            cur_mask_idx = 0
            while image_token_indices.numel() > 0:
                image_token_start = image_token_indices[0]
                if image_token_start in i_token_indices:
                    cur_visual_id = visual_id[cur_image_idx]
                    cur_image_idx += 1
                else:
                    cur_visual_id = visual_mask_id[cur_mask_idx]
                    cur_mask_idx += 1

                cur_attention_mask = torch.ones_like(cur_visual_id)
                cur_label = torch.ones_like(cur_visual_id) * -100


                cur_new_input_ids.append(input_id[:image_token_start])
                cur_new_input_ids.append(cur_visual_id)
                cur_new_input_ids.append(input_id[image_token_start + 1 : image_token_start + 2])


                cur_new_attention_masks.append(bz_attention_mask[:image_token_start])
                cur_new_attention_masks.append(cur_attention_mask)
                cur_new_attention_masks.append(bz_attention_mask[image_token_start + 1 : image_token_start + 2])

                if image_token_start in m_token_indices:
                    # cur_new_label.append(torch.ones_like(bz_attention_mask[:image_token_start - 1]))
                    cur_new_label.append(input_id[:image_token_start - 1])
                    cur_new_label.append(input_id[image_token_start - 1:image_token_start])
                    cur_new_label.append(cur_visual_id)
                    cur_new_label.append(input_id[image_token_start + 1 : image_token_start + 2])
                else:
                    cur_new_label.append(torch.ones_like(bz_attention_mask[:image_token_start]) * -100)
                    cur_new_label.append(cur_label)
                    cur_new_label.append(torch.ones_like(bz_attention_mask[image_token_start + 1 : image_token_start + 2]) * -100)

                input_id = input_id[image_token_start + 2 :]
                bz_attention_mask = bz_attention_mask[image_token_start + 2 :]
                image_token_indices = torch.where((input_id == IMAGE_TOKEN_INDEX) | (input_id == MASK_TOKEN_INDEX))[0]
                i_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
                m_token_indices = torch.where(input_id == MASK_TOKEN_INDEX)[0]

            if input_id.numel() > 0:
                cur_new_input_ids.append(input_id)
                cur_new_attention_masks.append(bz_attention_mask)
                cur_new_label.append(input_id)
           

            cur_new_input_ids = torch.cat(cur_new_input_ids)
            cur_new_attention_masks = torch.cat(cur_new_attention_masks)
            cur_new_label = torch.cat(cur_new_label)

            new_labels.append(cur_new_label)
            new_input_ids.append(cur_new_input_ids)
            new_attention_masks.append(cur_new_attention_masks)


        input_ids = torch.stack(new_input_ids)
        attention_mask = torch.stack(new_attention_masks)
        visual_ids = visual_ids.detach()
        labels = torch.stack(new_labels)

        return input_ids, attention_mask, labels

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
       
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        text_pred = self.neck.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_generate_length,
            **generate_kwargs
        )
        return text_pred



