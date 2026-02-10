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
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.transforms import Bbox
# from sgfmill import sgf
import tarfile
import json
from typing import List, Any, Optional
import multiprocessing as mp
import pickle
import numpy as np
import torch
import re
from tqdm import tqdm

def draw_board(info, next_pos):
    
    moves = info['moves']
    # extra_moves = info['extra_moves']
    extra_moves = None 
    ignore_moves = [1] * len(info['moves'])
    captured = info['captured'][:len(info['moves'])]
    extra_captured = None 
    if extra_moves is not None:
        moves = moves[:len(extra_moves)]
    else:
        # return 
        extra_moves = [None] * len(moves)
        extra_captured = [None] * len(moves)
    if sum(ignore_moves) == 0:
        return
    idx = 0
    _idx = "%09d" % idx
    # import pdb;pdb.set_trace()
    save_path = '/opt/tiger/mmagicinit/test_la/{}/'.format(_idx)
    os.system('mkdir -p {}'.format(save_path))
    max_num = 361
    # try:
    #O: white, X: black
    board_size = 9
    si = 0

    fig, ax = plt.subplots(figsize=(2.56, 2.56), facecolor='orange')
    ax.set_facecolor('orange')
    # Draw the grid
    
    
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
    # import pdb;pdb.set_trace()
    color_dict = {'b': 'black', 'w': 'white'}
    record_dot = {}
    role = ['b', 'w']
    for mi, (move, ignore_move, extra_move, cap, e_cap) in enumerate(zip(moves, ignore_moves, extra_moves, captured, extra_captured)):
        
        if mi > max_num:
            break
        if move[0] is None:
            continue
        if extra_move is not None and ignore_move == 1:
            if e_cap is not None:
                # import pdb;pdb.set_trace()
                for cap_pos in e_cap:
                    record_dot[str(cap_pos[0])+str(cap_pos[1])][0].remove()
            extra_postion, extra_color = extra_move
            assert extra_color == role[mi % 2]
            dot = ax.scatter(*extra_postion[::-1], s=380, c=color_dict[extra_color], zorder=2, antialiased=False)
            plt.savefig('{}/extra_{}.png'.format(save_path, mi), dpi=100, bbox_inches='tight', pad_inches=0)
            dot.remove()
            if e_cap is not None:
                for cap_pos in e_cap:
                    dot = ax.scatter(cap_pos[1], cap_pos[0], s=380, c=color_dict[record_dot[str(cap_pos[0])+str(cap_pos[1])][1]], zorder=2, antialiased=False)
                    record_dot[str(cap_pos[0])+str(cap_pos[1])][0] = dot


        position, color = move
        dot = ax.scatter(*position[::-1], s=380, c=color_dict[color], zorder=2, antialiased=False)
        record_dot[str(position[0])+str(position[1])] = [dot, color]
        plt.axis('tight')
        if cap is not None:
            # import pdb;pdb.set_trace()
            for cap_pos in cap:
                record_dot[str(cap_pos[0])+str(cap_pos[1])][0].remove()
        if mi == len(moves) - 1:
            if ignore_move == 1 or (ignore_move == 0 and ignore_moves[mi+1] == 1):
                plt.savefig('{}/{}.png'.format(save_path, mi), dpi=100, bbox_inches='tight', pad_inches=0)
            next_mi = mi + 1
    import pdb;pdb.set_trace()
    cap_pos = None
    for mi, move in enumerate(next_pos):
        position, color = move
        if cap_pos is not None:
            record_dot[str(cap_pos[0])+str(cap_pos[1])][0].remove()
        dot = ax.scatter(*position[::-1], s=380, c=color_dict[color], zorder=2, antialiased=False)
        record_dot[str(position[0])+str(position[1])] = [dot, color]
        plt.axis('tight')
        cap_pos = [position[0], position[1]]
        
        plt.savefig('{}/{}_{}.png'.format(save_path, next_mi, mi), dpi=100, bbox_inches='tight', pad_inches=0)


# with open('/opt/tiger/mmagicinit/ldm/data/go_dataset_size9/output_0_cap.json', 'r') as f:
#     infos = json.load(f)
next_pos = [[[2, 3], 'b'], [[3, 1], 'b'], [[4, 1], 'b'], [[5, 1], 'b']]
info = {'moves': [[[5, 4], 'b'], [[3, 4], 'w'], [[5, 6], 'b'], [[4, 2], 'w'], [[6, 2], 'b'], [[4, 4], 'w'], [[1, 4], 'b'], [[5, 3], 'w'], [[6, 4], 'b'], [[2, 6], 'w'], [[2, 2], 'b'], [[6, 3], 'w'], [[7, 3], 'b'], [[4, 6], 'w'], [[6, 6], 'b'], [[1, 5], 'w'], [[2, 4], 'b'], [[5, 7], 'w'], [[4, 7], 'b'], [[2, 5], 'w'], [[4, 5], 'b'], [[3, 6], 'w'], [[2, 1], 'b'], [[0, 4], 'w']], 'captured': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [[0, 3]], None, None, None, None, None, None, None, None, None, [[4, 7]], None, None, None, [[0, 1]], None, None, None, None, [[7, 8]], None, [[7, 2]], None, [[1, 1], [0, 0]], None, None, None, [[5, 5]], [[6, 8]], None, None, None, None, [[7, 4]], None, None, None, None]}
draw_board(info, next_pos)

import pdb;pdb.set_trace()