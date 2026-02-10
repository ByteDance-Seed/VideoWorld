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
import json
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from collections import defaultdict


def gen_action_id(action_types, results):
    action_ids = []
    for i, item in enumerate(results):
        prompt = item['prompt']
        in_act = False
        for ai, action_type in enumerate(action_types):
            flag = [act in prompt for act in action_type]
            if True in flag:
                in_act = True
                action_ids.append(ai)
                break
        if not in_act:
            print(f"not in act: {prompt}")
            action_ids.append(len(action_types))
    return action_ids

def generate_distinct_colors(n):
    hues = [(x*360/n) for x in range(n)]
    saturation = 0.9  # 高饱和度
    value = 0.8       # 适中的明度

    colors = []
    for hue in hues:
        colors.append(mcolors.hsv_to_rgb((hue/360.0, saturation, value)))
    
    return colors

def draw_tsne(t_sne_features, label_color, dir, tag):
    plt.figure()
    plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1],  color=label_color, cmap='jet')
    plt.savefig(f'{dir}/{tag}.jpg')
    plt.close()

gt_pos_list = []
encode_feat_list = []
results = torch.load('/opt/tiger/mmagicinit/la_test_calvin_results.pth')

import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
metrics = {}
bin_length = 0.1

rel_x_list = []
rel_y_list = []
rel_z_list = []

orn_x_list = []
orn_y_list = []
orn_z_list = []

gripper_list = []

encode_feat_list = defaultdict(list)
# A2L_dict_list = [defaultdict(int) for _ in range(gt_act_num)]
# A2LNums = defaultdict(int)
rel_x_list = defaultdict(list)
rel_y_list = defaultdict(list)
rel_z_list = defaultdict(list)
orn_x_list = defaultdict(list)
orn_y_list = defaultdict(list)
orn_z_list = defaultdict(list)
gripper_list = defaultdict(list)

perplexity = 50

# action_types = [['rotate'], ['push','slide the block', 'push the sliding door'], ['sweep', 'pull the handle'], ['pick up', 'grasp', 'lift', 'Take', 'take', 'store', 'press']]
# action_types = [['rotate'], ['push'], ['slide', 'sweep', 'turn the'], ['pull'], ['lift', 'Take', 'take', 'pick', 'grasp'], ['store', 'stack'], ['press', 'switch']]
action_types = [['lift']]
start_color = 'green'
end_color = 'red'
middle_color = 'yellow'
# 创建颜色映射对象
cmap = LinearSegmentedColormap.from_list("gradient", [start_color, middle_color, end_color], N=20)
colors = cmap(np.linspace(0, 1, 20))

act_colors = generate_distinct_colors(len(action_types)+1)
fig, ax = plt.subplots(figsize=(6, 2))
for i, color in enumerate(act_colors):
    rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='none', facecolor=color)
    ax.add_patch(rect)
ax.set_xlim(0, len(colors))
ax.set_ylim(0, 1)
ax.axis('off')
plt.savefig('/opt/tiger/mmagicinit/color_blocks.png', dpi=300)

action_ids = gen_action_id(action_types, results)
for i, item in enumerate(results):
    
    encode_feat = item.pop('encode_feat').permute(1, 0)
    

    for fi, (rel_pos, rel_orn, gripper) in enumerate(zip(item['rel_pos'], item['rel_orn'], item['gripper'])):
        rel_pos_x, rel_pos_y, rel_pos_z = rel_pos
        rel_orn_x, rel_orn_y, rel_orn_z = rel_orn
        rel_x_list[fi].append(int(((rel_pos_x + 1) // bin_length).item()))
        rel_y_list[fi].append(int(((rel_pos_y + 1) // bin_length).item()))
        rel_z_list[fi].append(int(((rel_pos_z + 1) // bin_length).item()))
        gripper_list[fi].append(gripper)

        orn_x_list[fi].append(int(((rel_orn_x + 1) // bin_length).item()))
        orn_y_list[fi].append(int(((rel_orn_y + 1) // bin_length).item()))
        orn_z_list[fi].append(int(((rel_orn_z + 1) // bin_length).item()))

    
        encode_feat_list[fi].append(encode_feat[fi])

x_label_color = {}
y_label_color = {}
z_label_color = {}
orn_x_label_color = {}
orn_y_label_color = {}
orn_z_label_color = {}
gripper_color = {}
act_label_colors = np.array([act_colors[i] for i in action_ids])
for fi in range(len(rel_x_list)):
    x_label_color[fi] = np.array([colors[i] for i in rel_x_list[fi]])
    y_label_color[fi] = np.array([colors[i] for i in rel_y_list[fi]])
    z_label_color[fi] = np.array([colors[i] for i in rel_z_list[fi]])

    orn_x_label_color[fi] = np.array([colors[i] for i in orn_x_list[fi]])
    orn_y_label_color[fi] = np.array([colors[i] for i in orn_y_list[fi]])
    orn_z_label_color[fi] = np.array([colors[i] for i in orn_z_list[fi]])

    gripper_color[fi] = np.array([colors[0] if i==0 else colors[-1] for i in gripper_list[fi]])
    


    encode_feats = torch.stack(encode_feat_list[fi]).numpy() #N, 512
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity).fit_transform(encode_feats)
    draw_tsne(t_sne_features, x_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_x_f{fi+1}_{perplexity}')
    draw_tsne(t_sne_features, y_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_y_f{fi+1}_{perplexity}')
    draw_tsne(t_sne_features, z_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_z_f{fi+1}_{perplexity}')

    draw_tsne(t_sne_features, orn_x_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_orn_x_f{fi+1}_{perplexity}')
    draw_tsne(t_sne_features, orn_y_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_orn_y_f{fi+1}_{perplexity}')
    draw_tsne(t_sne_features, orn_z_label_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'tsne_orn_z_f{fi+1}_{perplexity}')

    draw_tsne(t_sne_features, gripper_color[fi], dir='/opt/tiger/mmagicinit/', tag=f'gripper_f{fi+1}_{perplexity}')
    draw_tsne(t_sne_features, act_label_colors, dir='/opt/tiger/mmagicinit/', tag=f'act_f{fi+1}_{perplexity}')