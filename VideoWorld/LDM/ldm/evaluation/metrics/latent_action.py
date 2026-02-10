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

from typing import List, Optional, Sequence, Tuple
from mmengine.evaluator import BaseMetric
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mmengine.dist import is_main_process
from scipy import linalg
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from ldm.registry import METRICS
from ..functional import (disable_gpu_fuser_on_pt19, load_inception,
                          prepare_inception_feat)
from .base_gen_metric import GenerativeMetric
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

def generate_distinct_colors(n):
    hues = [(x*360/n) for x in range(n)]
    saturation = 0.9  # 高饱和度
    value = 0.8       # 适中的明度

    colors = []
    for hue in hues:
        colors.append(mcolors.hsv_to_rgb((hue/360.0, saturation, value)))
    
    return colors


def locate_new_piece_using_morphology(img0, img1, corner_x=6, corner_y=6, cell_size=14):
        # cv2.imwrite('./work_dirstest1.jpg', img0)
        # cv2.imwrite('./work_dirstest2.jpg', img1)
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
        cv2.imwrite('/opt/tiger/thresh.jpg', thresh)
        # 形态学变换，如膨胀操作
        kernel = np.ones((2,2),np.uint8)
        erode = cv2.erode(thresh, kernel, iterations=1)
        dilation = cv2.dilate(erode, kernel, iterations=2)
        cv2.imwrite('/opt/tiger/dilation.jpg', dilation)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # try:
        max_contour = max(contours, key=cv2.contourArea)
        
            
        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        grid_x = round((cx - corner_x) / cell_size)
        grid_y = round((cy - corner_y) / cell_size)

        return grid_x, grid_y

@METRICS.register_module()
class LAMetric(BaseMetric):
    def __init__(self,
                 la_num,
                 gt_act_num,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 draw_dir=None,):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.la_num = la_num
        self.gt_act_num = gt_act_num
        self.sample_model = 'ema'


    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        return None

    

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # import pdb;pdb.set_trace()
        for data_sample in data_samples:
            indice = data_sample['indice'].squeeze().item()
            img0 = data_sample['gt_img'][:, 0].permute(1, 2, 0).cpu().numpy()
            img1 = data_sample['gt_img'][:, 1].permute(1, 2, 0).cpu().numpy()
            try:
                pred_y, pred_x = locate_new_piece_using_morphology(img0, img1) 
            except:
                continue
            pred_x = 8 - pred_x
            pos = pred_x * 9 + pred_y
            data_sample['gt_action'] = pos
            data_sample['la_action'] = indice
            data_sample.pop('gt_img')
            data_sample.pop('fake_img')
            data_sample.pop('indice')
            self.results.append(data_sample)

    def compute_metrics(self, results: list):
        print("**********begin to compute the metrics****************")
        # import pdb;pdb.set_trace()
        metrics = {}
        A2L_dict_list = [defaultdict(int) for _ in range(self.gt_act_num)]
        A2LNums = defaultdict(int)
        for item in results:
            la_action = item['la_action']
            gt_action = item['gt_action']
            A2L_dict_list[gt_action][la_action] += 1
            A2LNums[gt_action] += 1
        for ai, A2L_dict in enumerate(A2L_dict_list):
            for la in A2L_dict:
                A2L_dict[la] = A2L_dict[la] / A2LNums[ai]
        with open('./work_dirsla_test.json', 'w') as f:
            json.dump(A2L_dict_list, f)
        return metrics



@METRICS.register_module()
class LAFeatMetric(BaseMetric):
    def __init__(self,
                 la_num,
                 gt_act_num,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 draw_dir=None,):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.la_num = la_num
        self.gt_act_num = gt_act_num
        self.sample_model = 'ema'


    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        return None

    
    def angle_between_angles(self, a, b):
        diff = b - a
        return (diff + torch.pi) % (2 * torch.pi) - torch.pi

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        import pdb;pdb.set_trace()
        max_pos = 0.02
        max_orn = 0.05
        for i, data_sample in enumerate(data_samples):
            encode_feat = data_sample['encode_feat'].squeeze() #C
            state0 = data_batch['data_samples'][i].states[0, 0]
            state1 = data_batch['data_samples'][i].states[0, 1]
            delta = state1 - state0
            rel_pos = torch.clip(delta[:3], -max_pos, max_pos) / max_pos
            # rel_pos_x, rel_pos_y, rel_pos_z = rel_pos
            rel_orn = self.angle_between_angles(state0[3:6], state1[3:6])
            rel_orn = torch.clip(rel_orn, -max_orn, max_orn) / max_orn
            gripper = state1[-1:]


            # indice = data_sample['indice'].squeeze().item()
            # img0 = data_sample['gt_img'][:, 0].permute(1, 2, 0).cpu().numpy()
            # img1 = data_sample['gt_img'][:, 1].permute(1, 2, 0).cpu().numpy()
        
           
            data_sample['rel_pos'] = rel_pos
            data_sample['rel_orn'] = rel_orn
            data_sample['gripper'] = gripper if gripper == 1 else 0
            # data_sample['la_action'] = indice
            data_sample['encode_feat'] = encode_feat
            # data_sample['delta_pos']
            data_sample.pop('gt_img')
            data_sample.pop('fake_img')
            data_sample.pop('indice')
            self.results.append(data_sample)
    def draw_tsne(self, t_sne_features, label_color, dir, tag):
        plt.figure()
        plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1],  color=label_color, cmap='jet')
        plt.savefig(f'{dir}/{tag}.jpg')
        plt.close()

    def compute_metrics(self, results: list):
        print("**********begin to compute the metrics****************")
        import pdb;pdb.set_trace()
        
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        metrics = {}
        bin_length = 0.1
        # rel_x_dict = defaultdict(list)
        # rel_y_dict = defaultdict(list)
        # rel_z_dict = defaultdict(list)
        rel_x_list = []
        rel_y_list = []
        rel_z_list = []

        orn_x_list = []
        orn_y_list = []
        orn_z_list = []

        gripper_list = []

        encode_feat_list = []
        # A2L_dict_list = [defaultdict(int) for _ in range(self.gt_act_num)]
        # A2LNums = defaultdict(int)
        for i, item in enumerate(results):
            rel_pos_x, rel_pos_y, rel_pos_z = item['rel_pos']
            rel_orn_x, rel_orn_y, rel_orn_z = item['rel_orn']
            gripper = item['gripper']
            rel_x_list.append(int(((rel_pos_x + 1) // bin_length).item()))
            rel_y_list.append(int(((rel_pos_y + 1) // bin_length).item()))
            rel_z_list.append(int(((rel_pos_z + 1) // bin_length).item()))

            orn_x_list.append(int(((rel_orn_x + 1) // bin_length).item()))
            orn_y_list.append(int(((rel_orn_y + 1) // bin_length).item()))
            orn_z_list.append(int(((rel_orn_z + 1) // bin_length).item()))

            gripper_list.append(gripper)
            encode_feat_list.append(item.pop('encode_feat'))
            # rel_x_dict[rel_pos_x + 1 // bin_length].append(item['encode_feat'])
            # rel_y_dict[rel_pos_x + 1 // bin_length].append(item['encode_feat'])
            # rel_z_dict[rel_pos_x + 1 // bin_length].append(item['encode_feat'])

        start_color = 'green'
        end_color = 'red'
        middle_color = 'yellow'
        # 创建颜色映射对象
        cmap = LinearSegmentedColormap.from_list("gradient", [start_color, middle_color, end_color], N=20)
        colors = cmap(np.linspace(0, 1, 20))
        x_label_color = np.array([colors[i] for i in rel_x_list])
        y_label_color = np.array([colors[i] for i in rel_y_list])
        z_label_color = np.array([colors[i] for i in rel_z_list])

        orn_x_label_color = np.array([colors[i] for i in orn_x_list])
        orn_y_label_color = np.array([colors[i] for i in orn_y_list])
        orn_z_label_color = np.array([colors[i] for i in orn_z_list])

        gripper_color = np.array([colors[0] if i==0 else colors[-1] for i in gripper_list ])
        
        encode_feat_list = torch.stack(encode_feat_list).numpy() #N, 512
        t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(encode_feat_list)
        self.draw_tsne(t_sne_features, x_label_color, dir='./work_dirs', tag='tsne_x')
        self.draw_tsne(t_sne_features, y_label_color, dir='./work_dirs', tag='tsne_y')
        self.draw_tsne(t_sne_features, z_label_color, dir='./work_dirs', tag='tsne_z')

        self.draw_tsne(t_sne_features, orn_x_label_color, dir='./work_dirs', tag='tsne_orn_x')
        self.draw_tsne(t_sne_features, orn_y_label_color, dir='./work_dirs', tag='tsne_orn_y')
        self.draw_tsne(t_sne_features, orn_z_label_color, dir='./work_dirs', tag='tsne_orn_z')

        self.draw_tsne(t_sne_features, gripper_color, dir='./work_dirs', tag='gripper')

 

        # for ai, A2L_dict in enumerate(A2L_dict_list):
        #     for la in A2L_dict:
        #         A2L_dict[la] = A2L_dict[la] / A2LNums[ai]
        # with open('./work_dirsla_test.json', 'w') as f:
        #     json.dump(A2L_dict_list, f)
        return metrics


@METRICS.register_module()
class LAFeatMFMetric(BaseMetric):
    def __init__(self,
                 la_num,
                 gt_act_num,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 draw_dir=None,):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.la_num = la_num
        self.gt_act_num = gt_act_num
        self.sample_model = 'ema'


    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        return None

    
    def angle_between_angles(self, a, b):
        diff = b - a
        return (diff + torch.pi) % (2 * torch.pi) - torch.pi

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # import pdb;pdb.set_trace()
        max_pos = 0.02
        max_orn = 0.05
        for i, data_sample in enumerate(data_samples):
            indice = data_sample['indice'].squeeze()
            encode_feat = data_sample['encode_feat'].squeeze(2, 3) #C, T
            state = data_batch['data_samples'][i].states[0]
            deltas = torch.stack([s1 - state[0] for s1 in state[1:]])
            rel_poses = torch.clip(deltas[:, :3], -max_pos, max_pos) / max_pos
            # rel_pos = torch.clip(delta[:3], -max_pos, max_pos) / max_pos
            # rel_pos_x, rel_pos_y, rel_pos_z = rel_pos
            rel_orns = []
            for s1 in state[1:]:
                rel_orn = self.angle_between_angles(state[0, 3:6], s1[3:6])
                rel_orns.append(rel_orn)
            rel_orns = torch.stack(rel_orns)
            rel_orns = torch.clip(rel_orns, -max_orn, max_orn) / max_orn
            gripper = state[:, -1:]
            gripper[gripper==-1] = 0

            # indice = data_sample['indice'].squeeze().item()
            # img0 = data_sample['gt_img'][:, 0].permute(1, 2, 0).cpu().numpy()
            # img1 = data_sample['gt_img'][:, 1].permute(1, 2, 0).cpu().numpy()
            data_sample['rel_action'] = data_sample['action'][:, :-1]
            data_sample['clip_start_end_id'] = data_sample['clip_start_end_id']
            data_sample['rel_pos'] = rel_poses
            data_sample['rel_orn'] = rel_orns
            data_sample['gripper'] = gripper 
            data_sample['prompt'] = data_batch['data_samples'][i].prompt
            data_sample['la_action'] = indice
            data_sample['encode_feat'] = encode_feat
            # data_sample['delta_pos']
            data_sample.pop('gt_img')
            data_sample.pop('fake_img')
            data_sample.pop('indice')
            
            self.results.append(data_sample)
    def draw_tsne(self, t_sne_features, label_color, dir, tag):
        plt.figure()
        plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1],  color=label_color, cmap='jet')
        plt.savefig(f'{dir}/{tag}.jpg')
        plt.close()
    def gen_action_id(self, action_types, results):
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
    def compute_metrics(self, results: list):
        print("**********begin to compute the metrics****************")
        # import pdb;pdb.set_trace()
        import matplotlib.patches as patches
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        metrics = {}
        bin_length = 0.1
       
        torch.save(results, './work_dirs/calvin_ldm_results.pth')
    
        rel_x_list = []
        rel_y_list = []
        rel_z_list = []

        orn_x_list = []
        orn_y_list = []
        orn_z_list = []

        gripper_list = []

        encode_feat_list = defaultdict(list)
       
        rel_x_list = defaultdict(list)
        rel_y_list = defaultdict(list)
        rel_z_list = defaultdict(list)
        orn_x_list = defaultdict(list)
        orn_y_list = defaultdict(list)
        orn_z_list = defaultdict(list)
        gripper_list = defaultdict(list)
        
    
        start_color = 'green'
        end_color = 'red'
        middle_color = 'yellow'
        # 创建颜色映射对象
        cmap = LinearSegmentedColormap.from_list("gradient", [start_color, middle_color, end_color], N=20)
        colors = cmap(np.linspace(0, 1, 20))

        action_ids = self.gen_action_id(action_types, results)
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
            t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=100).fit_transform(encode_feats)
            self.draw_tsne(t_sne_features, x_label_color[fi], dir='./work_dirs', tag=f'tsne_x_f{fi+1}')
            self.draw_tsne(t_sne_features, y_label_color[fi], dir='./work_dirs', tag=f'tsne_y_f{fi+1}')
            self.draw_tsne(t_sne_features, z_label_color[fi], dir='./work_dirs', tag=f'tsne_z_f{fi+1}')

            self.draw_tsne(t_sne_features, orn_x_label_color[fi], dir='./work_dirs', tag=f'tsne_orn_x_f{fi+1}')
            self.draw_tsne(t_sne_features, orn_y_label_color[fi], dir='./work_dirs', tag=f'tsne_orn_y_f{fi+1}')
            self.draw_tsne(t_sne_features, orn_z_label_color[fi], dir='./work_dirs', tag=f'tsne_orn_z_f{fi+1}')

            self.draw_tsne(t_sne_features, gripper_color[fi], dir='./work_dirs', tag=f'gripper_f{fi+1}')
            self.draw_tsne(t_sne_features, act_label_colors, dir='./work_dirs', tag=f'act_f{fi+1}')
 
        return metrics


@METRICS.register_module()
class LAGoFeatMetric(BaseMetric):
    def __init__(self,
                 la_num,
                 gt_act_num,
                 gt_select_frame,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 draw_dir=None,):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.la_num = la_num
        self.gt_act_num = gt_act_num
        self.sample_model = 'ema'
        self.gt_select_frame = gt_select_frame

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        return None



    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        import pdb;pdb.set_trace()
        
        for i, data_sample in enumerate(data_samples):
            indice = data_sample['indice'].squeeze().item()
            encode_feat = data_sample['encode_feat'].squeeze() #C
            gt_img = data_sample['gt_img'].permute(1, 2, 3, 0).cpu().numpy()
            img_start = gt_img[0]
            img_laters = [gt_img[idx] for idx in self.gt_select_frame]
            
            if data_sample.get('action') != None:
                actions = data_sample['action']
                captures = data_sample['capture']
                last_action = actions[-1][0]
                data_sample['actions'] = actions
                data_sample['captures'] = captures
                data_sample['gt_action'] = 9 * last_action[0] + last_action[1]
                data_sample['has_cap'] =  any([cap is not None for cap in captures])
            else:
                try:
                    gt_pos = [locate_new_piece_using_morphology(img_start, img_later) for img_later in img_laters] 
                except:
                    continue
                gt_pos = [(8 - pos[0])*9 + pos[1] for pos in gt_pos]
                data_sample['gt_action'] = gt_pos
            
            data_sample['la_action'] = indice
            data_sample.pop('gt_img')
            data_sample.pop('fake_img')
            data_sample.pop('indice')
            data_sample['encode_feat'] = encode_feat
            self.results.append(data_sample)
     
    def draw_tsne(self, t_sne_features, label_color, dir, tag):
        plt.figure()
        plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1],  color=label_color, cmap='jet')
        plt.savefig(f'{dir}/{tag}.jpg')
        plt.close()

    def compute_metrics(self, results: list):
        print("**********begin to compute the metrics****************")
        # import pdb;pdb.set_trace()
        
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        metrics = {}
        # bin_length = 0.1
        # rel_x_dict = defaultdict(list)
        # rel_y_dict = defaultdict(list)
        # rel_z_dict = defaultdict(list)
        # rel_x_list = []
        # rel_y_list = []
        # rel_z_list = []

        # orn_x_list = []
        # orn_y_list = []
        # orn_z_list = []

        # gripper_list = []
        gt_pos_list = []
        encode_feat_list = []
        # A2L_dict_list = [defaultdict(int) for _ in range(self.gt_act_num)]
        # A2LNums = defaultdict(int)
   
        torch.save(results, './work_dirs/go_ldm_results.pth')
   
        for i, item in enumerate(results):
            gt_pos = item['gt_action']
            gt_pos_list.append(gt_pos[0])
            encode_feat_list.append(item.pop('encode_feat'))
            

        from collections import Counter
        # 创建颜色映射对象
        # cmap = LinearSegmentedColormap.from_list("gradient", [start_color, middle_color, end_color], N=20)
        # unique_pos = list(set(gt_pos_list))
        pos_count = Counter(gt_pos_list)
        top_nums = [5, 10, 20, 30, 40]
        for top_num in top_nums:
            top_pos_count = pos_count.most_common(top_num)
            top_pos = [top_pos_c[0] for top_pos_c in top_pos_count]
            new_gt_pos_list = []
            new_encode_feat_list = []
            for pos, feat in zip(gt_pos_list, encode_feat_list):
                if pos in top_pos:
                    new_gt_pos_list.append(pos)
                    new_encode_feat_list.append(feat)
            colors = generate_distinct_colors(top_num)
            label_color = np.array([colors[top_pos.index(i)] for i in new_gt_pos_list])
        
            new_encode_feat_list = torch.stack(new_encode_feat_list).numpy() #N, 512
            t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=100).fit_transform(new_encode_feat_list)
            self.draw_tsne(t_sne_features, label_color, dir='./work_dirs', tag=f'tsne_go_{top_num}act')
        # self.draw_tsne(t_sne_features, y_label_color, dir='./work_dirs', tag='tsne_y')
        # self.draw_tsne(t_sne_features, z_label_color, dir='./work_dirs', tag='tsne_z')

        # self.draw_tsne(t_sne_features, orn_x_label_color, dir='./work_dirs', tag='tsne_orn_x')
        # self.draw_tsne(t_sne_features, orn_y_label_color, dir='./work_dirs', tag='tsne_orn_y')
        # self.draw_tsne(t_sne_features, orn_z_label_color, dir='./work_dirs', tag='tsne_orn_z')

        # self.draw_tsne(t_sne_features, gripper_color, dir='./work_dirs', tag='gripper')

 

        # for ai, A2L_dict in enumerate(A2L_dict_list):
        #     for la in A2L_dict:
        #         A2L_dict[la] = A2L_dict[la] / A2LNums[ai]
        # with open('./work_dirsla_test.json', 'w') as f:
        #     json.dump(A2L_dict_list, f)
        return metrics
