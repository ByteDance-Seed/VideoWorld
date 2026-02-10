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

import numpy as np
import torch
import torch.nn as nn
from mmengine.dist import is_main_process
from scipy import linalg
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from mmengine.evaluator import BaseMetric
from mmengine import is_filepath
import os
import pickle
from falcon.registry import METRICS
import matplotlib.pyplot as plt
import json
from collections import defaultdict
def draw_data(pred_prior, kata_prior, output_dir, idx, level, pred_winrate=None):
    # 你的数据
    #_data1 = {4: 0.17381612772379787, 6: 0.10311675440330001, 8: 0.0701606094392247, 10: 0.04628760877803369, 12: 0.025396883384183905, 14: 0.02165092398869654, 16: 0.011567344107814117, 18: 0.006278794553436143, 20: 0.002653338699936667, 22: 0.0018393022004938749, 24: 0.0012758540757955694, 26: 0.0016138638148312503, 28: 0.0008847561339347824, 30: 0.0006842214771265641, 32: 0.0005668473605460328, 34: 0.0005452372871143084, 36: 0.00043072494227361553, 38: 0.00048311725186781917, 40: 0.0005512011989283845, 42: 0.0005382811692043833, 44: 0.00024744656489556946, 46: 0.0001971471930487493, 48: 0.000252419825774728, 50: 0.0002406349955826, 52: 0.00034721789768178435, 54: 0.00022039276656983438, 56: 0.00011404316227673578, 58: 0.00017454566750131455, 60: 0.00014638358543206192, 62: 1.6849637749478e-05, 64: 0.00061441237249833, 3: 0.5165943595, 5: 0.9984671606666667, 7: 0.9943311723333332, 9: 0.9464150183333334, 11: 0.14760818027666667, 13: 0.004244168485, 15: 0.0007670498136666667, 17: 0.000478974941, 19: 0.00027702223193, 21: 0.00020378042951000003, 23: 0.000130212584, 25: 7.112645015e-05, 27: 4.722220681e-05, 29: 0.00011299381685, 31: 6.33913398e-05, 66: 6.42951248e-09, 68: 5.92445704e-09}
    # _data2 = {4: 0.17221197762327659, 6: 0.10837434237576594, 8: 0.05581078701813828, 10: 0.020415392810085108, 12: 0.0037083366520860197, 14: 0.0015212774453215224, 16: 0.0006171941426609521, 18: 0.0003449669666436619, 20: 0.0002169081428386885, 22: 0.00018499523051089287, 24: 0.000186186513406383, 26: 0.0001698148659076923, 28: 0.00019477108841571428, 30: 0.00016787185125967743, 32: 0.00016712844024181997, 34: 0.00012533783558764, 36: 6.0519684272574205e-05, 38: 6.252267918779868e-05, 40: 8.703706516104385e-05, 42: 6.231815398162511e-05, 3: 0.5164727735166667, 5: 0.5451484659166667, 7: 0.40968673564999997, 9: 0.3324837111266667, 11: 0.38386442460032, 13: 0.20115722969108002, 15: 0.047048128064240004, 17: 0.00129075502216, 19: 0.000610327814775, 21: 4.818778533333333e-05, 23: 5.25298049e-05, 44: 1.872540243016549e-05, 46: 6.656091048056998e-06, 48: 5.345049534116514e-06, 50: 9.437757163400999e-07, 52: 5.1466249232897495e-06, 54: 3.6791756172e-06, 56: 6.39096553e-11, 58: 5.80018256e-11, 60: 1.2349255e-10, 62: 1.86571203e-10, 64: 6.58009203e-11, 25: 9.088009921500001e-06, 27: 2.74197713248e-05, 29: 3.40517458665e-05, 31: 7.60568238e-06, 33: 8.69699872e-07, 35: 2.58372752e-07, 37: 2.03791844e-07, 39: 2.62898592e-11, 41: 1.27402533e-11, 43: 3.543001e-09, 45: 1.57056405e-07, 47: 1.72877268e-11}
    # _data3 = {4: 0.17370279450998935, 6: 0.10084246841039782, 8: 0.05041381487045163, 10: 0.028518101865045463, 12: 0.003690199486590909, 14: 0.00136263164097093, 16: 0.0008469952841279069, 18: 0.0006183755614109412, 20: 0.0003117164402527381, 22: 0.00023524252367493667, 24: 0.00018184914286763167, 26: 0.00015814965226407894, 28: 0.00018482565382477536, 30: 0.00016199368035515902, 32: 0.0001228727492422016, 34: 9.817186608143528e-05, 36: 8.927107411500139e-05, 38: 5.8736690785440705e-05, 40: 4.378514779331969e-05, 42: 3.490261850666347e-05, 44: 1.4396188957633807e-05, 46: 5.012504139797999e-06, 48: 2.5379099978662746e-06, 50: 7.195290088566421e-06, 52: 7.440366346686438e-07, 54: 5.159488582625295e-07, 56: 7.161742348383701e-08, 58: 2.188431612934241e-06, 60: 2.086806753296392e-06, 62: 3.409272971708581e-06, 3: 0.5161750853833333, 5: 0.998370751, 7: 0.9968938083333333, 9: 0.5027935176666666, 11: 0.1436820435, 13: 0.12786827015333332, 15: 0.0037987686233333334, 17: 0.004607193496666667, 19: 0.0009658570976666666, 21: 0.0005482542973333334, 23: 0.0003934085846666667, 25: 0.0003181119426666666, 27: 0.0001372537553, 29: 0.0001553416785333333, 31: 0.0003121780863333333, 33: 0.000248232493345, 35: 0.0002295280654346333, 37: 0.00016557107921476668, 39: 9.1820308332941e-05, 41: 7.581503095533333e-05, 43: 8.623206257691038e-05, 45: 1.0461426275238e-06, 47: 6.958440909491668e-07, 49: 2.568459319715e-07, 51: 5.77909985285e-08, 53: 1.2422218795000002e-10, 55: 9.372241890000001e-11, 64: 6.017863224023346e-06, 66: 1.9681181321307594e-06, 68: 1.7770958778666665e-10, 70: 2.950172052749999e-11, 72: 9.992028042499999e-11, 74: 2.0074645633333335e-11, 76: 2.40251152e-11, 78: 2.86065616e-11, 80: 2.91229263e-11} 
    output_dir = '/opt/tiger/' if output_dir is None else output_dir
    # 分别获取字典的键和值，作为横坐标和纵坐标
    # import pdb;pdb.set_trace()
    pred_prior_dict = {}
    kata_prior_dict = {}
    for i in pred_prior:
        prior = pred_prior[i]
        pred_prior_dict[i*2] = sum(prior) / len(prior)
    for i in kata_prior:
        prior = kata_prior[i]
        kata_prior_dict[i*2] = sum(prior) / len(prior)

    x1 = list(pred_prior_dict.keys())
    y1 = np.array(list(pred_prior_dict.values()))
    y1[y1 < 0] = 1e-2
    y1 = list(y1)

    x2 = list(kata_prior_dict.keys())
    y2 = list(kata_prior_dict.values())

    fig = plt.figure()
    # plt.cla()
    # plt.clf()
    # 绘制曲线图
    plt.plot(x1, y1, marker='o', label='Pred', markersize=4)
    plt.plot(x2, y2, marker='s', label=f'Kata_{level}', markersize=4)
    # 添加标题和坐标轴标签
    plt.title('Data Plot')
    plt.xlabel('Step')
    plt.ylabel('Prior')
    plt.yscale('log')
    # 显示网格
    plt.grid(True)
    # 显示图例
    plt.legend()
    # 显示图形
    plt.savefig(f'{output_dir}/prior_{level}_{idx}.png', dpi=300) 
    plt.close()

    if pred_winrate is not None:
        fig = plt.figure()
        winrate_dict = {}
        for i in pred_winrate:
            prior = pred_winrate[i]
            winrate_dict[i*2] = sum(prior) / len(prior)
        x1 = list(winrate_dict.keys())
        y1 = list(winrate_dict.values())
        plt.plot(x1, y1, marker='o', label='Pred', markersize=4)
        # plt.plot(x2, y2, marker='s', label=f'Kata_{level}', markersize=4)
        # 添加标题和坐标轴标签
        plt.title('Data Plot')
        plt.xlabel('Step')
        plt.ylabel('Winrate')
        plt.yscale('log')
        # 显示网格
        plt.grid(True)
        # 显示图例
        plt.legend()
        # 显示图形
        plt.savefig(f'{output_dir}/winrate_{level}_{idx}.png', dpi=300) 
        plt.close()


@METRICS.register_module()
class GoEva(BaseMetric):
    
    name = 'GoEva'

    def __init__(self,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 draw_dir=None,
                 mode='acc' ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.output_dir = output_dir
        self.mode = mode
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.draw_dir = draw_dir
        self.idx = 0
    

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            self.results.append(data_sample)

            
    def compute_metrics(self, results: list):
        # import pdb;pdb.set_trace()
        print("**********begin to compute the metrics****************")
        # if self.mode == 'go_battle':
        metrics = {}
        return metrics



    