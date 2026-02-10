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

import time
from typing import Optional, Sequence, Union

from mmengine.hooks import Hook 
import os

from ldm.registry import HOOKS

@HOOKS.register_module()
class SaveHDFSHook(Hook):
    
    def __init__(self):
        self.hdfs_root = 'hdfs://haruna/home/byte_uslab_cvg/user/xjjin/zhongwei_exp'
    def before_train(self, runner) -> None:
        work_dir = runner.work_dir
        exp_name = work_dir.split('/')[-2]
        exp_date = work_dir.split('/')[-1]
        os.system(f'hdfs dfs -mkdir -p {self.hdfs_root}/{exp_name}/{exp_date}')

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[Sequence, dict]] = None,
                    mode: str = 'train'):
        
        work_dir = runner.work_dir
        exp_name = work_dir.split('/')[-2]
        exp_date = work_dir.split('/')[-1]
        for file in os.listdir(work_dir):
            if not file.endswith('.pth'):
                os.system(f'hdfs dfs -put -f {work_dir}/{file} {self.hdfs_root}/{exp_name}/{exp_date}/')
            elif file.endswith('.pth') and not os.system(f'hdfs dfs -test -e {self.hdfs_root}/{exp_name}/{exp_date}/{file}'):
                os.system(f'hdfs dfs -put -f {work_dir}/{file} {self.hdfs_root}/{exp_name}/{exp_date}/')
    