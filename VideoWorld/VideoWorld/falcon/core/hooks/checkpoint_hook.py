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
import os
from mmengine.hooks import CheckpointHook as BaseCheckpointHook
from mmengine.structures import BaseDataElement
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
from math import inf
import os.path as osp
from falcon.registry import HOOKS
from mmengine.dist import is_main_process, master_only
DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class CheckpointHook(BaseCheckpointHook):
    
    out_dir: str

    priority = 'VERY_LOW'

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 save_hdfs: bool = False,
                 **kwargs) -> None:
        super().__init__(
                 interval,
                 by_epoch,
                 save_optimizer,
                 save_param_scheduler,
                 out_dir,
                 max_keep_ckpts,
                 save_last,
                 save_best,
                 rule,
                 greater_keys,
                 less_keys,
                 file_client_args,
                 filename_tmpl,
                 backend_args,
                 published_keys,
                 save_begin,
                 **kwargs)
 
        self.save_hdfs = save_hdfs
        self.hdfs_root = 'hdfs://haruna/home/byte_uslab_cvg/user/xjjin/zhongwei_exp'

    

    def _save_checkpoint_with_step(self, runner, step, meta):
        # import pdb;pdb.set_trace()
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore

        if self.save_hdfs:
            work_dir = runner.log_dir
            exp_name = work_dir.split('/')[-2]
            exp_date = work_dir.split('/')[-1]
            for file in os.listdir(work_dir):
                if not file.endswith('.pth'):
                    os.system(f'hdfs dfs -put -f {work_dir}/{file} {self.hdfs_root}/{exp_name}/{exp_date}/')
            # import pdb;pdb.set_trace()
            for file in os.listdir(runner.work_dir):
                if file.endswith('.pth') and os.system(f'hdfs dfs -test -e {self.hdfs_root}/{exp_name}/{file}'):
                    os.system(f'hdfs dfs -put -f {runner.work_dir}/{file} {self.hdfs_root}/{exp_name}/')

    

    