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
from mmengine.hooks import IterTimerHook as BaseIterTimerHook
from mmengine.structures import BaseDataElement

from falcon.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class IterTimerHook(BaseIterTimerHook):
    """IterTimerHooks inherits from :class:`mmengine.hooks.IterTimerHook` and
    overwrites :meth:`self._after_iter`.

    This hooks should be used along with
    :class:`mmagic.engine.runner.MultiValLoop` and
    :class:`mmagic.engine.runner.MultiTestLoop`.
    """
    def __init__(self, save_hdfs=False):
        super().__init__()
        self.save_hdfs = save_hdfs
        self.hdfs_root = 'hdfs://haruna/home/byte_uslab_cvg/user/xjjin/zhongwei_exp'

    def before_train(self, runner) -> None:
        """Synchronize the number of iterations with the runner after resuming
        from checkpoints.

        Args:
            runner: The runner of the training, validation or testing
                process.
        """
        # import pdb;pdb.set_trace()
        self.start_iter = runner.iter
        if self.save_hdfs:
            work_dir = runner.log_dir
            exp_name = work_dir.split('/')[-2]
            exp_date = work_dir.split('/')[-1]
            os.system(f'hdfs dfs -mkdir -p {self.hdfs_root}/{exp_name}/{exp_date}')

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict,
                                            Sequence[BaseDataElement]]] = None,
                    mode: str = 'train') -> None:
        """Calculating time for an iteration and updating "time"
        ``HistoryBuffer`` of ``runner.message_hub``. If `mode` is 'train', we
        take `runner.max_iters` as the total iterations and calculate the rest
        time. If `mode` in `val` or `test`, we use
        `runner.val_loop.total_length` or `runner.test_loop.total_length` as
        total number of iterations. If you want to know how `total_length` is
        calculated, please refers to
        :meth:`mmagic.engine.runner.MultiValLoop.run` and
        :meth:`mmagic.engine.runner.MultiTestLoop.run`.

        Args:
            runner (Runner): The runner of the training validation and
                testing process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict or sequence, optional): Outputs from model. Defaults
                to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # Update iteration time in `runner.message_hub`.
        message_hub = runner.message_hub
        message_hub.update_scalar(f'{mode}/time', time.time() - self.t)
        self.t = time.time()
        iter_time = message_hub.get_scalar(f'{mode}/time')
        if mode == 'train':
            self.time_sec_tot += iter_time.current()
            # Calculate average iterative time.
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1)
            # Calculate eta.
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            runner.message_hub.update_info('eta', eta_sec)
        else:
            if mode == 'val':
                cur_dataloader = runner.val_dataloader
            else:
                cur_dataloader = runner.test_dataloader

            self.time_sec_test_val += iter_time.current()
            time_sec_avg = self.time_sec_test_val / (batch_idx + 1)
            eta_sec = time_sec_avg * (len(cur_dataloader) - batch_idx - 1)
            runner.message_hub.update_info('eta', eta_sec)
        
        