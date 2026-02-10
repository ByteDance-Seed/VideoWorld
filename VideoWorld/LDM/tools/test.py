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

import argparse
import os
import os.path as osp
from collections import OrderedDict

import torch
import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

from ldm.utils import print_colored_log


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def process_checkpoint(path):
    data = torch.load(path,map_location='cpu')
    out = OrderedDict()
    for key,val in data['state_dict'].items():
        if key.startswith('generator.module.'):
            key = key.replace('generator.module.','generator.')
        elif key.startswith('discriminator.module.'):
            key = key.replace('discriminator.module.','discriminator.')
        out[key]=val
    data['state_dict']=out

    file_name = path.split('.pth')[0]


    torch.save(data,file_name+"_new.pth")


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.checkpoint != 'none':
        process_checkpoint(args.checkpoint)
    new_path = args.checkpoint.split('.pth')[0]+"_new.pth"
    args.checkpoint = new_path
    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    print_colored_log(f'Working directory: {cfg.work_dir}')
    print_colored_log(f'Log directory: {runner._log_dir}')

    if args.out:

        class SaveMetricHook(Hook):

            def after_test_epoch(self, _, metrics=None):
                if metrics is not None:
                    mmengine.dump(metrics, args.out)

        runner.register_hook(SaveMetricHook(), 'LOWEST')

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
