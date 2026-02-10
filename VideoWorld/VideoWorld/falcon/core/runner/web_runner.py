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
import copy
import warnings
from functools import partial
from typing import Dict, Optional, Union

import webdataset as wds
from mmengine.dataset import COLLATE_FUNCTIONS, worker_init_fn
from mmengine.dist import get_dist_info, get_rank
from mmengine.runner import FlexibleRunner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from torch.utils.data import DataLoader
from collections import namedtuple
from falcon.registry import DATA_SAMPLERS, DATASETS, RUNNERS
from falcon.utils import go_image_tar_decoder
from mmengine.registry import build_from_cfg
from falcon.registry import TRANSFORMS
from torchvision.transforms import Compose

@RUNNERS.register_module()
class WebRunner(FlexibleRunner):

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)
        # import pdb;pdb.set_trace()
        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict) and '_Web' not in dataset_cfg['type']: 
            dataset = DATASETS.build(dataset_cfg)
        elif isinstance(dataset_cfg, dict) and '_WebGo' in dataset_cfg['type']:
            shards = dataset_cfg['shards']
            test_mode = dataset_cfg.get('test_mode', False)
            dataset = wds.WebDataset(shards, resampled=True)
            pipeline_cfg = dataset_cfg.get('pipeline', False)
            mf = dataset_cfg.get('mf', False)
            sample_num = dataset_cfg.get('sample_num', 2)
            size = dataset_cfg.get('size', 256)
            pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline_cfg]
            pipeline = Compose(pipeline)
            dataset = dataset.map(partial(go_image_tar_decoder, pipeline=pipeline, sample_num=sample_num, mf=mf, size=size)).shuffle(10000)
            dataset.with_length(dataset_cfg['dataset_length'])
            # dataset.batched(dataloader_cfg['batch_size'], partial=True)

        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        use_web = dataloader_cfg.pop('use_web', False)
        
        if use_web:
            # build dataloader
            init_fn: Optional[partial]
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

            # `persistent_workers` requires pytorch version >= 1.7
            if ('persistent_workers' in dataloader_cfg
                    and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
                warnings.warn('`persistent_workers` is only available when '
                              'pytorch version >= 1.7')
                dataloader_cfg.pop('persistent_workers')

            # The default behavior of `collat_fn` in dataloader is to
            # merge a list of samples to form a mini-batch of Tensor(s).
            # However, in mmengine, if `collate_fn` is not defined in
            # dataloader_cfg, `pseudo_collate` will only convert the list of
            # samples into a dict without stacking the batch tensor.
            collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                                {'type': 'pseudo_collate'})

            collate_fn_type = collate_fn_cfg.pop('type')
            collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore

            batch_size = dataloader_cfg.pop('batch_size')
            data_len = len(dataset)
            # import pdb;pdb.set_trace()
            # train_nbatches = max(1, data_len //world_size)
            # dataset = (dataset.with_epoch(train_nbatches).with_length(train_nbatches))
            data_loader = wds.WebLoader(
                dataset.batched(batch_size, collate_fn, partial=False),
                # dataset,
                batch_size=None,
                worker_init_fn=init_fn,
                **dataloader_cfg)
            # data_loader = data_loader.ddp_equalize(dataset_cfg['dataset_length'] // batch_size)
            #{'pin_memory': True, 'num_workers': 0}
            _, world_size = get_dist_info()
            train_nbatches = max(1, data_len // batch_size)
            if test_mode:
                data_loader = data_loader.with_length(train_nbatches)
            else:
                data_loader = data_loader.with_length(train_nbatches)
            data_loader.dataset = dataset
            # batch_sampler = namedtuple('batch_sampler', ['batch_size'])
            data_loader.batch_size = batch_size
            return data_loader

        else:
            # build sampler
            sampler_cfg = dataloader_cfg.pop('sampler')
            if isinstance(sampler_cfg, dict):
                sampler_seed = None if diff_rank_seed else seed
                sampler = DATA_SAMPLERS.build(
                    sampler_cfg,
                    default_args={
                        'dataset': dataset,
                        'seed': sampler_seed
                    })
            else:
                # fallback to raise error in dataloader
                # if `sampler_cfg` is not a valid type
                sampler = sampler_cfg

            # build batch sampler
            batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
            if batch_sampler_cfg is None:
                batch_sampler = None
            elif isinstance(batch_sampler_cfg, dict):
                batch_sampler = DATA_SAMPLERS.build(
                    batch_sampler_cfg,
                    default_args={
                        'sampler': sampler,
                        'batch_size': dataloader_cfg.pop('batch_size')
                    })
            else:
                # fallback to raise error in dataloader
                # if `batch_sampler_cfg` is not a valid type
                batch_sampler = batch_sampler_cfg

            # build dataloader
            init_fn: Optional[partial]

            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

            # `persistent_workers` requires pytorch version >= 1.7
            if ('persistent_workers' in dataloader_cfg
                    and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
                warnings.warn('`persistent_workers` is only available when '
                              'pytorch version >= 1.7')
                dataloader_cfg.pop('persistent_workers')

            # The default behavior of `collat_fn` in dataloader is to
            # merge a list of samples to form a mini-batch of Tensor(s).
            # However, in mmengine, if `collate_fn` is not defined in
            # dataloader_cfg, `pseudo_collate` will only convert the list of
            # samples into a dict without stacking the batch tensor.
            collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                                {'type': 'pseudo_collate'})
            collate_fn_type = collate_fn_cfg.pop('type')
            collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
            data_loader = DataLoader(
                dataset=dataset,
                sampler=sampler if batch_sampler is None else None,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                worker_init_fn=init_fn,
                **dataloader_cfg)
            # import pdb;pdb.set_trace()
            # print("s_______________:", len(data_loader))
            return data_loader

    # def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
    #     """Build model.
    #
    #         If ``model`` is a dict, it will be used to build a nn.Module object.
    #         Else, if ``model`` is a nn.Module object it will be returned directly.
    #
    #         An example of ``model``::
    #
    #             model = dict(type='ResNet')
    #
    #         Args:
    #             model (nn.Module or dict): A ``nn.Module`` object or a dict to
    #                 build nn.Module object. If ``model`` is a nn.Module object,
    #                 just returns itself.
    #
    #         Note:
    #             The returned model must implement ``train_step``, ``test_step``
    #             if ``runner.train`` or ``runner.test`` will be called. If
    #             ``runner.val`` will be called or ``val_cfg`` is configured,
    #             model must implement `val_step`.
    #
    #         Returns:
    #             nn.Module: Model build from ``model``.
    #         """
    #
    #     with init_empty_weights():
    #         if isinstance(model, nn.Module):
    #             return model
    #         elif isinstance(model, dict):
    #             model = MODELS.build(model)
    #             return model  # type: ignore
    #         else:
    #             raise TypeError('model should be a nn.Module object or dict, '
    #                             f'but got {model}')
