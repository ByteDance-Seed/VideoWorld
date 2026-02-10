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
from mmengine import is_filepath
import os
import pickle
from falcon.registry import METRICS
from ..functional import (disable_gpu_fuser_on_pt19, load_i3d, prepare_i3d_feat)
from .base_gen_metric import GenerativeMetric


@METRICS.register_module('FVD-Full')
@METRICS.register_module('FVD')
@METRICS.register_module()
class FrechetVideoDistance(GenerativeMetric):
    """FID metric. In this metric, we calculate the distance between real
    distributions and fake distributions. The distributions are modeled by the
    real samples and fake samples, respectively. `Inception_v3` is adopted as
    the feature extractor, which is widely used in StyleGAN and BigGAN.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real images need for the metric. If -1
            is passed, means all real images in the dataset will be used.
            Defaults to -1.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        inception_pkl (str, optional): Path to reference inception pickle file.
            If `None`, the statistical value of real distribution will be
            calculated at running time. Defaults to None.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        need_cond_input (bool): If true, the sampler will return the
            conditional input randomly sampled from the original dataset.
            This require the dataset implement `get_data_info` and field
            `gt_label` must be contained in the return value of
            `get_data_info`. Noted that, for unconditional models, set
            `need_cond_input` as True may influence the result of evaluation
            results since the conditional inputs are sampled from the dataset
            distribution; otherwise will be sampled from the uniform
            distribution. Defaults to False.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'orig'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'FVD'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = -1,
                 inception_style='StyleGAN',
                 inception_path: Optional[str] = None,
                 inception_pkl: Optional[str] = None,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'gt_img',
                 need_cond_input: bool = False,
                 sample_model: str = 'orig',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sample_kwargs: dict = dict()):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         need_cond_input, sample_model, collect_device, prefix,
                         sample_kwargs)
        self.real_mean = None
        self.real_cov = None
        self.device = 'cpu'
        self.i3d, self.inception_style = self._load_i3d(
            inception_style, inception_path)
        self.inception_pkl = inception_pkl
        # import pdb;pdb.set_trace()
        if isinstance(inception_pkl, str):
            if is_filepath(inception_pkl) and os.path.exists(inception_pkl):
                with open(inception_pkl, 'rb') as file:
                    inception_state = pickle.load(file)
            if is_main_process():
                self.real_mean = inception_state['real_mean']
                self.real_cov = inception_state['real_cov']
            
    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Preparing inception feature for the real images.

        Args:
            module (nn.Module): The model to evaluate.
            dataloader (DataLoader): The dataloader for real images.
        """
        # import pdb;pdb.set_trace()
        self.device = module.data_preprocessor.device
        self.i3d.to(self.device)
        self.i3d.eval()
        inception_feat_dict = prepare_i3d_feat(
            dataloader, self, module.data_preprocessor, capture_mean_cov=True)
        if is_main_process():
            self.real_mean = inception_feat_dict['real_mean']
            self.real_cov = inception_feat_dict['real_cov']

    def _load_i3d(self, inception_style: str,
                        inception_path: Optional[str]
                        ) -> Tuple[nn.Module, str]:
        """Load inception and return the successful loaded style.

        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        if inception_style == 'StyleGAN':
            args = dict(type='StyleGAN', inception_path=inception_path)
        else:
            args = dict(type='Pytorch', normalize_input=False)
        inception, style = load_i3d(args, 'FVD')
        inception.eval()
        return inception, style

    def forward_i3d(self, image: Tensor) -> Tensor:
        """Feed image to inception network and get the output feature.

        Args:
            data_samples (Sequence[dict]): A batch of data sample dict used to
                extract inception feature.

        Returns:
            Tensor: Image feature extracted from inception.
        """
        # image must passed with 'bgr'
        image = image[:, [2, 1, 0]].to(self.device)

        if self.inception_style == 'StyleGAN':
            image = (image - 127.5) / 127.5
            # image = image.to(torch.uint8)
            self.i3d.eval()
            with disable_gpu_fuser_on_pt19():
                feat = self.i3d(image, return_features=True,rescale=False, resize=True)
        else:
            image = (image - 127.5) / 127.5  # to [-1, 1]
            feat = self.i3d(image)[0].view(image.shape[0], -1)
        return feat.detach()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']
            fake_imgs.append(fake_img_)

        # check whether shape in fake_imgs are same
        # import pdb;pdb.set_trace()
        img_shape = fake_imgs[0].shape
        if all([img.shape == img_shape for img in fake_imgs]):
            # all images have the same shape, forward inception altogether
            fake_imgs = torch.stack(fake_imgs, dim=0)
            feat = self.forward_i3d(fake_imgs.float()) 
            feat_list = list(torch.split(feat, 1))
        else:
            # images have different shape, forward separately
            feat_list = [
                self.forward_i3d(img[None, ...]) for img in fake_imgs
            ]
        self.fake_results += feat_list

    @staticmethod
    def _calc_fid(sample_mean: np.ndarray,
                  sample_cov: np.ndarray,
                  real_mean: np.ndarray,
                  real_cov: np.ndarray,
                  eps: float = 1e-6) -> Tuple[float]:
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        # import pdb;pdb.set_trace()
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))
                print(f'____________Imaginary component_________ {m}')
                # raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return float(fid), float(mean_norm), float(trace)

    def compute_metrics(self, fake_results: list) -> dict:
        """Compute the result of FID metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed FID metric and its mean and
                covariance.
        """
        # import pdb;pdb.set_trace()
        fake_feats = torch.cat(fake_results, dim=0)
        fake_feats_np = fake_feats.cpu().numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)

        return {'fid': fid, 'mean': mean, 'cov': cov}

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenerativeMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader


