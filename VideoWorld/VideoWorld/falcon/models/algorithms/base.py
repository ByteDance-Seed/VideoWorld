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

from typing import Optional

import torch
from mmengine.model import BaseModel as _BaseModel

from falcon.registry import MODELS


class BaseModel(_BaseModel):
    """BaseModel for SelfSup.

    All algorithms should inherit this module.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmcls.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmcls.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmcls.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        target_generator: (dict, optional): The target_generator module to
            generate targets for self-supervised learning optimization, such as
            HOG, extracted features from other modules(DALL-E, CLIP), etc.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
            self,
            generator: Optional[dict] = None,
            vbackbone: Optional[dict] = None,
            lbackbone: Optional[dict] = None,
            neck: Optional[dict] = None,
            head: Optional[dict] = None,
            pretrained: Optional[str] = None,
            # data_preprocessor: Optional[Union[dict, nn.Module]] = None,
            init_cfg: Optional[dict] = None):

        # if pretrained is not None:
        #     init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        #
        # if data_preprocessor is None:
        #     data_preprocessor = {}
        # # The build process is in MMEngine, so we need to add scope here.
        # data_preprocessor.setdefault('type',
        #                              'mmselfsup.SelfSupDataPreprocessor')

        super().__init__(init_cfg=init_cfg)

        if vbackbone is not None:
            self.vbackbone = MODELS.build(vbackbone)
        
        if generator is not None:
            self.generator = MODELS.build(generator)

        if lbackbone is not None:
            self.lbackbone = MODELS.build(lbackbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

    @property
    def with_generator(self):
        return hasattr(self, 'generator') and self.generator is not None

    @property
    def with_vbackbone(self):
        return hasattr(self, 'vbackbone') and self.vbackbone is not None

    @property
    def with_lbackbone(self):
        return hasattr(self, 'lbackbone') and self.lbackbone is not None

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def forward(self, img=None, mode='loss', **kwargs):
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        This module overwrites the abstract method in ``BaseModel``.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults (dict or list):
              - If ``mode == loss``, return a ``dict`` of loss tensor used
                for backward and logging.
              - If ``mode == predict``, return a ``list`` of
                :obj:`BaseDataElement` for computing metric
                and getting inference result.
              - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                or ``dict of tensor for custom use.
        """
        # import pdb;pdb.set_trace()
        if mode == 'tensor':
            feats = self.extract_feat(img, **kwargs)
            return feats
        elif mode == 'loss':
            return self.forward_train(img, **kwargs)
        elif mode == 'predict':
            return self.forward_test(img, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from the input tensor with shape (N, C, ...).

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation.
        """
        raise NotImplementedError

    def forward_train(self, img, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[SelfSupDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        raise NotImplementedError

    def forward_test(self, img, **kwargs) -> dict:
        """Predict results from the extracted features.

        This module returns the logits before loss, which are used to compute
        all kinds of metrics. This is a abstract method, and subclass should
        overwrite this methods if needed.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        raise NotImplementedError
