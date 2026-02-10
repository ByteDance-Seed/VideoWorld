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
from typing import List, Tuple

from mmcv.transforms.base import BaseTransform

from ldm.registry import TRANSFORMS
from ldm.structures import DataSample
from ldm.utils import all_to_tensor


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack data into DataSample for training, evaluation and testing.

    MMagic follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of DataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    def __init__(
        self,
        keys: Tuple[List[str], str] = ['merged', 'img'],
        meta_keys: Tuple[List[str], str] = [],
        data_keys: Tuple[List[str], str] = ['gt_img',],
    ) -> None:

        assert keys is not None, \
            'keys in PackInputs can not be None.'
        assert data_keys is not None, \
            'data_keys in PackInputs can not be None.'
        assert meta_keys is not None, \
            'meta_keys in PackInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys,
                                                 List) else [meta_keys]

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`DataSample`): The annotation info of the
                sample.
        """

        # prepare inputs
        inputs = dict()
        results['gt_img']=results['img']
        for k in self.keys:
            value = results.get(k, None)
            if value is not None:
                inputs[k] = all_to_tensor(value)

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.values())[0]

        data_sample = DataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {
            k: v
            for (k, v) in results.items()
            if k not in (self.data_keys + self.meta_keys)
        }
        data_sample.set_predefined_data(predefined_data)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {
            k: v
            for (k, v) in results.items() if k in self.meta_keys
        }
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            k: v
            for (k, v) in results.items() if k in self.data_keys
        }
        data_sample.set_tensor_data(required_data)

        return {'inputs': inputs, 'data_samples': data_sample}

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str


@TRANSFORMS.register_module()
class PackVideoInputs(BaseTransform):
    """Pack data into DataSample for training, evaluation and testing.

    MMagic follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of DataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    def __init__(
        self,
        keys: Tuple[List[str], str] = ['merged', 'imgs'],
        meta_keys: Tuple[List[str], str] = [],
        data_keys: Tuple[List[str], str] = ['gt_img',],
    ) -> None:

        assert keys is not None, \
            'keys in PackInputs can not be None.'
        assert data_keys is not None, \
            'data_keys in PackInputs can not be None.'
        assert meta_keys is not None, \
            'meta_keys in PackInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys,
                                                 List) else [meta_keys]

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`DataSample`): The annotation info of the
                sample.
        """

        # prepare inputs
        # import pdb;pdb.set_trace()
        inputs = dict()
        results['gt_img']=results['imgs']
        for k in self.keys:
            value = results.get(k, None)
            if value is not None:
                inputs[k] = all_to_tensor(value)

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.values())[0]

        data_sample = DataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {
            k: v
            for (k, v) in results.items()
            if k not in (self.data_keys + self.meta_keys)
        }
        data_sample.set_predefined_data(predefined_data)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {
            k: v
            for (k, v) in results.items() if k in self.meta_keys
        }
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            k: v
            for (k, v) in results.items() if k in self.data_keys
        }
        data_sample.set_tensor_data(required_data)

        return {'inputs': inputs, 'data_samples': data_sample}

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str


