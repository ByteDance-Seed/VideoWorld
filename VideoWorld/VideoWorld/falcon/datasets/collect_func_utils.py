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
from typing import Any, Mapping, Sequence

import torch
from mmengine.registry import FUNCTIONS
from mmengine.structures import BaseDataElement
from torch.utils.data._utils.collate import \
    default_collate as torch_default_collate


@FUNCTIONS.register_module()
def default_collate_token(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    Different from :func:`pseudo_collate`, ``default_collate`` will stack
    tensor contained in ``data_batch`` into a batched tensor with the
    first dimension batch size, and then move input tensor to the target
    device.

    Different from ``default_collate`` in pytorch, ``default_collate`` will
    not process ``BaseDataElement``.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Note:
        ``default_collate`` only accept input tensor with the same shape.

    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data_itement of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)

    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(default_collate_token(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [default_collate_token(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [default_collate_token(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [default_collate_token(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        tmp = {key: [d[key] for d in data_batch] for key in data_item}
        out = dict()
        for key, val in tmp.items():
            if key in ['prefix_ids', ]:
                ## left padding
                out[key] = torch.nn.utils.rnn.pad_sequence([item.flip(dims=[0]) for item in val], batch_first=True,
                                                           padding_value=2).flip(dims=[1])
            elif key in ['suffix_ids', ]:
                out[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True, padding_value=2)
            elif key in ['prefix_ids_att', ]:
                out[key] = torch.nn.utils.rnn.pad_sequence([item.flip(dims=[0]) for item in val], batch_first=True,
                                                           padding_value=0).flip(dims=[1])
            elif key in ['suffix_ids_att', ]:
                out[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True, padding_value=0)
            elif key in ['suffix_ids_label', ]:
                out[key] = torch.nn.utils.rnn.pad_sequence(val, batch_first=True, padding_value=-100)
            else:
                out[key] = default_collate_token(val)
        return data_item_type(out)
        # return data_item_type({
        #     key: default_collate([d[key] for d in data_batch])
        #     for key in data_item
        # })
    else:
        return torch_default_collate(data_batch)
