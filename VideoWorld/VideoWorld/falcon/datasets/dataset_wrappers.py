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
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from falcon.registry import DATASETS


@DATASETS.register_module()
class MyConcatDataset(_ConcatDataset):
    def __init__(self, datasets):
        self.datasets = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, Dataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(f"elements in datasets sequence should be config or"
                                f" `BaseDataset` instance, but got {type(dataset)}")

        super(MyConcatDataset, self).__init__(self.datasets)
