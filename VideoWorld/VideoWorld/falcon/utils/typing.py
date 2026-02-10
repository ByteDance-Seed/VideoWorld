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

from typing import Callable, Dict, List, Sequence, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement
from torch import Tensor

ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
SampleList = Sequence[BaseDataElement]

NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]

ConfigType = Union[ConfigDict, Dict]
