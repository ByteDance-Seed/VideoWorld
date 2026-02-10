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


import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__

mmcv_minimum_version = '2.0.0'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.4.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

# assert (mmcv_version >= digit_version(mmcv_minimum_version)
#         and mmcv_version < digit_version(mmcv_maximum_version)), \
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \
#     f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        ), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

__all__ = ['__version__', 'digit_version']
