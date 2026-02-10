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
# Copyright (c) Open-MMLab. All rights reserved.

__version__ = '1.2.0dev0'


def parse_version_info(version_str):
    ver_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            ver_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            ver_info.append(int(patch_version[0]))
            ver_info.append(f'rc{patch_version[1]}')
    return tuple(ver_info)


version_info = parse_version_info(__version__)
