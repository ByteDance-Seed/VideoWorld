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
import numpy as np
import cv2
import os
ann_file = open('/opt/tiger/gr1_rollout.txt', 'r')
lines = ann_file.readlines()
for li, line in enumerate(lines):
    # import pdb;pdb.set_trace()
    _line = line.split(' ')
    start, end, lang = _line[0], _line[1], _line[2]
    start = int(start)
    end = int(end)
    print(start, end, lang)
    for index in range(start, end+1):
        npz_path = f'/opt/tiger/gr1_rollout/episode_{index:07d}.npz'
        data = np.load(npz_path)
        os.system(f'mkdir -p /opt/tiger/gr1_vis/{li}')
        import pdb;pdb.set_trace()
        cv2.imwrite(f'/opt/tiger/gr1_vis/{li}/episode_{index:07d}.png', data['rgb_static'][:, :, ::-1])
