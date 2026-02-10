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
import json


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict: The result dict contains two items, "train" and "val", for
        the training log and validate log.

    Example:
        An example output:

        .. code-block:: python

            {
                'train': [
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 100},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 200},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 300},
                    ...
                ]
                'val': [
                    {"accuracy/top1": 32.1, "step": 1},
                    {"accuracy/top1": 50.2, "step": 2},
                    {"accuracy/top1": 60.3, "step": 2},
                    ...
                ]
            }
    """
    log_dict = dict(train=[], val=[])
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # A hack trick to determine whether the line is training log.
            mode = 'train' if 'lr' in log else 'val'
            log_dict[mode].append(log)

    return log_dict
