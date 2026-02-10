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

import logging

from mmengine.logging import print_log
from termcolor import colored


def print_colored_log(msg, level=logging.INFO, color='magenta'):
    """Print colored log with default logger.

    Args:
        msg (str): Message to log.
        level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.Log level,
            default to 'info'.
        color (str, optional): Color 'magenta'.
    """
    print_log(colored(msg, color), 'current', level)
