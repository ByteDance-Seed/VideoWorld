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

import re
import sys
import warnings


def modify_args():
    """Modify args of argparse.ArgumentParser."""
    for i, v in enumerate(sys.argv):
        if i == 0:
            assert v.endswith('.py')
        elif re.match(r'--\w+_.*', v):
            new_arg = v.replace('_', '-')
            warnings.warn(
                f'command line argument {v} is deprecated, '
                f'please use {new_arg} instead.',
                category=DeprecationWarning,
            )
            sys.argv[i] = new_arg
