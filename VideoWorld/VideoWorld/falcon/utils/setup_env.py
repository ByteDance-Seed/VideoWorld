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

import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:

    import falcon.core  # noqa: F401,F403
    import falcon.datasets  # noqa: F401,F403
    import falcon.evaluation  # noqa: F401,F403
    import falcon.models  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance(
        ) is None or not DefaultScope.check_instance_created('falcon')
        if never_created:
            DefaultScope.get_instance('falcon', scope_name='falcon')
            return
        current_scope = DefaultScope.get_current_instance()
        
