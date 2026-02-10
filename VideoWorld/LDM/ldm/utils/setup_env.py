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
import importlib
import warnings
from types import ModuleType
from typing import Optional

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:

    import mmagic.datasets  # noqa: F401,F403
    import mmagic.engine  # noqa: F401,F403
    import mmagic.evaluation  # noqa: F401,F403
    import mmagic.models  # noqa: F401,F403
    import mmagic.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
            or not DefaultScope.check_instance_created('mmagic')
        if never_created:
            DefaultScope.get_instance('mmagic', scope_name='mmagic')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmagic':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmagic", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmagic". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmagic-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmagic')


def try_import(name: str) -> Optional[ModuleType]:
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None

