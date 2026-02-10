# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from hydra.core.config_store import ConfigStore

from videoworld2.checkpointer import Checkpointer
from imaginaire.lazy_config import LazyCall as L

STANDARD_CHECKPOINTER = L(Checkpointer)(callbacks=None)


def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="standard", node=STANDARD_CHECKPOINTER)
