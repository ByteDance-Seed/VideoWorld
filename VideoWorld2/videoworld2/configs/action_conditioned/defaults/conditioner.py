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
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from videoworld2.configs.vid2vid.defaults.conditioner import Vid2VidCondition, Vid2VidConditioner
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


# NOTE: extend the condition class to include action
@dataclass(frozen=True)
class ActionConditionedCondition(Vid2VidCondition):
    action: Optional[torch.Tensor] = None


# NOTE: extend the conditioner class to include action
class ActionConditionedConditioner(Vid2VidConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ActionConditionedCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "action" in batch, "ActionConditionalConditioner requires 'action' in batch"
        output["action"] = batch["action"]
        return ActionConditionedCondition(**output)
