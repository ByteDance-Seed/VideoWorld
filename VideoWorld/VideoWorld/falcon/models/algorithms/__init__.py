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
from falcon.models.algorithms.base import BaseModel
from falcon.models.algorithms.IDM import VQInverseDynamicModel, FeatInverseDynamicModel
from falcon.models.algorithms.calvin_GR1_wostate_vq_idm import VideoWorldRobotics
from falcon.models.algorithms.go_battle_with_human import VideoWorldGoBattleVSHuman
from falcon.models.algorithms.go_battle_train_model import VideoWorldGoBattleTrainModel
from falcon.models.algorithms.go_battle_with_human_woKataGo import VideoWorldGoBattleVSHumanwoKataGo
# from falcon.models.algorithms.calvin_GR1_wostate_vq_onlyrgb import VQImageTextRGBGenCALVINGR1WOStateVQ


__all__ = [
    'BaseModel', 'VideoWorldGoBattleVSHuman', 'VideoWorldGoBattleTrainModel'
]
