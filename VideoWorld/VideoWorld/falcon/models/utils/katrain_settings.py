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
import sys

from katrain.core.base_katrain import KaTrainBase, Player
from katrain.core.constants import *

DEFAULT_PORT = 8587


class Logger(KaTrainBase):
    def __init__(self, debug_level=0, output_level=OUTPUT_INFO):
        super().__init__(force_package_config=True, debug_level=debug_level)
        self.output_level = output_level

    def log(self, msg, level=OUTPUT_INFO):
        if level <= self.output_level:
            print(msg, file=sys.stderr)


bot_strategies = {
    "dev": (
        AI_SIMPLE_OWNERSHIP,
        {"max_points_lost": 2.0, "settled_weight": 1.0, "opponent_fac": 0.5},
        {"max_visits": 500},
    ),
    "dev-beta": (
        AI_SIMPLE_OWNERSHIP,
        {"max_points_lost": 1.75, "settled_weight": 1.0, "opponent_fac": 0.5},
        {"max_visits": 500, "wide_root_noise": 0.02},
    ),
    "strong": (
        AI_SIMPLE_OWNERSHIP,
        {"max_points_lost": 1.1, "settled_weight": 1.0, "opponent_fac": 0.5, "min_visits": 3, "attach_penalty": 1, "tenuki_penalty": 0.5},
        {"max_visits": 50,"wide_root_noise": 0.02},
    ),
    # "dev": (AI_SCORELOSS, {"strength": 0.5}, {"max_visits": 500}),
    #    "dev": (AI_WEIGHTED, {"weaken_fac": 0.5},{}),
       "balanced": (AI_SCORELOSS, {"strength": 0.35}, {"max_visits": 500}),  # 1d?
    "territory": (AI_TERRITORY, {}, {}),
    #    "dev": (AI_POLICY, {}, {}),
    "ai_policy": (AI_POLICY, {}, {}),
    "weak": (AI_DEFAULT, {}, {"max_visits": 500,'max_time':15}),
    "influence": (AI_INFLUENCE, {}, {}),
    #    "balanced": (AI_PICK, {}, {}),
    "weighted": (AI_WEIGHTED, {}, {"weaken_fac": 1.0}),
    "local": (AI_LOCAL, {}, {}),
    "tenuki": (AI_TENUKI, {}, {}),
    "18k": (AI_RANK, {"kyu_rank": 18}, {}),
    "14k": (AI_RANK, {"kyu_rank": 14}, {}),
    "10k": (AI_RANK, {"kyu_rank": 10}, {}),
    "6k": (AI_RANK, {"kyu_rank": 6}, {}),
    "2k": (AI_RANK, {"kyu_rank": 2}, {}),
    "2d": (AI_RANK, {"kyu_rank": -1}, {}),
}

engine_overrides = {"dev": {"maxVisits": 500}}

greetings = {
    "dev": "Play in a way that simplifies the game.",
    "dev-beta": "Play in a way that simplifies the game.",
    #"strong": "Play top policy move.",
    "strong": "Play simple.",
    "weak": "Utility function inversed.",
    "influence": "Play an influential style.",
    "territory": "Play a territorial style.",
    #    "balanced": "Play the best move out of a random selection.",
    "balanced": "Having a mid-life crisis and now plays in a way that complicates the game.",
    "weighted": "Play a policy-weighted move.",
    "local": "Prefer local responses.",
    "tenuki": "Prefer to tenuki.",
    "18k": "Calibrated version of katrain-balanced for ~18k",
    "14k": "Calibrated version of katrain-balanced for ~14k",
    "10k": "Calibrated version of katrain-balanced for ~10k",
    "6k": "Calibrated version of katrain-balanced for ~6k",
    "2k": "Calibrated version of katrain-balanced for ~2k",
    "2d": "Calibrated version of katrain-balanced for ~2d",
}
