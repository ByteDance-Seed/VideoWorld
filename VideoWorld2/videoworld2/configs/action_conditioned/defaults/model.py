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

from videoworld2.configs.action_conditioned.config_action_conditioned import (
    ACTION_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B, 
    LDM_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B
)
from videoworld2.models.action_video2world_model import ActionConditionedPredict2Video2WorldModel
from videoworld2.models.video2world_model import Predict2ModelManagerConfig, Predict2Video2WorldModelConfig
from imaginaire.lazy_config import LazyCall as L
from videoworld2.configs.base.config_ldm import LDM_DEFAULT, COSMOS_TOKENIZER

ACTION_CONDITIONED_PREDICT2_V2W_2B_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionConditionedPredict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=ACTION_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=-1,
        ),
        _recursive_=False,
    ),
)

ACTION_CONDITIONED_PREDICT2_V2W_2B_FSDP_LDM_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ActionConditionedPredict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=LDM_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
            ldm_config=LDM_DEFAULT,
        ),
        _recursive_=False,
    ),
)


def register_model_action_conditioned() -> None:
    cs = ConfigStore.instance()
    # predict2 v2w 2b model
    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_predict2_v2w_2b_fsdp",
        node=ACTION_CONDITIONED_PREDICT2_V2W_2B_FSDP_CONFIG,
    )

    cs.store(
        group="model",
        package="_global_",
        name="action_conditioned_predict2_v2w_2b_ldm_fsdp",
        node=ACTION_CONDITIONED_PREDICT2_V2W_2B_FSDP_LDM_CONFIG,
    )
