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
from typing import Any, List

import attrs

from videoworld2.configs.action_conditioned.defaults.data import register_training_and_val_data_action_conditioned
from videoworld2.configs.action_conditioned.defaults.model import register_model_action_conditioned
from videoworld2.configs.base.defaults.callbacks import register_callbacks
from videoworld2.configs.base.defaults.checkpoint import register_checkpoint
from videoworld2.configs.base.defaults.data import register_training_and_val_data
from videoworld2.configs.base.defaults.ema import register_ema
from videoworld2.configs.base.defaults.model import register_model
from videoworld2.configs.base.defaults.optimizer import register_optimizer
from videoworld2.configs.base.defaults.scheduler import register_scheduler
from imaginaire import config
from imaginaire.trainer import ImaginaireTrainer as Trainer
from imaginaire.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"dataloader_train": None},
            {"dataloader_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "constant"},
            {"model": "predict2_video2world_fsdp_2b"},
            {"callbacks": ["basic"]},
            {"net": None},
            {"ema": None},
            {"checkpoint": None},
            {"ckpt_type": None},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "videoworld2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 10000
    c.trainer.run_validation = True
    c.trainer.max_val_iter = 10000
    c.trainer.callbacks = None

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_model()

    register_ema()
    register_checkpoint()
    register_callbacks()

    # action conditional post-training config
    register_training_and_val_data_action_conditioned()
    register_model_action_conditioned()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("videoworld2.configs.base.experiment", reload=True)
    import_all_modules_from_package("videoworld2.configs.action_conditioned.experiment", reload=True)
    return c
