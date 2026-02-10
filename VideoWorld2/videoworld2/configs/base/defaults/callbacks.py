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

from videoworld2.callbacks.device_monitor import DeviceMonitor
from videoworld2.callbacks.grad_clip import GradClip
from videoworld2.callbacks.iter_speed import IterSpeed
from videoworld2.callbacks.ldm_callback import LDMCallback
from videoworld2.callbacks.loss_log import LossLog
from imaginaire.callbacks.manual_gc import ManualGarbageCollection
from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils.callback import LowPrecisionCallback

BASIC_CALLBACKS = dict(
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    loss_log=L(LossLog)(),
    grad_clip=L(GradClip)(clip_norm=1.0),
    ldm_callback=L(LDMCallback)(every_n="${trainer.logging_iter}"),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
