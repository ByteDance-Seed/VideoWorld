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

from videoworld2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from videoworld2.datasets.data_sources.mock_data import get_image_dataset, get_video_dataset
from videoworld2.datasets.joint_dataloader import IterativeJointDataLoader
from imaginaire.lazy_config import LazyCall as L

_IMAGE_LOADER = L(get_cached_replay_dataloader)(
    dataset=L(get_image_dataset)(
        resolution="480",
    ),
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="image_dataloader",
)

_VIDEO_LOADER = L(get_cached_replay_dataloader)(
    dataset=L(get_video_dataset)(
        resolution="480",
        num_video_frames=93,  # number of pixel frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="video_dataloader",
)

MOCK_DATA_INTERLEAVE_CONFIG = L(IterativeJointDataLoader)(
    dataloaders={
        "image_data": {
            "dataloader": _IMAGE_LOADER,
            "ratio": 1,
        },
        "video_data": {
            "dataloader": _VIDEO_LOADER,
            "ratio": 1,
        },
    }
)

MOCK_DATA_IMAGE_ONLY_CONFIG = _IMAGE_LOADER

MOCK_DATA_VIDEO_ONLY_CONFIG = _VIDEO_LOADER


def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="dataloader_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="dataloader_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="dataloader_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="dataloader_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
