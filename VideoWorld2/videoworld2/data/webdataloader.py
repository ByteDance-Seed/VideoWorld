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
import torch

from videoworld2.datasets.dataset_provider import get_image_dataset, get_video_dataset
from videoworld2.datasets.joint_dataloader import IterativeJointDataLoader
from imaginaire.datasets.webdataset.dataloader import DataLoader as _DataLoader


def get_image_dataloader(dataset_name: str):
    return _DataLoader(
        dataset=get_image_dataset(
            dataset_name=dataset_name,
            resolution="480",
            is_train=True,
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=2,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )


def get_video_dataloader(dataset_name: str):
    return _DataLoader(
        dataset=get_video_dataset(
            dataset_name=dataset_name,
            video_decoder_name="video_naive_bytes",
            augmentor_name="video_basic_augmentor_v2",
            resolution="480",
            is_train=True,
            chunk_size=256,
            embedding_type="umt5_xxl",
            num_video_frames=81,
            min_fps_thres=3,
            max_fps_thres=60,
        ),
        batch_size=1,
        num_workers=8,
        prefetch_factor=2,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )


def joint_image_video_dataloader(image_dataset_name: str, video_dataset_name: str):
    image_dataloader = get_image_dataloader(dataset_name=image_dataset_name)
    video_dataloader = get_video_dataloader(dataset_name=video_dataset_name)
    return IterativeJointDataLoader(
        dataloaders={
            "image_data": {
                "dataloader": image_dataloader,
                "ratio": 1,
            },
            "video_data": {
                "dataloader": video_dataloader,
                "ratio": 1,
            },
        }
    )


class WebDataLoader(_DataLoader):
    def __init__(
        self,
        image_dataset_name: str = None,
        video_dataset_name: str = None,
        dataloader_type: str = "joint",
    ):
        assert dataloader_type == "video", "Currently supports video-only training."

        if dataloader_type == "image":
            self.dataloader = get_image_dataloader(dataset_name=image_dataset_name)
        elif dataloader_type == "video":
            self.dataloader = get_video_dataloader(dataset_name=video_dataset_name)
        else:
            self.dataloader = joint_image_video_dataloader(
                image_dataset_name=image_dataset_name, video_dataset_name=video_dataset_name
            )
        self.dataloader_iter = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for data_dict in self.dataloader_iter:
            # Pack the data in the format we need

            out_data = dict()
            out_data["first_frame"] = data_dict["video"][:, :, 0, :, :].permute(0, 2, 3, 1)
            out_data["video"] = data_dict["video"].to(torch.float32) / 255.0
            out_data["video"] = out_data["video"] * 2 - 1
            out_data["text"] = data_dict["ai_caption"]
            out_data["path"] = data_dict["__url__"]
            out_data["prompt_emb"] = {"context": data_dict["t5_text_embeddings"]}
            yield out_data
