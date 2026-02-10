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
from webdataset import WebLoader

from video2dataset.dataloader import get_video_dataset

# WebVid validation split
SHARDS = "dataset/{00000..00004}.tar"

if __name__ == "__main__":
    decoder_kwargs = {
        "n_frames": 8,  # get 8 frames from each video
        "fps": 10,  # downsample to 10 FPS
        "num_threads": 12,  # use 12 threads to decode the video
    }
    resize_size = crop_size = 256
    batch_size = 32

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
    )

    num_workers = 6  # 6 dataloader workers

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)

    for sample in dl:
        video_batch = sample["mp4"]
        print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])

        # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
        text_batch = sample["txt"]
        print(text_batch[0])
        metadata_batch = sample["json"]
