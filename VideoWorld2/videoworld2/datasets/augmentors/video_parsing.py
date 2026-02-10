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
import io
import random
from typing import Optional

import decord
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms.v2 import UniformTemporalSubsample

from imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from imaginaire.utils import log


class VideoParsing(Augmentor):
    """
    This augmentor is used to parse the video bytes and get the video frames.
    the return dict is back-compatible with old datasets, which video decoding happens in the decoder stage.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 2, "VideoParsing augmentor only supports two input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]

        self.key_for_caption = args["key_for_caption"]
        assert self.key_for_caption in [
            "t2w_windows",
            "i2w_windows_later_frames",
        ], "key_for_caption must be either t2w_windows or i2w_windows_later_frames"
        self.min_duration = args["min_duration"]
        self.min_fps = args["min_fps"]
        self.max_fps = args["max_fps"]
        self.num_frames = args["num_video_frames"]
        if self.num_frames > 0:
            self.sampler = UniformTemporalSubsample(self.num_frames)
        self.video_decode_num_threads = args["video_decode_num_threads"]

    def __call__(self, data_dict: dict) -> dict:
        try:
            meta_dict = data_dict[self.meta_key]
            video = data_dict[self.video_key]
        except Exception as e:
            log.warning(
                f"Cannot find video. url: {data_dict['__url__']}, key: {data_dict['__key__']}", rank0_only=False
            )
            return None

        if not isinstance(video, bytes):
            return data_dict

        video_info = {
            "fps": meta_dict["framerate"],
            "n_orig_video_frames": meta_dict["nb_frames"],
        }

        if video_info["fps"] < self.min_fps:
            log.warning(f"Video FPS {video_info['fps']} is less than min_fps {self.min_fps}", rank0_only=False)
            return None
        if video_info["fps"] > self.max_fps:
            log.warning(f"Video FPS {video_info['fps']} is greater than max_fps {self.max_fps}", rank0_only=False)
            return None

        options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))

        # Skip the last window if possible.
        # All windows except the last are 5 seconds long. The last window has a duration in the range [2.5s, 7.5), which is less preferred.
        if len(options) > 1:
            options = options[:-1]

        # shuffle options
        random.shuffle(options)
        video_frames = None
        for chunk_index, option in options:
            start_frame = option["start_frame"]
            end_frame = option["end_frame"]
            if (end_frame - start_frame) < self.min_duration * video_info["fps"]:
                continue

            video_buffer = io.BytesIO(video)
            video_reader = decord.VideoReader(video_buffer, num_threads=self.video_decode_num_threads)

            frame_indices = np.arange(start_frame, end_frame).tolist()

            # online hot-fix for alpamayo data. Skip the first 5 frames as there is chance that the first five frames contain black frames.
            if "alpamayo" in data_dict["__url__"].root:
                assert (
                    len(frame_indices) >= 5
                ), "Getting less than 5 frames for alpamayo videos. There is no way to skip the first five frames."
                frame_indices = frame_indices[5:]
                start_frame += 5
            video_frames = video_reader.get_batch(frame_indices).asnumpy()

            video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

            # Clean up
            video_reader.seek(0)  # set video reader point back to 0 to clean up cache
            del video_reader  # delete the reader to avoid memory leak
            break

        if video_frames is None:
            log.warning(
                f"No valid video frames found, return None. url: {data_dict['__url__']}, key: {data_dict['__key__']}",
                rank0_only=False,
            )
            return None

        video_info["chunk_index"] = chunk_index
        video_info["frame_start"] = start_frame
        video_info["frame_end"] = end_frame
        video_info["num_frames"] = end_frame - start_frame
        if self.num_frames > 0:
            video_frames = rearrange(self.sampler(rearrange(video_frames, "c t h w -> t c h w")), "t c h w -> c t h w")

        video_info["video"] = video_frames

        # update data_dict, make it back-compatible with old datasets, which video decoding happens in the decoder stage.
        data_dict[self.video_key] = video_info

        return data_dict
