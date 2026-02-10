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
"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import ffmpeg
import tempfile
from typing import Literal

from .subsampler import Subsampler


class ResolutionSubsampler(Subsampler):
    """
    Adjusts the resolution of the videos to the specified height and width.
    Please do not set both video_size and height/width. This will result in an error.
    If both height and width are set, scale mode output will have the specified height (ignoring width).

    Args:
        resize_mode (list[str]): List of resize modes to apply. Possible options are:
            scale: scale video keeping aspect ratios
            crop: center crop to video_size x video_size
            pad: center pad to video_size x video_size
        height (int): Height of video.
        width (int): Width of video.
        video_size (int): Both height and width.
        encode_format (str): Format to encode in (i.e. mp4)
    """

    def __init__(
        self,
        resize_mode: Literal["scale", "crop", "pad"],
        height: int = -1,
        width: int = -1,
        video_size: int = -1,
        encode_format: str = "mp4",
    ):
        if video_size > 0 and (height > 0 or width > 0):
            raise ValueError("Either set video_size, or set height and/or width")
        self.resize_mode = resize_mode
        self.height = height if video_size < 0 else video_size
        self.width = width if video_size < 0 else video_size
        self.video_size = video_size
        self.encode_format = encode_format

    def __call__(self, streams, metadata=None):
        video_bytes = streams["video"]
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                    if "scale" in self.resize_mode:
                        if self.height > 0:
                            _ = _.filter("scale", -2, self.height)
                        else:
                            _ = _.filter("scale", self.width, -2)
                    if "crop" in self.resize_mode:
                        _ = _.filter("crop", w=self.width, h=self.height)
                    if "pad" in self.resize_mode:
                        _ = _.filter("pad", w=self.width, h=self.height)
                    _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], None, str(err)

                with open(f"{tmpdir}/output.mp4", "rb") as f:
                    subsampled_bytes.append(f.read())
        streams["video"] = subsampled_bytes
        return streams, metadata, None
