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
frame subsampler adjusts the fps of the videos to some constant value
"""

import tempfile
import os
import copy
import ffmpeg

from .subsampler import Subsampler
from .clipping_subsampler import _get_seconds


class FrameSubsampler(Subsampler):
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Subsamples the frames of the video in terms of spacing and quantity (frame_rate, which ones etc.)
    Args:
        frame_rate (int): Target frame rate of the videos.
        downsample_method (str): determiens how to downsample the video frames:
            fps: decreases the framerate but sample remains a valid video
            first_frame: only use the first frame of a video of a video and output as image
            yt_subtitle: temporary special case where you want a frame at the beginning of each yt_subtitle
                         we will want to turn this into something like frame_timestamps and introduce
                         this as a fusing option with clipping_subsampler
        encode_format (str): Format to encode in (i.e. mp4)

    TODO: n_frame
    TODO: generalize interface, should be like (frame_rate, n_frames, sampler, output_format)
    # frame_rate - spacing
    # n_frames - quantity
    # sampler - from start, end, center out
    # output_format - save as video, or images
    """

    def __init__(self, frame_rate, downsample_method="fps", encode_format="mp4"):
        self.frame_rate = frame_rate
        self.downsample_method = downsample_method
        self.output_modality = "video" if downsample_method == "fps" else "jpg"
        self.encode_formats = {"video": encode_format}

    def __call__(self, streams, metadata=None):
        video_bytes = streams["video"]
        subsampled_bytes, subsampled_metas = [], []
        for i, vid_bytes in enumerate(video_bytes):
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    ext = "mp4"
                    if self.downsample_method == "fps":
                        _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                        _ = _.filter("fps", fps=self.frame_rate)
                        _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                    elif self.downsample_method == "keyframe":
                        _ = ffmpeg.input(f"{tmpdir}/input.mp4", discard="nokey")
                        # _ = _.filter("select", "eq(pict_type,I)")
                        _ = _.output(f"{tmpdir}/output.mp4", **{"c:s": "copy", "c": "copy", "copyts": None})
                        _ = _.run(capture_stdout=True, quiet=True)
                    elif "frame" in self.downsample_method:
                        _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                        _ = _.filter("select", "eq(n,0)")
                        _ = _.output(f"{tmpdir}/output.jpg").run(capture_stdout=True, quiet=True)
                        ext = "jpg"
                    elif self.downsample_method == "yt_subtitle":
                        subtitles = metadata[i]["yt_meta_dict"]["subtitles"]
                        starts = [_get_seconds(s["start"]) for s in subtitles]

                        for frame_id, start_t in enumerate(starts):
                            frame_key = f"{frame_id:04d}"
                            meta_frame = copy.deepcopy(metadata[i])

                            meta_frame["frame_time"] = subtitles[frame_id]["start"]
                            meta_frame["frame_subtitle"] = subtitles[frame_id]["lines"]
                            meta_frame["key"] = f"{meta_frame['key']}_{frame_key}"

                            _ = ffmpeg.input(f"{tmpdir}/input.mp4", ss=start_t)
                            _ = _.output(f"{tmpdir}/frame_{frame_id}.jpg", vframes=1, **{"q:v": 2}).run(
                                capture_stdout=True, quiet=True
                            )
                            with open(f"{tmpdir}/frame_{frame_id}.jpg", "rb") as f:
                                subsampled_bytes.append(f.read())
                            subsampled_metas.append(meta_frame)

                except Exception as err:  # pylint: disable=broad-except
                    return [], None, str(err)

                if self.downsample_method != "yt_subtitle":
                    with open(f"{tmpdir}/output.{ext}", "rb") as f:
                        subsampled_bytes.append(f.read())
                else:
                    metadata = subsampled_metas

        streams[self.output_modality] = subsampled_bytes
        return streams, metadata, None
