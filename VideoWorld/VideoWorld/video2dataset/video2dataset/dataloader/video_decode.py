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
"""Video Decoders"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import decord
import tempfile
from typing import Iterable

from webdataset.autodecode import decoders


decord.bridge.set_bridge("torch")


class PRNGMixin:
    """
    Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing.
    """

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class AbstractVideoDecoder(PRNGMixin):
    def get_frames(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} is abstract")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} is abstract")


class VideoDecorder(AbstractVideoDecoder):
    """Basic video decoder that uses decord"""

    def __init__(
        self,
        n_frames=None,
        uniformly_sample=None,
        fps=None,
        num_threads=4,
        tmpdir="/tmp/",
        min_fps=1,
        max_fps=32,
        return_bytes=False,
        pad_frames=False,
        test_mode=False
    ):
        super().__init__()
        self.n_frames = n_frames
        self.test_mode = test_mode
        self.pad_frames = pad_frames
        if fps is not None and not isinstance(fps, Iterable):
            fps = [
                fps,
            ]
        if uniformly_sample:
            assert fps is None, "fps not compatible with uniformly_sample..."
        self.uniformly_sample = uniformly_sample
        self.fps = fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.return_bytes = return_bytes
        # for frame rate conditioning
        if self.fps == "sample":
            # this means we sample fps in every iteration
            self.fs_ids = {fr: i for i, fr in enumerate(range(self.min_fps, self.max_fps + 1))}
        elif isinstance(self.fps, list):
            self.fs_ids = {fr: i for i, fr in enumerate(self.fps)}
        else:
            self.fs_ids = None
            assert self.fps is None, f"invalid fps value specified: {self.fps}...."

        self.num_threads = num_threads

        infostring2 = ""
        if self.fps == "sample":
            infostring1 = "a randomly sampled"
            infostring2 = f" in between [{self.min_fps},{self.max_fps}]"
        elif isinstance(self.fps, list):
            infostring1 = ",".join([str(f) for f in self.fps])
        else:
            infostring1 = "native"
        info = (
            f'Decoding video clips of length {self.n_frames} with "decord".'
            + f" Subsampling clips to {infostring1} fps {infostring2}"
            + self.pad_frames * "Padding videos that are too short"
        )

        print(info)

        self.tmpdir = tmpdir
        print(f"Setting {self.tmpdir} as temporary directory for the decoder")

    def get_frames(self, reader, n_frames, stride, **kwargs):  # pylint: disable=arguments-differ
        # import pdb;pdb.set_trace()
        # print(n_frames, stride, len(reader))
        if n_frames * stride > len(reader):
            if not self.pad_frames:
                raise ValueError("video clip not long enough for decoding")
            n_frames = len(reader) // stride
            # print("n_frame:", n_frames)
        # sample frame start and choose scene
        if n_frames == len(reader) or n_frames == len(reader) // stride:
            frame_start = 0
        else:
            frame_start = self.prng.choice(int(len(reader)) - int(n_frames * stride), 1).item()
        frame_start = 0 if self.test_mode else frame_start
        # print(self.test_mode, frame_start)
        # only decode the frames which are actually needed
        frames = reader.get_batch(np.arange(frame_start, frame_start + n_frames * stride, stride).tolist())

        # TODO: maybe its useful to inform the user which frmaes are padded
        # can just output first_pad_index or a mask or something
        pad_start = len(frames)
        if self.pad_frames and frames.shape[0] < self.n_frames:
            frames = F.pad(frames, (0, 0) * 3 + (0, self.n_frames - frames.shape[0]))  # pylint: disable=not-callable

        return frames, frame_start, pad_start

    def __call__(self, key, data, scene_list=None):  # pylint: disable=arguments-differ
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            return None
        # import pdb;pdb.set_trace()
        additional_info = {}
        # read_fail = False
        # import pdb;pdb.set_trace()
        repeat_time = 3
        with tempfile.TemporaryDirectory(dir=self.tmpdir) as dirname:
            fname = os.path.join(dirname, f"file.{extension}")
            try_num = 1
            with open(fname, "wb") as stream:
                stream.write(data)
            while(True):
                try:
                    reader = decord.VideoReader(fname, num_threads=self.num_threads)
                    break
                except:
                    print("_________Error when decode video___________, try time {}".format(try_num))
                    if try_num < repeat_time:
                        try_num += 1
                        continue
                    # read_fail = True
                    frames = [torch.zeros(224, 224, 3).numpy() for _ in range(self.n_frames)]
                    infos = {'pad_masks':torch.zeros(self.n_frames), 'native_fps': 1, 'start_frame': 0, 'original_height': torch.ones(self.n_frames)*224, 'original_width': torch.ones(self.n_frames)*224, 'fps_id':torch.zeros(self.n_frames)}
                    return (frames, infos)

        native_fps = int(np.round(reader.get_avg_fps()))
        # if native_fps < 8:
        #     import pdb;pdb.set_trace()
        if isinstance(self.fps, list):
            fps_choices = list(filter(lambda x: x <= native_fps, self.fps))
            if not fps_choices:
                fps_choices = self.fps
                # return None
            chosen_fps = self.prng.choice(list(fps_choices), 1).item()

        elif self.fps == "sample":
            if native_fps < self.min_fps:
                return None
            max_fps = min(native_fps, self.max_fps)
            additional_info.update({"fps_id": 0})
            chosen_fps = self.prng.choice(np.arange(self.min_fps, max_fps + 1), 1).item()
        else:
            chosen_fps = native_fps
        
        fs_id = self.fs_ids[chosen_fps] if self.fs_ids else 0
        stride = max(int(np.round(native_fps / chosen_fps)), 1)
        max_stride = max(len(reader) // self.n_frames, 1)
        stride = min(stride, max_stride)
        if self.n_frames is None:
            n_frames = len(reader) // stride
        else:
            n_frames = self.n_frames

        if self.uniformly_sample:
            additional_info.update({"fps_id": torch.Tensor([fs_id] * self.n_frames).long()})
            t = len(reader)
            indices = np.linspace(0, t - 1, self.n_frames)
            indices = np.clip(indices, 0, t - 1).astype(int)
            frames, start_frame, pad_start = reader.get_batch(indices), indices[0], len(indices)
        else:
            additional_info.update({"fps_id": torch.Tensor([fs_id] * n_frames).long()})
            frames, start_frame, pad_start = self.get_frames(reader, n_frames, stride, scene_list=scene_list)
        frames = frames.float().numpy()

        additional_info["original_height"] = torch.full((frames.shape[0],), fill_value=frames.shape[1]).long()
        additional_info["original_width"] = torch.full((frames.shape[0],), fill_value=frames.shape[2]).long()

        pad_masks = torch.zeros((frames.shape[0],))
        pad_masks[:pad_start] = 1.0
        additional_info["pad_masks"] = pad_masks

        if self.n_frames is not None and frames.shape[0] < self.n_frames:
            raise ValueError("Decoded video not long enough, skipping")

        # return compatible with torchvisioin API
        additional_info.update({"native_fps": chosen_fps if chosen_fps is not None else native_fps})
        additional_info.update({"start_frame": start_frame})
        # import pdb;pdb.set_trace()
        if self.return_bytes:
            additional_info.update({"video_bytes": data})
        out = (list(frames), additional_info)
        return out


class VideoDecorderWithCutDetection(VideoDecorder):
    """Video decoder that uses decord with cut detection"""

    def __init__(self, *args, cuts_key="npy", **kwargs):
        super().__init__(*args, **kwargs)
        self.cuts_key = cuts_key
        if self.cuts_key not in decoders:
            raise KeyError(
                f"{self.__class__.__name__} received {self.cuts_key} as cuts_key,"
                + " but that one is no decoder known to webdataset"
            )

    def __call__(self, key, data):  # pylint: disable=arguments-differ,signature-differs
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            return None

        cut_list = decoders[self.cuts_key](data[self.cuts_key])
        data = data[extension]

        return super().__call__(key, data, scene_list=cut_list)

    def get_frames(self, reader, n_frames, stride, scene_list):  # pylint: disable=arguments-differ
        min_len = n_frames * stride
        # filter out subclips shorther than minimal required length
        scene_list = list(filter(lambda x: x[1] - x[0] > min_len, scene_list))
        if len(scene_list) == 0:
            raise ValueError("video clips not long enough for decoding")

        clip_id = self.prng.choice(len(scene_list), 1).item()
        start, end = scene_list[clip_id].tolist()
        # sample frame start and choose scene
        frame_start = self.prng.choice(int(end - start) - int(n_frames * stride), 1).item() + start
        # only decode the frames which are actually needed
        frames = reader.get_batch(np.arange(frame_start, frame_start + n_frames * stride, stride).tolist())

        return frames, frame_start
