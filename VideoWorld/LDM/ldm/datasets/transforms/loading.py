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
import os.path as osp


import io
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from typing import Dict, List, Optional, Union

from ldm.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TransferData(BaseTransform):

    def __init__(self):
        pass

    def transform(self, data):
        jpg = data[0]
        txt = data[1]
        results = dict()
        results['img'] = jpg
        results['text'] = txt.decode()
        return results


@TRANSFORMS.register_module()
class ImageReader(BaseTransform):

    def __init__(self, to_float32=False, to_rgb=True, copy_num=1, backend="pillow"):
        self.to_float32 = to_float32
        if to_rgb:
            self.channel_order = 'rgb'
        else:
            self.channel_order = 'bgr'

        self.copy_num = copy_num
        self.backend = backend

    def transform(self, results):
        img = None
        try:
            img = mmcv.image.imfrombytes(results['img'], channel_order=self.channel_order, backend=self.backend)
            if self.to_float32:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.uint8)
        except Exception:
            print("skip one example")

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadImageNetFromFile(BaseTransform):

    def __init__(self, to_float32=False, to_rgb=True, with_text=False, backend="pillow"):
        self.to_float32 = to_float32
        self.with_text = with_text
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend

    def transform(self, results):
        results['filename'] = osp.join(results['data_root'], results['prefix'], results['filename'])
        filename = results['filename']
        try:
            img = mmcv.imread(filename, channel_order="rgb", backend=self.backend)
        except:
            print("Error: loading image")
            img = mmcv.imread('/opt/tiger/mmagicinit/ldm/data/go_dataset_size9/empty_board.png', channel_order="rgb", backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        results['gt_label'] = int(results['label'])
        if self.with_text:
            results['text'] = "this is an image."

        return results


@TRANSFORMS.register_module()
class LoadImageListFromFile(BaseTransform):

    def __init__(self, to_float32=False, to_rgb=True, backend="pillow"):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend

    def transform(self, results):
        imgs = list()
        if len(results['filename'])>0:
            for img_name in results['filename']:
                file_name = osp.join(results['data_root'], results['prefix'], img_name)
                img = mmcv.imread(file_name, channel_order="rgb", backend=self.backend)
                if self.to_float32:
                    img = img.astype(np.float32)
                else:
                    img = img.astype(np.uint8)
                imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@TRANSFORMS.register_module()
class DecordInit(BaseTransform):
    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def _get_video_reader(self,filename):
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results):
        """
        Perform the Decord initialization.
        :param results:
        :return:
        """
        container = self._get_video_reader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@TRANSFORMS.register_module()
class DecordDecode(BaseTransform):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - video_reader
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets.
            Defaults to ``'accurate'``.
    """

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class SampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self,num_frames: int,ori_clip_len: float):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps

        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results.get('start_index',0)
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class UniformSample(BaseTransform):
    """Uniformly sample frames from the video.

    Modified from https://github.com/facebookresearch/SlowFast/blob/64a
    bcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159.

    To sample an n-frame clip from the video. UniformSample basically
    divides the video into n segments of equal length and randomly samples one
    frame from each segment.

    Required keys:

        - total_frames
        - start_index

    Added keys:

        - frame_inds
        - clip_len
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False) -> None:

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def _get_sample_clips(self, num_frames: int) -> np.ndarray:
        """To sample an n-frame clip from the video. UniformSample basically
        divides the video into n segments of equal length and randomly samples
        one frame from each segment. When the duration of video frames is
        shorter than the desired length of the target clip, this approach will
        duplicate the sampled frame instead of looping the sample in "loop"
        mode. In the test mode, when we need to sample multiple clips,
        specifically 'n' clips, this method will further divide the segments
        based on the number of clips to be sampled. The 'i-th' clip will.

        sample the frame located at the position 'i * len(segment) / n'
        within the segment.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            seq (np.ndarray): the indexes of frames of sampled from the video.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        inds = []
        if not self.test_mode:
            for i in range(self.clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                inds.append(np.random.randint(start, end + 1))
        else:
            duration = seg_size / (self.num_clips + 1)
            for k in range(self.num_clips):
                for i in range(self.clip_len):
                    start = int(np.round(seg_size * i))
                    frame_index = start + int(duration * (k + 1))
                    inds.append(frame_index)

        return np.array(inds)

    def transform(self, results: Dict) -> Dict:
        """Perform the Uniform Sampling.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        inds = self._get_sample_clips(num_frames)
        start_index = results.get('start_index',0)
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}')
        return repr_str

