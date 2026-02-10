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

from falcon.registry import TRANSFORMS
import os.path as osp

import torch
import io
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient
from mmcv.transforms import to_tensor
from typing import Dict, List, Optional, Union, Tuple
from falcon.structures import DataSample
def image_to_tensor(img):
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        Tensor: The output tensor.
    """

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img)
    tensor = to_tensor(img).permute(2, 0, 1).contiguous()

    return tensor

def can_convert_to_image(value):
    """Judge whether the input value can be converted to image tensor via
    :func:`images_to_tensor` function.

    Args:
        value (any): The input value.

    Returns:
        bool: If true, the input value can convert to image with
            :func:`images_to_tensor`, and vice versa.
    """
    if isinstance(value, (List, Tuple)):
        return all([can_convert_to_image(v) for v in value])
    elif isinstance(value, np.ndarray) and len(value.shape) > 1:
        return True
    elif isinstance(value, torch.Tensor):
        return True
    else:
        return False


def all_to_tensor(value):
    """Trans image and sequence of frames to tensor.

    Args:
        value (np.ndarray | list[np.ndarray] | Tuple[np.ndarray]):
            The original image or list of frames.

    Returns:
        Tensor: The output tensor.
    """

    if not can_convert_to_image(value):
        return value

    if isinstance(value, (List, Tuple)):
        # sequence of frames
        if len(value) == 1:
            tensor = image_to_tensor(value[0])
        else:
            frames = [image_to_tensor(v) for v in value]
            tensor = torch.stack(frames, dim=0)
    elif isinstance(value, np.ndarray):
        tensor = image_to_tensor(value)
    else:
        # Maybe the data has been converted to Tensor.
        tensor = to_tensor(value)

    return tensor


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


# @TRANSFORMS.register_module()
# class LoadImageNetFromFile(BaseTransform):

#     def __init__(self, to_float32=False, to_rgb=True, with_text=False, backend="pillow"):
#         self.to_float32 = to_float32
#         self.with_text = with_text
#         self.channel_order = 'rgb'
#         if not to_rgb:
#             self.channel_order = 'bgr'

#         self.backend = backend

#     def transform(self, results):
#         results['filename'] = osp.join(results['data_root'], results['prefix'], results['filename'])
#         filename = results['filename']
#         img = mmcv.imread(filename, channel_order="rgb", backend=self.backend)
#         if self.to_float32:
#             img = img.astype(np.float32)
#         else:
#             img = img.astype(np.uint8)

#         results['img'] = img
#         results['original_shape'] = img.shape[:2]
#         results['img_shape'] = img.shape[:2]
#         results['gt_label'] = int(results['label'])
#         if self.with_text:
#             results['text'] = "this is an image."

#         return results


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
        # import pdb;pdb.set_trace()
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
            imgs = container.get_batch(frame_inds).numpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.numpy())
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
class CenterCropLongEdgeVideo(BaseTransform):
    """Center crop the given image by the long edge.

    Args:
        keys (list[str]): The images to be cropped.
    """

    def __init__(self, keys='imgs'):
        assert keys, 'Keys should not be empty.'
        self.keys = keys

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        imgs = results[self.keys]
        img_height, img_width = imgs[0].shape[:2]
        crop_size = min(img_height, img_width)
        y1 = 0 if img_height == crop_size else \
            int(round(img_height - crop_size) / 2)
        x1 = 0 if img_width == crop_size else \
            int(round(img_width - crop_size) / 2)
        y2 = y1 + crop_size - 1
        x2 = x1 + crop_size - 1

        out_list=[]
        for img in imgs:
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            out_list.append(img)

        results[self.keys]=out_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str



@TRANSFORMS.register_module()
class ResizeVideo(BaseTransform):
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation,
            accepted values are "nearest", "bilinear", "bicubic", "area",
            "lanczos". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        """Static method for resizing keypoint."""
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str





@TRANSFORMS.register_module()
class PackVideoInputs(BaseTransform):
    """Pack data into DataSample for training, evaluation and testing.

    MMagic follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of DataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    def __init__(
        self,
        keys: Tuple[List[str], str] = ['merged', 'imgs'],
        meta_keys: Tuple[List[str], str] = [],
        data_keys: Tuple[List[str], str] = ['gt_img',],
    ) -> None:

        assert keys is not None, \
            'keys in PackInputs can not be None.'
        assert data_keys is not None, \
            'data_keys in PackInputs can not be None.'
        assert meta_keys is not None, \
            'meta_keys in PackInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys,
                                                 List) else [meta_keys]

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`DataSample`): The annotation info of the
                sample.
        """
        # import pdb;pdb.set_trace()
        # prepare inputs
        inputs = dict()
        results['gt_img']=results['imgs']
        for k in self.keys:
            value = results.get(k, None)
            if value is not None:
                inputs[k] = all_to_tensor(value)

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.values())[0]

        data_sample = DataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {
            k: v
            for (k, v) in results.items()
            if k not in (self.data_keys + self.meta_keys)
        }
        data_sample.set_predefined_data(predefined_data)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {
            k: v
            for (k, v) in results.items() if k in self.meta_keys
        }
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            k: v
            for (k, v) in results.items() if k in self.data_keys
        }
        data_sample.set_tensor_data(required_data)
        # import pdb;pdb.set_trace()
        return {'inputs': inputs, 'data_samples': data_sample}

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str
