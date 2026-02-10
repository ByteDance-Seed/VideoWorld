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

import math
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
import torch
from falcon.registry import TRANSFORMS
# from mmdet.datasets.transforms import Rotate

@TRANSFORMS.register_module()
# class PseudoVideoRotate(Rotate):
class PseudoVideoRotate(BaseTransform):

    def transform(self, results: dict) -> dict:
        if self._random_disable():
            return results
        
        # import pdb;pdb.set_trace()
        video = results['img']
        mask_gt = results['mask_gt']
        new_video = []
        new_mask_gt = []
        for img, gt_img in zip(video, mask_gt):
            mag = self._get_mag()
            self.homography_matrix = self._get_homography_matrix(results, mag)
            self._record_homography_matrix(results)
            img = self._transform_img(results, mag, img)
            gt_img = self._transform_seg(results, mag, gt_img)

            new_video.append(img)
            new_mask_gt.append(gt_img)

        new_video = np.stack(new_video, axis = 0)
        new_mask_gt = np.stack(new_mask_gt, axis=0)
        results['img'] = new_video
        results['mask_gt'] = new_mask_gt

        return results
    def _transform_seg(self, results: dict, mag: float, mask_gt=None) -> None:
        """Rotate the segmentation map."""
        mask_gt = mmcv.imrotate(
            mask_gt,
            mag,
            border_value=0,
            interpolation='nearest')
        return mask_gt
    def _transform_img(self, results, mag: float, img=None) -> None:
        img = mmcv.imrotate(
            img,
            mag,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

        return img
@TRANSFORMS.register_module()
class MMResize(BaseTransform):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                 float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict, img_in=None) -> None:
        """Resize images with ``results['scale']``."""
        _img = img_in if img_in is not None else results.get('img', None)
        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    _img,
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = _img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    _img,
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            if img_in is not None:
                return img, (w_scale, h_scale)
            else:
                results['img'] = img
                results['img_shape'] = img.shape[:2]
                results['scale_factor'] = (w_scale, h_scale)
                results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict, mask_gt_in=None) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        _mask_gt = mask_gt_in if mask_gt_in is not None else results.get('mask_gt', None)
        if results.get('mask_gt', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    _mask_gt,
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            if mask_gt_in is None:
                results['pred_label'] = gt_seg
            else:
                return gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int | Sequence): Desired output size of the crop. If
            crop_size is an int instead of sequence like (h, w), a square crop
            (crop_size, crop_size) is made.
        padding (int | Sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (bool): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - ``constant``: Pads with a constant value, this value is specified
              with pad_val.
            - ``edge``: pads with the last value at the edge of the image.
            - ``reflect``: Pads with reflection of image without repeating the
              last value on the edge. For example, padding [1, 2, 3, 4]
              with 2 elements on both sides in reflect mode will result
              in [3, 2, 1, 2, 3, 4, 3, 2].
            - ``symmetric``: Pads with reflection of image repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with
              2 elements on both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 crop_size: Union[Sequence, int],
                 padding: Optional[Union[Sequence, int]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
        else:
            assert crop_size > 0
            self.crop_size = (crop_size, crop_size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        target_h, target_w = self.crop_size
        if w == target_w and h == target_h:
            return 0, 0, h, w
        elif w < target_w or h < target_h:
            target_w = min(w, target_w)
            target_h = min(w, target_h)

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)

        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

            img = mmcv.impad(
                img,
                padding=(w_pad, h_pad, w_pad, h_pad),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            np.array([
                offset_w,
                offset_h,
                offset_w + target_w - 1,
                offset_h + target_h - 1,
            ]))
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', padding={self.padding}'
        repr_str += f', pad_if_needed={self.pad_if_needed}'
        repr_str += f', pad_val={self.pad_val}'
        repr_str += f', padding_mode={self.padding_mode})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(BaseTransform):
    """Crop the given image to random scale and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(self,
                 scale: Union[Sequence, int],
                 crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
                aspect_ratio_range[0] > aspect_ratio_range[1]):
            raise ValueError(
                'range should be of kind (min, max). '
                f'But received crop_ratio_range {crop_ratio_range} '
                f'and aspect_ratio_range {aspect_ratio_range}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + target_w - 1,
                                    offset_h + target_h - 1
            ]))
        img = mmcv.imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)

        if results.get('mask_gt', None) is not None:
            gt_img = results['mask_gt']
            gt_img = mmcv.imcrop(
                gt_img,
                bboxes=np.array([
                    offset_w, offset_h, offset_w + target_w - 1,
                                        offset_h + target_h - 1
                ]))
            gt_img = mmcv.imresize(
                gt_img,
                tuple(self.scale[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)
            results['pred_label'] = gt_img.astype(np.float32)
        else:
            results['pred_label'] = img
        results['img'] = img
        results['img_shape'] = img.shape
        
        

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(scale={self.scale}'
        repr_str += ', crop_ratio_range='
        repr_str += f'{tuple(round(s, 4) for s in self.crop_ratio_range)}'
        repr_str += ', aspect_ratio_range='
        repr_str += f'{tuple(round(r, 4) for r in self.aspect_ratio_range)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class VideoMMResize(MMResize):
    def transform(self, results: dict) -> dict:
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'][0].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore

        video = results['img']
        mask_gt = results['mask_gt']
        new_video = []
        new_mask_gt = []
        scale_factors = []

        for img, gt_img in zip(video, mask_gt):
            img, scale_factor = self._resize_img(results, img)
            gt_img = self._resize_seg(results, gt_img)
            
            new_video.append(img)
            new_mask_gt.append(gt_img)
            scale_factors.append(scale_factor)
        
        new_video = np.stack(new_video, axis = 0)
        new_mask_gt = np.stack(new_mask_gt, axis=0)
        
        results['img'] = new_video
        results['img_shape'] = [img.shape for img in video]
        results['scale_factor'] = scale_factors
        results['pred_label'] = new_mask_gt
        results['keep_ratio'] = self.keep_ratio

        return results

@TRANSFORMS.register_module()
class VideoRandomResizedCrop(RandomResizedCrop):

    def transform(self, results: dict) -> dict:
        video = results['img']

        mask_gt = results['mask_gt']
        new_video = []
        new_mask_gt = []
        for img, gt_img in zip(video, mask_gt):
            offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
            img = mmcv.imcrop(
                img,
                bboxes=np.array([
                    offset_w, offset_h, offset_w + target_w - 1,
                                        offset_h + target_h - 1
                ]))
            img = mmcv.imresize(
                img,
                tuple(self.scale[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)

            gt_img = mmcv.imcrop(
                gt_img,
                bboxes=np.array([
                    offset_w, offset_h, offset_w + target_w - 1,
                                        offset_h + target_h - 1
                ]))
            gt_img = mmcv.imresize(
                gt_img,
                tuple(self.scale[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)

            new_video.append(img)
            new_mask_gt.append(gt_img)
        
        new_video = np.stack(new_video, axis = 0)
        new_mask_gt = np.stack(new_mask_gt, axis=0)

        results['img'] = new_video
        results['img_shape'] = [img.shape for img in video]

        results['pred_label'] = new_mask_gt

        return results

@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required
    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, rescale=True, norm_pred_label=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.rescale = rescale
        self.norm_pred_label = norm_pred_label

    def normalize(self, img):
        # import pdb;pdb.set_trace()
        # print(img.shape)
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1. / np.float64(self.std.reshape(1, -1))
        if self.to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.subtract(img, mean)
        img = cv2.multiply(img, stdinv)
        return img

    def transform(self, results):

        img = results['img']
        if self.rescale:
            img = img.astype(np.float32) / 255.0
        img = self.normalize(img)

        results['img'] = img
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)

        if self.norm_pred_label and results.get('pred_label', None) is not None:
            pred_label = results['pred_label']
            if self.rescale:
                pred_label = pred_label.astype(np.float32) / 255.0
            pred_label = self.normalize(pred_label)

            results['pred_label'] = pred_label

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_rgb={self.to_bgr}, ')
        return repr_str


@TRANSFORMS.register_module()
class VideoNormalize(Normalize):
    def transform(self, results):
        video = results['img']
        if self.rescale:
            video = video.astype(np.float32) / 255.0
        new_video = []
      
        for img in video:
            img = self.normalize(img)
            new_video.append(img)
        new_video = np.stack(new_video, axis=0)
        results['img'] = new_video
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)
        return results

@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """Randomly selects a rectangle region in an image and erase pixels.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=0.4,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
               and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand'], \
            'Please select `mode` from ["const", "rand"].'
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
               and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                   and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        """Fill pixels to the patch of image."""
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.erase_prob

    @cache_randomness
    def random_patch(self, img_h, img_w):
        """Randomly generate patch the erase."""
        # convert the aspect ratio to log space to equally handle width and
        # height.
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        return top, left, h, w

    def transform(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        if self.random_disable():
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        img = self._fill_pixels(img, *self.random_patch(img_h, img_w))

        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str
