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
from copy import deepcopy
from typing import  Union,List, Dict

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.utils import  is_tuple_of

from ldm.registry import TRANSFORMS


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

@TRANSFORMS.register_module()
class CenterCropLongEdge(BaseTransform):
    """Center crop the given image by the long edge.

    Args:
        keys (list[str]): The images to be cropped.
    """

    def __init__(self, keys='img'):
        assert keys, 'Keys should not be empty.'
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            img = results[key]
            img_height, img_width = img.shape[:2]
            crop_size = min(img_height, img_width)
            y1 = 0 if img_height == crop_size else \
                int(round(img_height - crop_size) / 2)
            x1 = 0 if img_width == crop_size else \
                int(round(img_width - crop_size) / 2)
            y2 = y1 + crop_size - 1
            x2 = x1 + crop_size - 1

            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            results[key] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str



@TRANSFORMS.register_module()
class Resize(BaseTransform):
    """Resize data to a specific size for training or resize the images to fit
    the network input regulation for testing.

    When used for resizing images to fit network input regulation, the case is
    that a network may have several downsample and then upsample operation,
    then the input height and width should be divisible by the downsample
    factor of the network.
    For example, the network would downsample the input for 5 times with
    stride 2, then the downsample factor is 2^5 = 32 and the height
    and width should be divisible by 32.

    Required keys are the keys in attribute "keys", added or modified keys are
    "keep_ratio", "scale_factor", "interpolation" and the
    keys in attribute "keys".

    Required Keys:

    - Required keys are the keys in attribute "keys"

    Modified Keys:

    - Modified the keys in attribute "keys" or save as new key ([OUT_KEY])

    Added Keys:

    - [OUT_KEY]_shape
    - keep_ratio
    - scale_factor
    - interpolation

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (str | list[str]): The image(s) to be resized.
        scale (float | tuple[int]): If scale is tuple[int], target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size.
            Note that when it is used, `size_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used together with `scale`.
        size_factor (int): Let the output shape be a multiple of size_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used together with `size_factor`.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used.
            Default: None.
        output_keys (list[str] | None): The resized images. Default: None
            Note that if it is not `None`, its length should be equal to keys.
    """

    def __init__(self,
                 keys: Union[str, List[str]] = 'img',
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear',
                 backend=None,
                 output_keys=None):

        assert keys, 'Keys should not be empty.'
        keys = [keys] if not isinstance(keys, list) else keys
        if output_keys:
            assert len(output_keys) == len(keys)
        else:
            output_keys = keys
        if size_factor:
            assert scale is None, ('When size_factor is used, scale should ',
                                   f'be None. But received {scale}.')
            assert keep_ratio is False, ('When size_factor is used, '
                                         'keep_ratio should be False.')
        if max_size:
            assert size_factor is not None, (
                'When max_size is used, '
                f'size_factor should also be set. But received {size_factor}.')
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif is_tuple_of(scale, int):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        elif scale is not None:
            raise TypeError(
                f'Scale must be None, float or tuple of int, but got '
                f'{type(scale)}.')

        self.keys = keys
        self.output_keys = output_keys
        self.scale = scale
        self.size_factor = size_factor
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def _resize(self, img):
        """Resize function.

        Args:
            img (np.ndarray): Image.

        Returns:
            img (np.ndarray): Resized image.
        """
        if isinstance(img, list):
            for i, image in enumerate(img):
                size, img[i] = self._resize(image)
            return size, img
        else:
            if self.keep_ratio:
                img, self.scale_factor = mmcv.imrescale(
                    img,
                    self.scale,
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    img,
                    self.scale,
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                self.scale_factor = np.array((w_scale, h_scale),
                                             dtype=np.float32)

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            return img.shape, img

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.size_factor:
            h, w = results[self.keys[0]].shape[:2]
            new_h = h - (h % self.size_factor)
            new_w = w - (w % self.size_factor)
            if self.max_size:
                new_h = min(self.max_size - (self.max_size % self.size_factor),
                            new_h)
                new_w = min(self.max_size - (self.max_size % self.size_factor),
                            new_w)
            self.scale = (new_w, new_h)

        for key, out_key in zip(self.keys, self.output_keys):
            if key in results:
                size, results[out_key] = self._resize(results[key])
                results[f'{out_key}_shape'] = size
                # copy metainfo
                if f'ori_{key}_shape' in results:
                    results[f'ori_{out_key}_shape'] = deepcopy(
                        results[f'ori_{key}_shape'])
                if f'{key}_channel_order' in results:
                    results[f'{out_key}_channel_order'] = deepcopy(
                        results[f'{key}_channel_order'])
                if f'{key}_color_type' in results:
                    results[f'{out_key}_color_type'] = deepcopy(
                        results[f'{key}_color_type'])

        results['scale_factor'] = self.scale_factor
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = self.interpolation
        results['backend'] = self.backend

        return results





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



