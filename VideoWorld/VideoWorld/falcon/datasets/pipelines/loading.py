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
import random
from pycocotools import mask
import cv2
import lmdb
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from falcon.registry import TRANSFORMS
from .utils import openai_imagenet_template


class LmdbReader(object):
    def __init__(self):
        self.id_context = dict()

    def read(self, lmdb_file, image_id):
        if lmdb_file in self.id_context:
            env = self.id_context[lmdb_file].begin()
            tmp = env.get(image_id.encode())
            return tmp
        else:
            self.id_context[lmdb_file] = lmdb.open(lmdb_file, readonly=True, lock=False)
            env = self.id_context[lmdb_file].begin()
            tmp = env.get(image_id.encode())
            return tmp


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

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 copy_num=1,
                 backend='pillow'):
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
            img = mmcv.image.imfrombytes(
                results['img'],
                channel_order=self.channel_order,
                backend=self.backend)
            if self.to_float32:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.uint8)
        except Exception:
            print('skip one example')

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadImageNetFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 cls_gen=False,
                 with_text=False,
                 img_root=None,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.with_text = with_text
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.cls_gen = cls_gen
        self.img_root = img_root

    def transform(self, results):
        # import pdb;pdb.set_trace()
        if self.img_root is not None:
            results['filename'] = osp.join(self.img_root, results['prefix'],
                                           results['filename'])
        else:
            results['filename'] = osp.join(results['data_root'], results['prefix'],
                                           results['filename'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        if 'label' in results:
            results['gt_label'] = int(results['label'])
        if self.with_text:
            cls_name = results.get('cls_name', "")
            results['text'] = 'draw an image about ' + cls_name

        if self.cls_gen:
            results['task'] = "Get the category name of the image."
        results['mask_gt'] = img
        return results


@TRANSFORMS.register_module()
class LoadImageNetINalistFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 cls_gen=True,
                 with_text=True,
                 random_tmp=False,
                 img_root="./data/",
                 backend='pillow'):
        self.to_float32 = to_float32
        self.with_text = with_text
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.cls_gen = cls_gen
        self.img_root = img_root
        self.random_tmp = random_tmp
        self.template = openai_imagenet_template

    def transform(self, results):
        prefix = results.get("prefix", "")
        folder = results.get('folder', "")
        if self.img_root is not None:
            results['filename'] = osp.join(self.img_root, folder, prefix,
                                           results['filename'])
        else:
            results['filename'] = osp.join(results['data_root'], folder, prefix,
                                           results['filename'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        if 'label' in results:
            results['gt_label'] = int(results['label'])
        if self.with_text:
            cls_name = results.get('cls_name', "")

            text = 'this is an image about ' + cls_name
            if self.random_tmp:
                temp = random.choice(self.template)
                text = temp(cls_name)
            results['text'] = text

        if self.cls_gen:
            results['task'] = "Get the category name of the image."

        return results


@TRANSFORMS.register_module()
class LoadiNaturaFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 cls_gen=False,
                 with_text=False,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.with_text = with_text
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.cls_gen = cls_gen

    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        if 'category_id' in results:
            results['gt_label'] = int(results['category_id'])
        cls_name = ""
        if self.with_text:
            cls_name = results.get('name', "")
        else:
            cls_name = ""

        results['text'] = 'this is an image about ' + cls_name

        results['task'] = "Get the category name of the image."

        return results


@TRANSFORMS.register_module()
class LoadImageNetTextFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 cls_gen=False,
                 with_text=False,
                 img_root=None,
                 test_mode=False,
                 random_tmp=False,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.with_text = with_text
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.cls_gen = cls_gen
        self.img_root = img_root
        self.random_tmp = random_tmp
        self.template = openai_imagenet_template
        self.test_mode = test_mode

    def transform(self, results):
        if self.img_root is not None:
            results['filename'] = osp.join(self.img_root, results['prefix'],
                                           results['filename'])
        else:
            results['filename'] = osp.join(results['data_root'], results['prefix'],
                                           results['filename'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        if 'label' in results:
            results['gt_label'] = int(results['label'])
        if self.with_text:
            cls_name = results.get('cls_name', "")

            text = 'this is an image about ' + cls_name
            if self.random_tmp:
                temp = random.choice(self.template)
                text = temp(cls_name)

            results['text'] = text

        if self.cls_gen:
            results['task'] = "Get the category name of the image."

        return results


@TRANSFORMS.register_module()
class LoadVGTextFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend

    def transform(self, results):
        results['filename'] = osp.join(results['data_root'], results['prefix'],
                                       str(results['image_id']) + ".jpg")

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        task = results.get('task')

        if task == "dense":
            results[
                'task'] = "Describe a rectangular area located at({},{}) with a width of {} and height of {} within the {}*{} image".format(
                results['x'], results['y'], results['width'], results['height'], results['original_shape'][0],
                results['original_shape'][1])
            text = results['phrase']
        else:
            results[
                'task'] = "Get the category name of a rectangular area located at({},{}) with a width of {} and height of {} within the {}*{} image".format(
                results['x'], results['y'], results['w'], results['h'], results['original_shape'][0],
                results['original_shape'][1])
            text = 'this is an image about ' + results['names'][0]

        results['text'] = text

        return results


@TRANSFORMS.register_module()
class LoadImageTextFromlmdb(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 img_root=None,
                 use_chat=True,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.img_root = img_root
        self.reader = LmdbReader()
        self.use_chat = use_chat

    def transform(self, results):
        img_db = None
        if self.img_root is not None:
            img_db = osp.join(self.img_root, results['lmdb'])
        else:
            img_db = osp.join(results['data_root'], "image", results['lmdb'])

        filename = str(results['id'])
        img = self.reader.read(img_db, filename)
        img = mmcv.image.imfrombytes(img, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        if self.use_chat:
            results['text'] = results['chat_caption']
        else:
            results['text'] = results['caption']

        results['task'] = "Describe the image with long caption"

        return results


@TRANSFORMS.register_module()
class LoadImageClsTextFromlmdb(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 with_text=True,
                 img_root=None,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.img_root = img_root
        self.reader = LmdbReader()
        self.with_text = with_text

    def transform(self, results):
        img_db = None
        if self.img_root is not None:
            img_db = osp.join(self.img_root, results['lmdb'])
        else:
            img_db = osp.join(results['data_root'], "image", results['lmdb'])

        filename = str(results['id'])
        img = self.reader.read(img_db, filename)
        img = mmcv.image.imfrombytes(img, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        if 'label' in results:
            results['gt_label'] = int(results['labels'])

        cls_name = ""
        if self.with_text:
            cls_name = results.get('cls_name', "")
        else:
            cls_name = ""

        results['text'] = 'this is an image about ' + cls_name

        results['task'] = "Get the category name of the image."

        return results


@TRANSFORMS.register_module()
class LoadCOCOFromFile(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend

    def transform(self, results):

        ids = results['file_name'].split("_")[-1].split(".jpg")[0]

        results['filename'] = osp.join(results['data_root'], results['prefix'],
                                       results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        results['text'] = results['caption']

        results['task'] = "describe the image: "
        results['image_id'] = int(ids)

        return results


@TRANSFORMS.register_module()
class LoadCOCOMask(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="",
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix

    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], self.prefix,
                                       results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadLlavaPre(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="images",
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix

    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], self.prefix,
                                       results['image'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        gt = np.ones(results['img_shape'], dtype=np.float32)
        gt = np.stack([gt, gt, gt], axis=-1)
        results['mask_gt'] = gt * 255
        results['prompt'] = None

        return results

@TRANSFORMS.register_module()
class LoadVISMask(BaseTransform):
    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="",
                 backend='pillow',
                 frame_range=10,
                 frame_sample_num=3,
                 test_mode=False):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix
        self.frame_range = [-frame_range, frame_range]
        self.frame_sample_num = frame_sample_num
        self.test_mode = test_mode
    def transform(self, results):
        # import pdb;pdb.set_trace()
        valid_frame_names = results['valid_frame_names']
        video_length = len(valid_frame_names)
        frame_ids = list(range(0, video_length))
        if self.test_mode:
            sample_frame_ids = frame_ids
        else:
            key_frame_id = random.sample(list(range(video_length)), 1)[0]
            left = max(0, key_frame_id + self.frame_range[0])
            right = min(key_frame_id + self.frame_range[1], video_length - 1)
        

            valid_ids = frame_ids[left:right + 1]
            # if key_frame_id not in valid_ids:
            #     valid_ids.append(key_frame_id)
            assert len(
                valid_ids
            ) > 0, 'After filtering key frame, there are no valid frames'
            # if len(valid_ids) < self.num_ref_imgs:
            #     valid_ids = valid_ids * self.num_ref_imgs
            sample_frame_ids = random.sample(valid_ids, self.frame_sample_num)
            sample_frame_ids = sorted(sample_frame_ids)

        data_root = results['data_root']
        img_root = osp.join(data_root, results['data_prefix']['img_path'])
        results['sampled_frame_names'] = [osp.join(img_root, valid_frame_names[i]) for i in sample_frame_ids]
        results['bboxes'] = [results['bboxes'][i] for i in sample_frame_ids]
        results['segmentations'] = [results['segmentations'][i] for i in sample_frame_ids]

        frames = []
        masks = []
        for filename, bbox, seg in zip(results['sampled_frame_names'], results['bboxes'], results['segmentations']):
            img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
            if self.to_float32:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.uint8)

            if len(seg) == 0:
                m = np.zeros(
                    (results["height"], results["width"])
                ).astype(np.uint8)
            else:
                if type(seg["counts"]) == list:  # polygon
                    rle = mask.frPyObjects(
                        seg,
                        results["height"],
                        results["width"],
                    )
                else:
                    rle = seg
                    # for i in range(len(rle)):
                    if not isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].encode()
                m = mask.decode(rle)
                # m = np.sum(
                #     m, axis=2
                # )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8

            frames.append(img)
            masks.append(m)

        frames = np.stack(frames, axis=0)
        masks = np.stack(masks, axis=0)
        results['img'] = frames
        results['mask_gt'] = masks
        results['original_shape'] = (results["height"], results["width"])
        results['img_shape'] = (results["height"], results["width"])


        return results
@TRANSFORMS.register_module()
class LoadADE20KPseudoVideoMask(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="",
                 backend='pillow',
                 frame_sample_num=3,
                 rotation_degree=10):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix
        self.frame_sample_num = frame_sample_num
    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], self.prefix,
                                       results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        mask_name = osp.join(results['data_root'], results['ann_name'])
        mask = mmcv.imread(mask_name, channel_order='rgb', backend=self.backend)[:, :, 0]
        class_id = results['clsid_list'][0]
        mask = np.where(mask == class_id, 1, 0)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        frames = np.expand_dims(img, axis=0)
        frames = np.repeat(frames, self.frame_sample_num, axis=0)

        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, self.frame_sample_num, axis=0)

        results['img'] = frames
        results['mask_gt'] = mask
        

        return results


@TRANSFORMS.register_module()
class LoadADE20KMask(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="",
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix

    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], self.prefix,
                                       results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        mask_name = osp.join(results['data_root'], results['ann_name'])
        mask = mmcv.imread(mask_name, channel_order='rgb', backend=self.backend)[:, :, 0]
        class_id = results['clsid_list'][0]
        mask = np.where(mask == class_id, 1, 0)
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['mask_gt'] = mask

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results


@TRANSFORMS.register_module()
class LoadNYUDepthMask(BaseTransform):

    def __init__(self,
                 to_float32=False,
                 to_rgb=True,
                 prefix="",
                 align_ratio=100000 / 255.0,
                 backend='pillow'):
        self.to_float32 = to_float32
        self.channel_order = 'rgb'
        if not to_rgb:
            self.channel_order = 'bgr'

        self.backend = backend
        self.prefix = prefix

        self.align_ratio = align_ratio

    def transform(self, results):

        results['filename'] = osp.join(results['data_root'], self.prefix,
                                       results['file_name'])

        filename = results['filename']
        img = mmcv.imread(filename, channel_order='rgb', backend=self.backend)
        mask_name = osp.join(results['data_root'], results['ann_name'])
        # mask = mmcv.imread(mask_name, channel_order='rgb', backend=self.backend)[:, :, 0]
        with open(mask_name, 'rb') as f:
            mask = np.frombuffer(f.read(), dtype=np.uint8)
            mask = cv2.imdecode(mask, cv2.IMREAD_UNCHANGED)
        cls_name = results['cls_name']
        mask = mask / self.align_ratio
        if self.to_float32:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.uint8)

        results['mask_gt'] = mask

        results['img'] = img
        results['original_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]

        return results
