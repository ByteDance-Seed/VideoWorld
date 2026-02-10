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
import random

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from falcon.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MaskPromptSelect(BaseTransform):

    def __init__(self, test_mode=False, only_mask=True, random_color=True):
        self.test_mode = test_mode
        self.only_mask = only_mask
        self.random_color = random_color
        if not self.test_mode:
            self.prompt_list = [
                "Segment the {}.",
                "Segment {}.",
                "Perform image segmentation to extract the {} from the image.",
                "Separate the {} from the rest of the image using image segmentation.",
                "Use computer vision techniques to segment the {} from the image.",
                "Employ image segmentation to isolate the {} in the image.",
                "Apply segmentation algorithms to extract the {} from the picture.",
                "Use computer vision to partition the image and focus on the {}.",
                "Utilize image segmentation to extract the {}'s features from the image.",
                "Segment the {} using image processing techniques.",
                "Perform object segmentation to isolate the {} in the image.",
                "Employ image segmentation to detect and extract the {} from the image.",
                "Use computer vision to accurately segment the {} in the picture.",
                "Extract the {} from the image by performing image segmentation.",
                "Use image segmentation to locate and extract the {} from the image.",
                "Partition the image using computer vision techniques to highlight the {}.",
                "Employ segmentation methods to detect and extract the {} from the image.",
                "Use image segmentation algorithms to identify and extract the {}.",
                "Apply computer vision techniques to accurately segment the {} in the image.",
                "Separate the {} from the background using image segmentation.",
                "Use segmentation algorithms to distinguish the {} from the rest of the image.",
                "Use image segmentation to identify the boundaries of the {} in the image.",
                "Split the {} in this picture.",
                "Break down {} for the image in its entirety.",
                "Segment the {} in this image with the help of segmentation's algorithm.",
                "Segment the {} in this image using segmentation's algorithm.",
                "Utilize segmentation's algorithm to split the {} in this visual representation.",
                "Use segmentation to split the {} in this graph.",
                "Segment the {} in this image using segmentation's algorithm. How can you do it?",
                "The {} in this image can be segmented using segmentation's algorithm. How does it work?",
                "segmentation algorithm for {} in this image.",
                "Use segmentation to split the {} in this graph.",
                "Segment the {} in this image using segmentation's algorithm.",
                "Using Segmentation, slice the {} in this image.",
                "Divide by {} in this image using segmentation's algorithm.",
                "To split this figure into segments, use the Segmentation algorithm to divide the {} invert.",
                "Sorting by division using segmentation's technique, specifically the {} in this diagram and its related components.",
                "Slice through the {} range on this graphic with segments using Segmentions technique.",
                "Can you assist me in segmenting the {} in this image using segmentation?",
                "I need help with segmenting the {} in this image using segmentation techniques. Can you lend a hand?",
                "Would it be possible for you to aid me with segmenting the {} in this image using segmentation?",
                "Can you assist me in segmenting the {} in this image using segmentation? I need help.",
                "I need help with segmenting the {} in this image using segmentation techniques. Can you lend a hand? Thank you.",
                "Would it be possible for you to aid me with segmenting the {} in this image using segmentation? I need help.",
                "Please aid me by assisting me in performing segmentation tasks to determine the segmented {} in this image.",
                "In this image, I am attempting to perform segmentation on the {} and would appreciate your help.",
                "I need help with segmentsation of the given {} in this image.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image. Could you lend me a hand?",
                "Can you show me how to use the Segmentation tool to divide the {} in this image? Would you be able to assist me with this task?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image? I could use some assistance.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "I require your assistance with utilizing the Segmentations tool to break down the {} in this image.",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image. Could you lend me a hand?",
                "Can you show me how to use the Segmentation tool to divide the {} in this image? Would you be able to assist me with this task?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image? I could use some assistance.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "I require your assistance with utilizing the Segmentations tool to break down the {} in this image.",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image.",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image?",
                "Can you assist me with dividing the specified {} in this image?",
                "I am seeking assistance with the segmentation of the specified {} in this image.",
                "Could you help me separate the indicated {} in this image?",
                "I am in need of help with dividing the specified {} in this picture.",
                "Can you lend a hand in segmenting the given {} within this image?",
                "I require assistance with splitting the indicated {} in this picture.",
                "Could you help me with the segmentation of the specified {} in this photograph?",
                "I am seeking guidance on dividing the given {} within this image.",
                "Can you provide assistance with segmenting the indicated {} in this image?",
                "It would be great if you could assist me with the segmentation of the {} in this image.",
                "I'm trying to segment the {} in this image and would be grateful for your aid.",
                "Can you lend me a hand with segmenting the {} in this image?",
                "My goal is to perform segmentation on the {} in this image and I would appreciate any help you can provide.",
                "I could use your help in segmenting the {} in this image.",
                "I'm attempting to perform segmentation on the {} in this image and would welcome your assistance.",
                "Would you be willing to help me with the segmentation of the {} in this image?",
                "If you could lend me your expertise, I would greatly appreciate your help with segmenting the {} in this image.",
                "My aim is to segment the {} in this image and I would be thankful for your help.",
                "I need your help in performing segmentation on the {} in this image.",
                "Can you lend a hand by helping me with the segmentation tasks to find out the segmented {} in this image?",
                "I need your assistance to perform segmentation tasks to determine the segmented {} in this image.",
                "Kindly help me perform segmentation tasks to identify the segmented {} in this image.",
                "I am seeking your assistance to perform segmentation tasks and determine the segmented {} in this image.",
                "Can you please help me out by assisting me in performing segmentation tasks to find the segmented {} in this image?",
                "I need your help to perform segmentation tasks and determine the segmented {} in this image.",
                "I require assistance with segmentation tasks to identify the segmented {} in this image. Could you please help me out?",
                "Can you help me with segmenting the {} in this image using segmentation? I require assistance.",
                "Is it feasible for you to help me with segmenting the {} in this image by using segmentation? I am in need of help.",
                "Could you possibly aid me in segmenting the {} in this image using segmentation? I need some help.",
                "Would it be possible for you to assist me in segmenting the {} in this image via segmentation? I am in need of assistance.",
                "Can you help me segment the {} in this image with segmentation? I require some assistance.",
                "Is it possible for you to aid me with segmenting the {} in this image by using segmentation? I need some assistance.",
                "Would you be able to help me with segmenting the {} in this image using segmentation? I need assistance.",
                "Can you assist me in segmenting the {} in this image with segmentation? I am in need of help.",
                "Could you aid me with segmenting the {} in this image by using segmentation? I require assistance.",
                "Is it possible for you to help me segment the {} in this image using segmentation? I am in need of assistance.",
                "Can you assist me with segmenting the {} in this image using segmentation techniques? Your help would be greatly appreciated.",
                "I require aid with segmenting the {} in this picture using segmentation techniques. Could you give me a hand? Much obliged.",
                "Could you lend me a hand with segmenting the {} in this image using segmentation techniques? I need assistance, thank you.",
                "I'm in need of some assistance with segmenting the {} in this image using segmentation techniques. Can you help me out? Thanks.",
                "I need help segmenting the {} in this image using segmentation techniques. Could you assist me? Many thanks.",
                "I require your aid in partitioning the {} depicted in this image through segmentation. Can you help me out?",
                "Can you lend me a hand in segmenting the {} in this picture using segmentation? I'm in need of assistance.",
                "I'm having difficulty segmenting the {} in this image using segmentation. Would you be able to assist me?",
                "Using segmentation, can you support me in dividing the {} in this image? I'm seeking assistance.",
                "Would you mind helping me in segmenting the {} displayed in this image using segmentation? I need some help.",
                "I'm in need of assistance with segmenting the {} in this image using segmentation. Can you assist me?",
                "Can you provide assistance in segmenting the {} in this image through segmentation? I'm in need of help.",
                "Can you guide me through the process of segmenting the {} in this image using segmentation? I require your assistance.",
                "I'm having difficulty with segmenting the {} in this image through segmentation. Could you assist me?",
                "Would you be willing to assist me in segmenting the {} in this image using segmentation?",
                "I require some help. Can you help me to separate the {} in this image by using segmentation?",
                "Is it feasible for you to assist me in dividing the {} within this image utilizing segmentation?",
                "Would you be able to lend a hand in segmenting the {} within this image?",
                "Could you assist me in segmenting the {} depicted in this image using segmentation?",
                "Is it possible for you to aid me with the segmentation of the {} within this image?",
                "Can you help me to partition the {} in this image by using segmentation?",
                "Would it be possible for you to assist me with segmenting the {} within this image using segmentation techniques?",
                "Could you help me in segmenting the {} depicted in this image by using segmentation?",
                "Can you provide me with some assistance in segmenting the {} in this image using segmentation techniques?",
                "Is it feasible for you to lend a hand with segmenting the {} within this image using segmentation methods?",
                "Could you assist me in segmenting the {} in this picture using segmentation techniques?",
                "I'm having difficulty segmenting the {} in this image using segmentation techniques. Would you be willing to help me?",
                "I'm looking for someone to help me segment the {} in this image using segmentation techniques. Can you offer your assistance?",
                "I require assistance with segmenting the {} in this picture using segmentation techniques. Are you available to assist?",
                "Would you be able to lend me a hand with segmenting the {} in this image using segmentation techniques?",
                "I'm seeking assistance with segmenting the {} in this image using segmentation techniques. Can you provide any help?",
                "Could you help me out with segmenting the {} in this picture using segmentation techniques?",
                "I'm in need of help with segmenting the {} in this image using segmentation techniques. Would you mind helping me?",
                "Can you offer any assistance with segmenting the {} in this picture using segmentation techniques? I'm having trouble.",
                "I'm struggling to segment the {} in this image using segmentation techniques. Can you lend a hand with the task?",
                "Could you help me divide the {} in this picture into segments using segmentation?",
                "I need your assistance to segment the {} in this image using segmentation. Can you help me?",
                "Using segmentation, would you be able to aid me in dividing the {} in this image into segments?",
                "I'm looking for someone to assist me in segmenting the {} in this picture using segmentation.",
                "Can you help me separate the {} in this image into segments using segmentation, please?",
                "I require your assistance in segmenting the {} in this image using segmentation.",
                "Would you be able to help me segment the {} in this picture using segmentation, please?",
                "Can you help me divide the {} in this image into segments using segmentation?",
                "Could you please assist me in segmenting the {} in this picture using segmentation?",
                "Use segmentation technique to divide the {} range on the chart into segments.",
                "Cut the {} range on this graph into segments using the segmentation method.",
                "Segment the {} range on this graph using the segmentation technique.",
                "Divide the {} range on the chart into segments using segmentation.",
                "Employ the segmentation technique to cut the {} range on this graph into segments.",
                "Cut the {} range on this chart into sections using segmentation.",
                "Use the segmentation method to segment the {} range on this graph.",
                "Break down the {} range on this chart into segments using segmentation.",
                "Utilize the segmentation technique to divide the {} range on this chart into segments.",
                "Segment the {} range on this graph by using the segmentation technique to slice through it.",
                "The {} invert can be divided into segments by utilizing the Segmentation algorithm.",
                "Employing the Segmentation algorithm can help you divide the {} invert into distinct segments.",
                "You can use the Segmentation algorithm to segment the {} invert into smaller parts.",
                "To divide the {} invert into sections, try using the Segmentation algorithm.",
                "Utilizing the Segmentation algorithm is an effective way to split the {} invert into segments.",
                "If you need to segment the {} invert, the Segmentation algorithm can help you do that.",
                "By applying the Segmentation algorithm, you can split the {} invert into distinct segments.",
                "Utilizing a segmentation algorithm, partition the image and calculate the {}.",
                "Segment the image and determine the {} using an algorithm.",
                "Calculate the {} in the image through the use of a segmentation algorithm.",
                "Apply a segmentation algorithm to divide the image by {}.",
                "Employ an algorithm that segments the image and calculates the {}.",
                "Divide the image by {} using a segmentation algorithm.",
                "Utilize a segmentation algorithm to partition the image and determine the {}.",
                "Calculate the {} within the image using a segmentation algorithm to partition it.",
                "Using a segmentation algorithm, partition the image and calculate the {} within it.",
                "Employ an algorithm to segment the image and calculate the {}.",
                "Slice the {} in this image by means of segmentation.",
                "Use segmentation to cut the {} in this image into slices.",
                "Divide the {} in this image into segments using segmentation.",
                "Use segmentation to segment the {} shown in this image.",
                "Cut the {} displayed in this image into slices using segmentation.",
                "Slice the {} in this image into sections with the help of segmentation.",
                "Employ segmentation to divide the {} in this image into slices.",
                "Utilize segmentation to slice the {} shown in this image.",
                "Segment the {} in this image into slices by using segmentation.",
                "Apply segmentation to cut the {} in this image into slices.",
                "Isolate the {} in the image through the use of segmentation algorithms.",
                "Employ segmentation algorithms to partition the {} in the given image.",
                "Apply segmentation techniques to divide the {} in the image.",
                "Use segmentation algorithms to segment the {} in the image.",
                "Use segmentation methods to partition the {} in the image.",
                "Use segmentation techniques to isolate the {} in the image.",
                "Implement segmentation algorithms to separate the {} in the image.",
                "Utilize segmentation techniques to segment the {} in the image.",
                "Apply a segmentation algorithm to partition the {} in the given image.",
                "Employ a segmentation technique to isolate the {} in the image.",
                "Employ segmentation techniques to divide the {} values displayed on this graph.",
                "Apply segmentation to separate the {} data points in this chart.",
                "Use segmentation methodology to break down the {}s depicted on this graph.",
                "Implement segmentation strategies to partition the {} values shown in this chart.",
                "Adopt segmentation approaches to split the {} data on this graph.",
                "Utilize segmentation methods to segment the {} figures in this visualization.",
                "Utilize segmentation algorithms to divide the {} information in this chart.",
                "Employ segmentation tactics to split the {} values depicted on this graph.",
                "Apply segmentation procedures to separate the {} values on this chart.",
                "An algorithm for segmenting {} within this image.",
                "This image requires a segmentation algorithm for {} identification.",
                "The task at hand is to develop a segmentation algorithm for {} in this image.",
                "{} within this image needs to be identified using a segmentation algorithm.",
                "A segmentation algorithm is needed to detect {} in this image.",
                "The aim is to create an algorithm that can segment {} within this image.",
                "{} identification within this image can be achieved using a segmentation algorithm.",
                "This image calls for a segmentation algorithm that can accurately detect {}.",
                "The development of a segmentation algorithm is necessary for identifying {} in this image.",
                "The objective is to design a segmentation algorithm that can isolate {} in this image.",
                "Segmentation's algorithm can be used to divide the {} in this image.",
                "The segmentation algorithm is capable of breaking down the {} shown in this image.",
                "By applying segmentation's algorithm, the {} in this image can be separated.",
                "The segmentation's algorithm can be utilized to segment the {} displayed in this image.",
                "The {} shown in this image can be segmented using the segmentation algorithm.",
                "Through the use of segmentation's algorithm, the {} in this image can be partitioned.",
                "The segmentation algorithm is able to isolate the {} in this image.",
                "The {} depicted in this image can be broken down by the segmentation's algorithm.",
                "By employing segmentation's algorithm, it is possible to segment the {} in this image.",
                "The segmentation's algorithm can be implemented to segment the {} in this image.",
                "Utilize a segmentation algorithm to divide the {} in this image. How would you accomplish this?",
                "Segment the {} shown in this image with the help of a segmentation algorithm. What is your approach?",
                "Using a segmentation algorithm, divide the {} in this picture. What methods can you use?",
                "Employ a segmentation algorithm to separate the {} in this image. What are your techniques?",
                "Segment the {} featured in this image using a segmentation algorithm. What strategies can you employ?",
                "Apply a segmentation algorithm to divide the {} displayed in this image. How would you go about this?",
                "Use a segmentation algorithm to segment the {} in this picture. What is your plan of action?",
                "Divide the {} in this image using a segmentation algorithm. What steps can you take?",
                "How can you segment the {} in this image with a segmentation algorithm?",
                "The algorithm for segmentation can be employed to segment the {} in the image.",
                "By using a segmentation algorithm, it is possible to segment the {} in the image.",
                "Using a segmentation algorithm, the image's {} can be segmented.",
                "The {} in the image can be divided into segments using a segmentation algorithm.",
                "It is feasible to segment the {} in the image by employing a segmentation algorithm.",
                "Segmentation of the {} in the image can be achieved through the use of a segmentation algorithm.",
                "By using the segmentation algorithm, it is possible to segment the {} within the image.",
                "Segmentation's algorithm can be utilized to segment the {} in the image.",
                "The image's {} can be partitioned into segments by implementing a segmentation algorithm.",
                "Split the {} in this chart by using segmentation.",
                "Employ segmentation to divide the {} depicted in this graph.",
                "Use segmentation technique to break down the {} shown in this graph.",
                "The {} in this graph can be separated using segmentation.",
                "Segment the {} displayed in this graph.",
                "Divvy up the {} in this chart with the help of segmentation.",
                "Use segmentation to partition the {} in this visual representation.",
                "Employ the technique of segmentation to split the {} illustrated in this graph.",
                "The {} in this graph can be divided using the method of segmentation.",
                "Break down the {} in this chart by utilizing segmentation.",
                "Use the segmentation algorithm to divide the {} shown in this visual representation.",
                "Apply the segmentation algorithm to separate the {} in this visual representation.",
                "Utilize the segmentation algorithm to break down the {} depicted in this visual representation.",
                "Deploy the segmentation algorithm to partition the {} displayed in this visual representation.",
                "Employ the segmentation algorithm to subdivide the {} illustrated in this visual representation.",
                "Implement the segmentation algorithm to fragment the {} exhibited in this visual representation.",
                "Utilize the segmentation algorithm to split up the {} demonstrated in this visual representation.",
                "Use the segmentation algorithm to slice the {} presented in this visual representation.",
                "Employ the segmentation algorithm to dissect the {} represented in this visual illustration.",
                "Utilize the segmentation algorithm to separate the {} depicted in this visual depiction.",
                "Utilize a segmentation algorithm to divide the {} in the given image.",
                "Apply segmentation techniques to partition the {} shown in the image.",
                "Break down the {} in the image using a segmentation algorithm.",
                "Employ a segmentation algorithm to segment the {} displayed in the image.",
                "Utilize a segmentation technique to partition the {} depicted in the image.",
                "Apply a segmentation algorithm to divide the {} shown in the image into segments.",
                "Use segmentation methods to partition the {} in the given image.",
                "Segment the {} displayed in the image using a segmentation algorithm.",
                "Break up the {} in the image into sections by using a segmentation algorithm.",
                "Apply segmentation techniques to segment the {} in the given image.",
                "Use a segmentation algorithm to divide the {} in this image into segments.",
                "Employ a segmentation algorithm to partition the {} in this image.",
                "Apply a segmentation algorithm to break down the {} in this image into sections.",
                "Implement a segmentation algorithm to separate the {} in this image into distinct parts.",
                "Utilize a segmentation algorithm to isolate the {} in this image into different segments.",
                "Utilize a segmentation algorithm to divide the {} in this image into smaller parts.",
                "Deploy a segmentation algorithm to split the {} in this image into separate sections.",
                "Use a segmentation algorithm to categorize the {} in this image into different groups.",
                "Implement a segmentation algorithm to separate the {} in this image into individual parts.",
                "Apply a segmentation algorithm to divide the {} in this image into distinct components.",
                "Provide a {} breakdown for the complete image.",
                "Analyze the image and give a {} breakdown for the whole thing.",
                "Break down the {} composition of the entire image.",
                "Can you give me a {} breakdown of the image as a whole?",
                "Please provide a breakdown of the {}s for the complete image.",
                "I would like to see a {} breakdown of the entire image.",
                "What is the {} breakdown for the entire image?",
                "Tell me the {} breakdown of the image in its entirety.",
                "Divide this image by the {} indicated.",
                "Perform image division with the {} value.",
                "Divide this image by the ratio indicated by the {} value.",
                "Divide the {} shown in this image.",
                "Separate the {} within this picture.",
                "Break down the {} depicted in this photo.",
                "Cut the {} displayed in this snapshot.",
                "Fragment the {} presented in this visual.",
                "Partition the {} demonstrated in this illustration.",
                "Subdivide the {} shown in this rendering.",
                "Give a comprehensive breakdown {} of the entire picture.",
                "Break down the image in its totality by {}s.",
                "Present a {} breakdown for the whole image.",
                "Dissect the complete picture by {}s.",
                "Outline the {} breakdown of the entire image.",
                "Divide the image into {}s for a complete breakdown.",
                "Provide a breakdown of the image in its entirety using {}s.",
                "Divide the {} shown in this image."
            ]
        else:
            self.prompt_list = [
                'Could someone please break down the {} \into individual parts?',
                'Can you provide me with a segment of the {}?',
                'Please divide the {} \into smaller parts',
                'Segment the {}',
                'Can you provide me with a segment of the {}?',
                'Help me segment the {}',
                'Would you be willing to split the {} with me?'
            ]



    def transform(self, results):
        ann_list = results['ann_list']
        cls_list = results['cls_list']
        class_name = random.choice(cls_list)
        h, w = results['original_shape']
        gt = np.zeros((h, w), dtype=np.uint8)
        gt_rle = np.zeros((h, w), dtype=np.uint8)
        for item in ann_list:
            if item['category_name'] == class_name:
                if isinstance(item['segmentation'], list):
                    for s in item['segmentation']:
                        s_array = np.array(s).reshape(-1, 2)  # [n_points, 2]
                        cv2.fillPoly(gt, s_array.astype(np.int32)[np.newaxis, :, :], (1, 1, 1))
                else:
                    mask_rle = np.zeros(w * h, dtype=np.uint8)
                    rle = item['segmentation']['counts']
                    start = 0
                    for i, value in enumerate(rle):
                        start += value
                        if i % 2 == 1:
                            mask_rle[start - value:start] = 1
                    gt_rle = mask_rle.reshape(w, h).T

        gt = np.logical_or(gt, gt_rle)



        prompt_item = random.choice(self.prompt_list)
        prompt_item = prompt_item.format(class_name)
        if not self.test_mode:
            color_list = []
            if self.only_mask:
                color_list = [["white", "black"], ["black", "white"], ] if self.random_color else [["white", "black"],]
            else:
                color_list = [["white", "black"], ["black", "white"], ['white', 'input image'],
                              ['black', 'input image']]
            color = random.choice(color_list)
            prompt_item = prompt_item + " The target is annotated with {} color and background is {}.".format(color[0],
                                                                                                              color[1])
            if color[0] == "white" and color[1] == "black":
                gt = gt
                gt = gt * 255
                gt = np.stack([gt, gt, gt], axis=-1)
            elif color[0] == "black" and color[1] == "white":
                gt = 1.0 - gt
                gt = gt * 255
                gt = np.stack([gt, gt, gt], axis=-1)
            elif color[0] == 'white' and color[1] == 'input image':
                mask = 1.0 - gt
                tmp_gt = gt * 255
                tmp_gt = np.stack([tmp_gt, tmp_gt, tmp_gt], axis=-1)
                gt = mask[:, :, np.newaxis] * results['img'] + tmp_gt
            else:
                mask = 1.0 - gt
                tmp_gt = np.stack([mask, mask, mask], axis=-1)
                gt = tmp_gt * results['img']

        else:
            prompt_item = prompt_item + " The target is annotated with white color and background is black."
            gt = gt * 255
            gt = np.stack([gt, gt, gt], axis=-1)

        results['prompt'] = prompt_item

        gt = gt.astype(np.float32)
        results['mask_gt'] = gt

        gt_shape = gt.shape
        img_shape = results['img'].shape
        assert gt_shape[0] == img_shape[0] and gt_shape[1] == img_shape[1]

        return results


@TRANSFORMS.register_module()
class MaskPromptSelectADE20k(BaseTransform):

    def __init__(self, test_mode=False, only_mask=True, random_color=True):
        self.test_mode = test_mode
        self.only_mask = only_mask
        self.random_color = random_color

        if not self.test_mode:
            self.prompt_list = [
                "Segment the {}.",
                "Segment {}.",
                "Perform image segmentation to extract the {} from the image.",
                "Separate the {} from the rest of the image using image segmentation.",
                "Use computer vision techniques to segment the {} from the image.",
                "Employ image segmentation to isolate the {} in the image.",
                "Apply segmentation algorithms to extract the {} from the picture.",
                "Use computer vision to partition the image and focus on the {}.",
                "Utilize image segmentation to extract the {}'s features from the image.",
                "Segment the {} using image processing techniques.",
                "Perform object segmentation to isolate the {} in the image.",
                "Employ image segmentation to detect and extract the {} from the image.",
                "Use computer vision to accurately segment the {} in the picture.",
                "Extract the {} from the image by performing image segmentation.",
                "Use image segmentation to locate and extract the {} from the image.",
                "Partition the image using computer vision techniques to highlight the {}.",
                "Employ segmentation methods to detect and extract the {} from the image.",
                "Use image segmentation algorithms to identify and extract the {}.",
                "Apply computer vision techniques to accurately segment the {} in the image.",
                "Separate the {} from the background using image segmentation.",
                "Use segmentation algorithms to distinguish the {} from the rest of the image.",
                "Use image segmentation to identify the boundaries of the {} in the image.",
                "Split the {} in this picture.",
                "Break down {} for the image in its entirety.",
                "Segment the {} in this image with the help of segmentation's algorithm.",
                "Segment the {} in this image using segmentation's algorithm.",
                "Utilize segmentation's algorithm to split the {} in this visual representation.",
                "Use segmentation to split the {} in this graph.",
                "Segment the {} in this image using segmentation's algorithm. How can you do it?",
                "The {} in this image can be segmented using segmentation's algorithm. How does it work?",
                "segmentation algorithm for {} in this image.",
                "Use segmentation to split the {} in this graph.",
                "Segment the {} in this image using segmentation's algorithm.",
                "Using Segmentation, slice the {} in this image.",
                "Divide by {} in this image using segmentation's algorithm.",
                "To split this figure into segments, use the Segmentation algorithm to divide the {} invert.",
                "Sorting by division using segmentation's technique, specifically the {} in this diagram and its related components.",
                "Slice through the {} range on this graphic with segments using Segmentions technique.",
                "Can you assist me in segmenting the {} in this image using segmentation?",
                "I need help with segmenting the {} in this image using segmentation techniques. Can you lend a hand?",
                "Would it be possible for you to aid me with segmenting the {} in this image using segmentation?",
                "Can you assist me in segmenting the {} in this image using segmentation? I need help.",
                "I need help with segmenting the {} in this image using segmentation techniques. Can you lend a hand? Thank you.",
                "Would it be possible for you to aid me with segmenting the {} in this image using segmentation? I need help.",
                "Please aid me by assisting me in performing segmentation tasks to determine the segmented {} in this image.",
                "In this image, I am attempting to perform segmentation on the {} and would appreciate your help.",
                "I need help with segmentsation of the given {} in this image.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image. Could you lend me a hand?",
                "Can you show me how to use the Segmentation tool to divide the {} in this image? Would you be able to assist me with this task?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image? I could use some assistance.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "I require your assistance with utilizing the Segmentations tool to break down the {} in this image.",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image. Could you lend me a hand?",
                "Can you show me how to use the Segmentation tool to divide the {} in this image? Would you be able to assist me with this task?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image? I could use some assistance.",
                "Would you be able to assist me in using the Segmentation feature to divide the {} in this image?",
                "I require your assistance with utilizing the Segmentations tool to break down the {} in this image.",
                "Can you help me out with using the Segmentation function to divide the {} in this picture?",
                "I am curious about how to use the Segmentations feature to split the {} in this image.",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "I need help using the Segmentations function to split the {} in this image. Can you assist me?",
                "Would it be possible for you to show me how to use the Segmentation feature to divide the {} in this picture?",
                "Can you lend me a hand with utilizing the Segmentation function to split the {} in this image?",
                "I'm having trouble using the Segmentation tool to split the {} in this image. Could you assist me?",
                "Can you guide me through using the Segmentation feature to split the {} in this image?",
                "Can you assist me with dividing the specified {} in this image?",
                "I am seeking assistance with the segmentation of the specified {} in this image.",
                "Could you help me separate the indicated {} in this image?",
                "I am in need of help with dividing the specified {} in this picture.",
                "Can you lend a hand in segmenting the given {} within this image?",
                "I require assistance with splitting the indicated {} in this picture.",
                "Could you help me with the segmentation of the specified {} in this photograph?",
                "I am seeking guidance on dividing the given {} within this image.",
                "Can you provide assistance with segmenting the indicated {} in this image?",
                "It would be great if you could assist me with the segmentation of the {} in this image.",
                "I'm trying to segment the {} in this image and would be grateful for your aid.",
                "Can you lend me a hand with segmenting the {} in this image?",
                "My goal is to perform segmentation on the {} in this image and I would appreciate any help you can provide.",
                "I could use your help in segmenting the {} in this image.",
                "I'm attempting to perform segmentation on the {} in this image and would welcome your assistance.",
                "Would you be willing to help me with the segmentation of the {} in this image?",
                "If you could lend me your expertise, I would greatly appreciate your help with segmenting the {} in this image.",
                "My aim is to segment the {} in this image and I would be thankful for your help.",
                "I need your help in performing segmentation on the {} in this image.",
                "Can you lend a hand by helping me with the segmentation tasks to find out the segmented {} in this image?",
                "I need your assistance to perform segmentation tasks to determine the segmented {} in this image.",
                "Kindly help me perform segmentation tasks to identify the segmented {} in this image.",
                "I am seeking your assistance to perform segmentation tasks and determine the segmented {} in this image.",
                "Can you please help me out by assisting me in performing segmentation tasks to find the segmented {} in this image?",
                "I need your help to perform segmentation tasks and determine the segmented {} in this image.",
                "I require assistance with segmentation tasks to identify the segmented {} in this image. Could you please help me out?",
                "Can you help me with segmenting the {} in this image using segmentation? I require assistance.",
                "Is it feasible for you to help me with segmenting the {} in this image by using segmentation? I am in need of help.",
                "Could you possibly aid me in segmenting the {} in this image using segmentation? I need some help.",
                "Would it be possible for you to assist me in segmenting the {} in this image via segmentation? I am in need of assistance.",
                "Can you help me segment the {} in this image with segmentation? I require some assistance.",
                "Is it possible for you to aid me with segmenting the {} in this image by using segmentation? I need some assistance.",
                "Would you be able to help me with segmenting the {} in this image using segmentation? I need assistance.",
                "Can you assist me in segmenting the {} in this image with segmentation? I am in need of help.",
                "Could you aid me with segmenting the {} in this image by using segmentation? I require assistance.",
                "Is it possible for you to help me segment the {} in this image using segmentation? I am in need of assistance.",
                "Can you assist me with segmenting the {} in this image using segmentation techniques? Your help would be greatly appreciated.",
                "I require aid with segmenting the {} in this picture using segmentation techniques. Could you give me a hand? Much obliged.",
                "Could you lend me a hand with segmenting the {} in this image using segmentation techniques? I need assistance, thank you.",
                "I'm in need of some assistance with segmenting the {} in this image using segmentation techniques. Can you help me out? Thanks.",
                "I need help segmenting the {} in this image using segmentation techniques. Could you assist me? Many thanks.",
                "I require your aid in partitioning the {} depicted in this image through segmentation. Can you help me out?",
                "Can you lend me a hand in segmenting the {} in this picture using segmentation? I'm in need of assistance.",
                "I'm having difficulty segmenting the {} in this image using segmentation. Would you be able to assist me?",
                "Using segmentation, can you support me in dividing the {} in this image? I'm seeking assistance.",
                "Would you mind helping me in segmenting the {} displayed in this image using segmentation? I need some help.",
                "I'm in need of assistance with segmenting the {} in this image using segmentation. Can you assist me?",
                "Can you provide assistance in segmenting the {} in this image through segmentation? I'm in need of help.",
                "Can you guide me through the process of segmenting the {} in this image using segmentation? I require your assistance.",
                "I'm having difficulty with segmenting the {} in this image through segmentation. Could you assist me?",
                "Would you be willing to assist me in segmenting the {} in this image using segmentation?",
                "I require some help. Can you help me to separate the {} in this image by using segmentation?",
                "Is it feasible for you to assist me in dividing the {} within this image utilizing segmentation?",
                "Would you be able to lend a hand in segmenting the {} within this image?",
                "Could you assist me in segmenting the {} depicted in this image using segmentation?",
                "Is it possible for you to aid me with the segmentation of the {} within this image?",
                "Can you help me to partition the {} in this image by using segmentation?",
                "Would it be possible for you to assist me with segmenting the {} within this image using segmentation techniques?",
                "Could you help me in segmenting the {} depicted in this image by using segmentation?",
                "Can you provide me with some assistance in segmenting the {} in this image using segmentation techniques?",
                "Is it feasible for you to lend a hand with segmenting the {} within this image using segmentation methods?",
                "Could you assist me in segmenting the {} in this picture using segmentation techniques?",
                "I'm having difficulty segmenting the {} in this image using segmentation techniques. Would you be willing to help me?",
                "I'm looking for someone to help me segment the {} in this image using segmentation techniques. Can you offer your assistance?",
                "I require assistance with segmenting the {} in this picture using segmentation techniques. Are you available to assist?",
                "Would you be able to lend me a hand with segmenting the {} in this image using segmentation techniques?",
                "I'm seeking assistance with segmenting the {} in this image using segmentation techniques. Can you provide any help?",
                "Could you help me out with segmenting the {} in this picture using segmentation techniques?",
                "I'm in need of help with segmenting the {} in this image using segmentation techniques. Would you mind helping me?",
                "Can you offer any assistance with segmenting the {} in this picture using segmentation techniques? I'm having trouble.",
                "I'm struggling to segment the {} in this image using segmentation techniques. Can you lend a hand with the task?",
                "Could you help me divide the {} in this picture into segments using segmentation?",
                "I need your assistance to segment the {} in this image using segmentation. Can you help me?",
                "Using segmentation, would you be able to aid me in dividing the {} in this image into segments?",
                "I'm looking for someone to assist me in segmenting the {} in this picture using segmentation.",
                "Can you help me separate the {} in this image into segments using segmentation, please?",
                "I require your assistance in segmenting the {} in this image using segmentation.",
                "Would you be able to help me segment the {} in this picture using segmentation, please?",
                "Can you help me divide the {} in this image into segments using segmentation?",
                "Could you please assist me in segmenting the {} in this picture using segmentation?",
                "Use segmentation technique to divide the {} range on the chart into segments.",
                "Cut the {} range on this graph into segments using the segmentation method.",
                "Segment the {} range on this graph using the segmentation technique.",
                "Divide the {} range on the chart into segments using segmentation.",
                "Employ the segmentation technique to cut the {} range on this graph into segments.",
                "Cut the {} range on this chart into sections using segmentation.",
                "Use the segmentation method to segment the {} range on this graph.",
                "Break down the {} range on this chart into segments using segmentation.",
                "Utilize the segmentation technique to divide the {} range on this chart into segments.",
                "Segment the {} range on this graph by using the segmentation technique to slice through it.",
                "The {} invert can be divided into segments by utilizing the Segmentation algorithm.",
                "Employing the Segmentation algorithm can help you divide the {} invert into distinct segments.",
                "You can use the Segmentation algorithm to segment the {} invert into smaller parts.",
                "To divide the {} invert into sections, try using the Segmentation algorithm.",
                "Utilizing the Segmentation algorithm is an effective way to split the {} invert into segments.",
                "If you need to segment the {} invert, the Segmentation algorithm can help you do that.",
                "By applying the Segmentation algorithm, you can split the {} invert into distinct segments.",
                "Utilizing a segmentation algorithm, partition the image and calculate the {}.",
                "Segment the image and determine the {} using an algorithm.",
                "Calculate the {} in the image through the use of a segmentation algorithm.",
                "Apply a segmentation algorithm to divide the image by {}.",
                "Employ an algorithm that segments the image and calculates the {}.",
                "Divide the image by {} using a segmentation algorithm.",
                "Utilize a segmentation algorithm to partition the image and determine the {}.",
                "Calculate the {} within the image using a segmentation algorithm to partition it.",
                "Using a segmentation algorithm, partition the image and calculate the {} within it.",
                "Employ an algorithm to segment the image and calculate the {}.",
                "Slice the {} in this image by means of segmentation.",
                "Use segmentation to cut the {} in this image into slices.",
                "Divide the {} in this image into segments using segmentation.",
                "Use segmentation to segment the {} shown in this image.",
                "Cut the {} displayed in this image into slices using segmentation.",
                "Slice the {} in this image into sections with the help of segmentation.",
                "Employ segmentation to divide the {} in this image into slices.",
                "Utilize segmentation to slice the {} shown in this image.",
                "Segment the {} in this image into slices by using segmentation.",
                "Apply segmentation to cut the {} in this image into slices.",
                "Isolate the {} in the image through the use of segmentation algorithms.",
                "Employ segmentation algorithms to partition the {} in the given image.",
                "Apply segmentation techniques to divide the {} in the image.",
                "Use segmentation algorithms to segment the {} in the image.",
                "Use segmentation methods to partition the {} in the image.",
                "Use segmentation techniques to isolate the {} in the image.",
                "Implement segmentation algorithms to separate the {} in the image.",
                "Utilize segmentation techniques to segment the {} in the image.",
                "Apply a segmentation algorithm to partition the {} in the given image.",
                "Employ a segmentation technique to isolate the {} in the image.",
                "Employ segmentation techniques to divide the {} values displayed on this graph.",
                "Apply segmentation to separate the {} data points in this chart.",
                "Use segmentation methodology to break down the {}s depicted on this graph.",
                "Implement segmentation strategies to partition the {} values shown in this chart.",
                "Adopt segmentation approaches to split the {} data on this graph.",
                "Utilize segmentation methods to segment the {} figures in this visualization.",
                "Utilize segmentation algorithms to divide the {} information in this chart.",
                "Employ segmentation tactics to split the {} values depicted on this graph.",
                "Apply segmentation procedures to separate the {} values on this chart.",
                "An algorithm for segmenting {} within this image.",
                "This image requires a segmentation algorithm for {} identification.",
                "The task at hand is to develop a segmentation algorithm for {} in this image.",
                "{} within this image needs to be identified using a segmentation algorithm.",
                "A segmentation algorithm is needed to detect {} in this image.",
                "The aim is to create an algorithm that can segment {} within this image.",
                "{} identification within this image can be achieved using a segmentation algorithm.",
                "This image calls for a segmentation algorithm that can accurately detect {}.",
                "The development of a segmentation algorithm is necessary for identifying {} in this image.",
                "The objective is to design a segmentation algorithm that can isolate {} in this image.",
                "Segmentation's algorithm can be used to divide the {} in this image.",
                "The segmentation algorithm is capable of breaking down the {} shown in this image.",
                "By applying segmentation's algorithm, the {} in this image can be separated.",
                "The segmentation's algorithm can be utilized to segment the {} displayed in this image.",
                "The {} shown in this image can be segmented using the segmentation algorithm.",
                "Through the use of segmentation's algorithm, the {} in this image can be partitioned.",
                "The segmentation algorithm is able to isolate the {} in this image.",
                "The {} depicted in this image can be broken down by the segmentation's algorithm.",
                "By employing segmentation's algorithm, it is possible to segment the {} in this image.",
                "The segmentation's algorithm can be implemented to segment the {} in this image.",
                "Utilize a segmentation algorithm to divide the {} in this image. How would you accomplish this?",
                "Segment the {} shown in this image with the help of a segmentation algorithm. What is your approach?",
                "Using a segmentation algorithm, divide the {} in this picture. What methods can you use?",
                "Employ a segmentation algorithm to separate the {} in this image. What are your techniques?",
                "Segment the {} featured in this image using a segmentation algorithm. What strategies can you employ?",
                "Apply a segmentation algorithm to divide the {} displayed in this image. How would you go about this?",
                "Use a segmentation algorithm to segment the {} in this picture. What is your plan of action?",
                "Divide the {} in this image using a segmentation algorithm. What steps can you take?",
                "How can you segment the {} in this image with a segmentation algorithm?",
                "The algorithm for segmentation can be employed to segment the {} in the image.",
                "By using a segmentation algorithm, it is possible to segment the {} in the image.",
                "Using a segmentation algorithm, the image's {} can be segmented.",
                "The {} in the image can be divided into segments using a segmentation algorithm.",
                "It is feasible to segment the {} in the image by employing a segmentation algorithm.",
                "Segmentation of the {} in the image can be achieved through the use of a segmentation algorithm.",
                "By using the segmentation algorithm, it is possible to segment the {} within the image.",
                "Segmentation's algorithm can be utilized to segment the {} in the image.",
                "The image's {} can be partitioned into segments by implementing a segmentation algorithm.",
                "Split the {} in this chart by using segmentation.",
                "Employ segmentation to divide the {} depicted in this graph.",
                "Use segmentation technique to break down the {} shown in this graph.",
                "The {} in this graph can be separated using segmentation.",
                "Segment the {} displayed in this graph.",
                "Divvy up the {} in this chart with the help of segmentation.",
                "Use segmentation to partition the {} in this visual representation.",
                "Employ the technique of segmentation to split the {} illustrated in this graph.",
                "The {} in this graph can be divided using the method of segmentation.",
                "Break down the {} in this chart by utilizing segmentation.",
                "Use the segmentation algorithm to divide the {} shown in this visual representation.",
                "Apply the segmentation algorithm to separate the {} in this visual representation.",
                "Utilize the segmentation algorithm to break down the {} depicted in this visual representation.",
                "Deploy the segmentation algorithm to partition the {} displayed in this visual representation.",
                "Employ the segmentation algorithm to subdivide the {} illustrated in this visual representation.",
                "Implement the segmentation algorithm to fragment the {} exhibited in this visual representation.",
                "Utilize the segmentation algorithm to split up the {} demonstrated in this visual representation.",
                "Use the segmentation algorithm to slice the {} presented in this visual representation.",
                "Employ the segmentation algorithm to dissect the {} represented in this visual illustration.",
                "Utilize the segmentation algorithm to separate the {} depicted in this visual depiction.",
                "Utilize a segmentation algorithm to divide the {} in the given image.",
                "Apply segmentation techniques to partition the {} shown in the image.",
                "Break down the {} in the image using a segmentation algorithm.",
                "Employ a segmentation algorithm to segment the {} displayed in the image.",
                "Utilize a segmentation technique to partition the {} depicted in the image.",
                "Apply a segmentation algorithm to divide the {} shown in the image into segments.",
                "Use segmentation methods to partition the {} in the given image.",
                "Segment the {} displayed in the image using a segmentation algorithm.",
                "Break up the {} in the image into sections by using a segmentation algorithm.",
                "Apply segmentation techniques to segment the {} in the given image.",
                "Use a segmentation algorithm to divide the {} in this image into segments.",
                "Employ a segmentation algorithm to partition the {} in this image.",
                "Apply a segmentation algorithm to break down the {} in this image into sections.",
                "Implement a segmentation algorithm to separate the {} in this image into distinct parts.",
                "Utilize a segmentation algorithm to isolate the {} in this image into different segments.",
                "Utilize a segmentation algorithm to divide the {} in this image into smaller parts.",
                "Deploy a segmentation algorithm to split the {} in this image into separate sections.",
                "Use a segmentation algorithm to categorize the {} in this image into different groups.",
                "Implement a segmentation algorithm to separate the {} in this image into individual parts.",
                "Apply a segmentation algorithm to divide the {} in this image into distinct components.",
                "Provide a {} breakdown for the complete image.",
                "Analyze the image and give a {} breakdown for the whole thing.",
                "Break down the {} composition of the entire image.",
                "Can you give me a {} breakdown of the image as a whole?",
                "Please provide a breakdown of the {}s for the complete image.",
                "I would like to see a {} breakdown of the entire image.",
                "What is the {} breakdown for the entire image?",
                "Tell me the {} breakdown of the image in its entirety.",
                "Divide this image by the {} indicated.",
                "Perform image division with the {} value.",
                "Divide this image by the ratio indicated by the {} value.",
                "Divide the {} shown in this image.",
                "Separate the {} within this picture.",
                "Break down the {} depicted in this photo.",
                "Cut the {} displayed in this snapshot.",
                "Fragment the {} presented in this visual.",
                "Partition the {} demonstrated in this illustration.",
                "Subdivide the {} shown in this rendering.",
                "Give a comprehensive breakdown {} of the entire picture.",
                "Break down the image in its totality by {}s.",
                "Present a {} breakdown for the whole image.",
                "Dissect the complete picture by {}s.",
                "Outline the {} breakdown of the entire image.",
                "Divide the image into {}s for a complete breakdown.",
                "Provide a breakdown of the image in its entirety using {}s.",
                "Divide the {} shown in this image."
            ]
        else:
            self.prompt_list = [
                'Could someone please break down the {} \into individual parts?',
                'Can you provide me with a segment of the {}?',
                'Please divide the {} \into smaller parts',
                'Segment the {}',
                'Can you provide me with a segment of the {}?',
                'Help me segment the {}',
                'Would you be willing to split the {} with me?'
            ]

    def transform(self, results):
        
        cls_list = results['cls_list']
        class_names = random.choice(cls_list)
        class_name = class_names
        # if ", " in class_names:
        #     class_name_items = class_names.split(', ')
        #     class_name = random.choice(class_name_items)
        h, w = results['original_shape']

        prompt_item = random.choice(self.prompt_list)
        prompt_item = prompt_item.format(class_name)

        gt = results['mask_gt']
        # import pdb;pdb.set_trace()
        if not self.test_mode:
            color_list = []
            if self.only_mask:
                color_list = [["white", "black"], ["black", "white"], ] if self.random_color else [["white", "black"],]
            else:
                color_list = [["white", "black"], ["black", "white"], ['white', 'input image'],
                              ['black', 'input image']]
            color = random.choice(color_list)
            prompt_item = prompt_item + " The target is annotated with {} color and background is {}.".format(color[0],
                                                                                                              color[1])
            if color[0] == "white" and color[1] == "black":
                gt = gt
                gt = gt * 255
                gt = np.stack([gt, gt, gt], axis=-1)
            elif color[0] == "black" and color[1] == "white":
                gt = 1.0 - gt
                gt = gt * 255
                gt = np.stack([gt, gt, gt], axis=-1)
            elif color[0] == 'white' and color[1] == 'input image':
                mask = 1.0 - gt
                tmp_gt = gt * 255
                tmp_gt = np.stack([tmp_gt, tmp_gt, tmp_gt], axis=-1)
                gt = mask[:, :, np.newaxis] * results['img'] + tmp_gt
            else:
                mask = 1.0 - gt
                tmp_gt = np.stack([mask, mask, mask], axis=-1)
                gt = tmp_gt * results['img']

        else:
            prompt_item = prompt_item + " The target is annotated with white color and background is black."
            gt = gt * 255
            gt = np.stack([gt, gt, gt], axis=-1)

        results['prompt'] = prompt_item

        gt = gt.astype(np.float32)
        results['mask_gt'] = gt

        gt_shape = gt.shape
        img_shape = results['img'].shape
        assert gt_shape[0] == img_shape[0] and gt_shape[1] == img_shape[1]

        return results


@TRANSFORMS.register_module()
class MaskPromptSelectVIS(BaseTransform):
    def __init__(self, test_mode=False, only_mask=True, use_text=False):
        self.test_mode = test_mode
        self.only_mask = only_mask
        self.use_text = use_text

        if self.use_text:
            self.prompt_list = [
                "The first frame of the video is {f_frame}, and the target object in this frame is {f_mask}, the remaining frames are {r_frame}, please segment the target object in the remaining frames.",]
        else:
            self.prompt_list = [ "{f_frame} {f_mask} {r_frame}.", ]
    def transform(self, results):
        prompt_item = random.choice(self.prompt_list)
        gt = results['mask_gt']
        prompt_item = prompt_item + " The target is annotated with white color and background is black." if self.use_text else prompt_item
        gt = gt * 255
        gt = np.stack([gt, gt, gt], axis=-1)
        results['prompt'] = prompt_item

        gt = gt.astype(np.float32)
        results['mask_gt'] = gt
        gt_shape = gt.shape
        img_shape = results['img'].shape
        assert gt_shape[0] == img_shape[0] and gt_shape[1] == img_shape[1]

        return results

@TRANSFORMS.register_module()
class MaskPromptSelectNYU(BaseTransform):

    def __init__(self, test_mode=False, only_mask=True):
        self.test_mode = test_mode
        self.only_mask = only_mask

        if not self.test_mode:
            self.prompt_list = [
                "Estimate the depth of this image."
                "Approximate the depth of this image."
                "Make an estimation of how deep the this image is."
                "Provide a rough calculation of the image's depth."
                "Give an approximate measurement of the image's depth."
                "Make an informed guess of the depth of the image."
                "Make an estimation of how deep the image goes."
                "Create a monocular depth map."
                "Plese Create a monocular depth map."
                "Generate a depth map using a single camera perspective."
                "Produce a depth map with a monocular setup."
                "Construct a depth map using a lone camera view."
                "Formulate a monocular-based depth map."
                "Develop a depth map using only one camera input."
                "Build a depth map utilizing a single camera viewpoint."
                "Craft a monocular-generated depth map."
                "Generate a depth map from a monocular source."
                "Create a depth map using a solitary camera perspective."
                "Produce a monocular-derived depth map."
                "Determine the image's depth."
                "Make an estimation of the depth in this image."
                "Assess the depth present in this image."
                "Calculate the depth of this image."
                "Gauge the depth portrayed in this image."
                "Evaluate the depth depicted in this image."
                "Approximate the depth of this image."
                "Measure the depth captured in this image."
                "Determine the spatial depth of this image."
                "Quantify the depth perception of this image."
                "Estimate the depth of the image."
                "Calculate the depth of the image."
                "Specify the depth of the image."
                "Determine the depth of the image."
                "Make an educated guess about how deep the image is."
                "Specify the depth of the image. How would you go about doing that?"
                "How much can you tell me about how deep this picture was."
                "Find out about how deep an object was in the image taken."
                "Produce a depth map using only one eye."
                "Construct an angled monocular depth map."
                "Make your own monocular depth map."
                "Produce a depth map using only one eye."
                "Create a monocular depth map."

            ]
        else:
            self.prompt_list = [
                "Approximate the depth of this image.",
                "Make an estimation of how deep the this image is.",
                "Provide a rough calculation of the image's depth.",
                "Give an approximate measurement of the image's depth.",
                "Make an informed guess of the depth of the image.",
                "Make an estimation of how deep the image goes.",
                "Estimate the depth of this image."
            ]

    def transform(self, results):

        prompt_item = random.choice(self.prompt_list)

        gt = results['mask_gt']

        prompt_item = prompt_item + " White represents the farthest, black represents the nearest:"

        gt = np.stack([gt, gt, gt], axis=-1)

        results['prompt'] = prompt_item

        gt = gt.astype(np.float32)
        results['mask_gt'] = gt

        gt_shape = gt.shape
        img_shape = results['img'].shape
        assert gt_shape[0] == img_shape[0] and gt_shape[1] == img_shape[1]

        return results
