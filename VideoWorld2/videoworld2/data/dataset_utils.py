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
import torch
import torchvision.transforms.functional as F
import albumentations as A
import numpy as np
import kornia as K
import kornia.augmentation as K_aug

class Resize_Preprocess:
    def __init__(self, size):
        """
        Initialize the preprocessing class with the target size.
        Args:
        size (tuple): The target height and width as a tuple (height, width).
        """
        self.size = size

    def __call__(self, video_frames):
        """
        Apply the transformation to each frame in the video.
        Args:
        video_frames (torch.Tensor): A tensor representing a batch of video frames.
        Returns:
        torch.Tensor: The transformed video frames.
        """
        # Resize each frame in the video
        resized_frames = torch.stack([F.resize(frame, self.size, antialias=True) for frame in video_frames])
        return resized_frames

class GeometricAugmentor():
    def __init__(self,) -> None:
        self.transform = A.ReplayCompose([
               
                A.ShiftScaleRotate(
                    shift_limit=0.1,    
                    scale_limit=0.2,   
                    rotate_limit=30,   
                    # border_mode=cv2.BORDER_WRAP,
                    p=0.7
                ),
                
                A.Perspective(
                    # scale=0.1,
                    keep_size=True,
                    # pad_mode='reflect',
                    # fit_output=True,
                    p=0.7
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
                # A.GaussianBlur(blur_limit=(1, 3), p=0.5), 
                # A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                # A.ISONoise()

            ], p=1)
        # self.turn_on = turn_on

    def __call__(self, video_frames: dict) -> dict:
        # import pdb;pdb.set_trace()
        # if not self.turn_on:
        #     return data_dict
        # for key in self.input_keys:
        if isinstance(video_frames, torch.Tensor):
            data = video_frames.permute(0, 2, 3, 1).numpy()
        first_frame = data[0]
        transformed = self.transform(image=first_frame)
        replay_params = transformed['replay']

        augmented_frames = [transformed['image']]
        for frame in data[1:]:
            
            transformed_frame = A.ReplayCompose.replay(replay_params, image=frame)
            augmented_frames.append(transformed_frame['image'])
        augmented_frames = torch.from_numpy(np.stack(augmented_frames)).permute(0, 3, 1, 2)
 
        return augmented_frames

class GeometricAugmentor_K():
    def __init__(self,) -> None:
        self.transform = K_aug.AugmentationSequential(
           
            K_aug.RandomAffine(
                degrees=30.0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2), 
                p=0.7,
                same_on_batch=True,
            ),
           
            K_aug.RandomPerspective(
                distortion_scale=0.2, 
                p=0.7,
                same_on_batch=True,
            ),
            
            K_aug.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,  
                hue=0.055, 
                p=1.0,
                same_on_batch=True,
            ),
            data_keys=["input"],
        )
        # self.turn_on = turn_on

    def __call__(self, video_frames: dict) -> dict:
        # import pdb;pdb.set_trace()
        if video_frames.dtype == torch.uint8:
            video_frames = video_frames.float()
        if video_frames.max() > 1:
            video_frames = video_frames / 255.0
        augmented_frames = self.transform(video_frames) 
        # augmented_frames = torch.clip(augmented_frames, 0, 255)
        return augmented_frames



class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True
