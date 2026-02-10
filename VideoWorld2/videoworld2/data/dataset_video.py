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
import os
import pickle
import traceback
import warnings

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as T
from torchvision.transforms.v2 import UniformTemporalSubsample
import random
from videoworld2.data.dataset_utils import Resize_Preprocess, ToTensorVideo, GeometricAugmentor, GeometricAugmentor_K
from imaginaire.utils import log
from collections import defaultdict
"""
Test the dataset with the following command:
python -m cosmos_predict2.data.dataset_video
"""


class Dataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        video_size,
        ctrl_type=None,
        extra_video=False, #For LDM
        ctrl_path="",
        sample_way='uniform',
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.sample_way = sample_way

        video_dir = os.path.join(self.dataset_dir, "videos") if 'cosmos_nemo_assets' in dataset_dir else self.dataset_dir
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = sorted(self.video_paths)
       
        log.info(f"{len(self.video_paths)} videos in total")

        self.wrong_number = 0
       
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        self.ctrl_type = ctrl_type
        self.extra_video = extra_video
        self.ctrl_path = ctrl_path
        # import pdb;pdb.set_trace()
        # if extra_video:
        extra_video_dict = defaultdict(list)
        for video_path in os.listdir(video_dir):
            if not video_path.endswith(".mp4"):
                continue
            video_name = video_path.split('/')[-1]
            prefix = '_'.join(video_name.split('_')[:3])
            suffix = video_name.split('_')[-1].replace('.mp4', '')
            extra_video_dict[prefix + '_' + suffix].append(os.path.join(video_dir, video_path))
        self.extra_video_dict = extra_video_dict
    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, video_path):
        # import pdb;pdb.set_trace()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # frame_ids = np.linspace(0, len(vr) - 1).astype(np.int32)
        frame_ids = np.arange(0, len(vr)).tolist()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path):
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        if self.sample_way == 'uniform':
            frames = UniformTemporalSubsample(self.sequence_length)(frames)
        elif self.sample_way == 'random':
            if len(frames) > self.sequence_length:
                start_idx = random.randint(0, len(frames) - self.sequence_length)
                frames = frames[start_idx:start_idx + self.sequence_length]
            else:
                frames = frames[:self.sequence_length]
        else:
            frames = frames[:self.sequence_length]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps


    def _load_control_data(self, sample):
        ctrl_data, fps = self._load_video(sample['ctrl_path'])
        return ctrl_data
        
    def __getitem__(self, index):
        try:
            # import pdb;pdb.set_trace()
            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            video_path = self.video_paths[index]
            video_name = video_path.split('/')[-1]
            t5_embedding_path = os.path.join(
                self.t5_dir,
                os.path.basename(video_path).replace(".mp4", ".pickle"),
            )

            data["video"] = video
            
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
                # "start_frame_id": '0',
            }

            # For LDM, Same dynamic but different apperance
            if self.extra_video:
                prefix = '_'.join(video_name.split('_')[:3])
                suffix = video_name.split('_')[-1].replace('.mp4', '')
                try:
                    extra_video_path = random.choice(self.extra_video_dict[prefix + '_' + suffix])
                    extra_video, _ = self._get_frames(extra_video_path)
                    extra_video = extra_video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                    data["extra_video"] = extra_video
                except:
                    log.info(f'extra_video prefix: {prefix}, suffix:{suffix} not found')

            # Just add these to fit the interface
            try:
                with open(t5_embedding_path, "rb") as f:
                    t5_embedding = pickle.load(f)[0]  # [n_tokens, 1024]
                n_tokens = t5_embedding.shape[0]
                if n_tokens < 512:
                    t5_embedding = np.concatenate(
                        [t5_embedding, np.zeros((512 - n_tokens, 1024), dtype=np.float32)], axis=0
                    )
            except:
                t5_embedding = np.zeros((512, 1024), dtype=np.float32)
                n_tokens = 0
            t5_text_mask = torch.zeros(512, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = fps
            data["image_size"] = torch.tensor([704, 1280, 704, 1280])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280)

            try:
                noise_path = f"{self.dataset_dir}/noise/" + video_name.replace(".mp4", '.npy')
                video_noise = np.load(noise_path)  # [n_tokens, 1024]
                video_noise = torch.from_numpy(video_noise).permute(3, 0, 1, 2)  
                data["video_noise"] = video_noise
            except:
                pass

            if self.ctrl_type is not None:
                if self.extra_video:
                    ctrl_data = extra_video
                else:
                    ctrl_data, fps = self._get_frames(os.path.join(self.ctrl_path, video_name))
                    ctrl_data = ctrl_data.permute(1, 0, 2, 3)
    
                if self.ctrl_type == 'depth':
                    data["control_input_depth"] = ctrl_data
                elif self.ctrl_type == 'recon_image':
                    data["recon_image"] = ctrl_data

            return data
        except Exception:
            # import pdb;pdb.set_trace()
            warnings.warn(
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.samples))]

class InferDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        num_frames,
        video_size,
        ctrl_type=None,
        ctrl_path=""
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """
        # import pdb;pdb.set_trace()
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
       
        video_dir = os.path.join(self.dataset_dir, "videos") if 'cosmos_nemo_assets' in dataset_dir else self.dataset_dir
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        video_names = defaultdict(list)
        for name in os.listdir(video_dir):
            if not name.endswith(".mp4"):
                continue
            # _name = "paper_airplane_"+name.split('_')[2]
            _name = '_'.join(name.split('_')[:-1])
            video_names[_name].append(os.path.join(video_dir, name))
        
        self.video_paths = []
        for name in video_names:
            sorted_videos = sorted(video_names[name], key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.mp4', '')))
            self.video_paths.append(sorted_videos)

        log.info(f"{len(self.video_paths)} videos in total")

        self.wrong_number = 0
       
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        self.ctrl_type = ctrl_type
        self.ctrl_path = ctrl_path
    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, video_path):
        # import pdb;pdb.set_trace()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # frame_ids = np.linspace(0, len(vr) - 1).astype(np.int32)
        frame_ids = np.arange(0, len(vr)).tolist()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path):
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = UniformTemporalSubsample(self.sequence_length)(frames)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps


    def _load_control_data(self, sample):
        ctrl_data, fps = self._load_video(sample['ctrl_path'])
        return ctrl_data
    def __getitem__(self, index):
        try:
            # import pdb;pdb.set_trace()
            data = dict()
            video_paths = self.video_paths[index]
            videos = []
            t5_embeddings = []
            control_input = []
            video_noises = []
            for video_path in video_paths:
                video, fps = self._get_frames(video_path)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                # video_path = self.video_paths[index]
            
                video_name = video_path.split('/')[-1]
                # clip_index = video_name.split('_')[-2]
                prefix = '_'.join(video_name.split('_')[:-1])
                _video_name = f"{prefix}_0.mp4"
                t5_embedding_path = os.path.join(
                    self.t5_dir,
                    # os.path.basename(video_path).replace(".mp4", ".pickle"),
                    _video_name.replace(".mp4", ".pickle"),
                )
                # Just add these to fit the interface
                try:
                    with open(t5_embedding_path, "rb") as f:
                        t5_embedding = pickle.load(f)[0]  # [n_tokens, 1024]
                    n_tokens = t5_embedding.shape[0]
                    if n_tokens < 512:
                        t5_embedding = np.concatenate(
                            [t5_embedding, np.zeros((512 - n_tokens, 1024), dtype=np.float32)], axis=0
                        )
                except:
                    t5_embedding = np.zeros((512, 1024), dtype=np.float32)

                try:
                    noise_path = f"{self.dataset_dir}/noise/" + video_name.replace(".mp4", '.npy')
                    video_noise = np.load(noise_path)  # [n_tokens, 1024]
                    video_noise = torch.from_numpy(video_noise).permute(3, 0, 1, 2)  
                    video_noises.append(video_noise)
                except:
                    pass

                t5_embedding = torch.from_numpy(t5_embedding)
                if self.ctrl_type is not None:
                    ctrl_data, fps = self._get_frames(os.path.join(self.ctrl_path, video_name))
                    ctrl_data = ctrl_data.permute(1, 0, 2, 3)
                    control_input.append(ctrl_data)

                videos.append(video)
                t5_embeddings.append(t5_embedding)
                

            video = torch.stack(videos, dim=0) 
            t5_embedding = torch.stack(t5_embeddings, dim=0)
            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
            }
            if len(video_noises) > 0:
                data["video_noise"] = torch.stack(video_noises, dim=0)
            if len(control_input) > 0:
                control_input = torch.stack(control_input, dim=0)
                data["recon_image"] = control_input
            
            t5_text_mask = torch.zeros(512, dtype=torch.int64)
            # t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = t5_embedding
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = fps
            data["image_size"] = torch.tensor([704, 1280, 704, 1280])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280)            

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.video_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = InferDataset(
        dataset_dir="/opt/tiger/PointVIS/VideoWorld_Cosmos_Predict/datasets/HowTo100M/task_vid/paper_airplane_allstep/boat_complex_0",
        num_frames=93,
        video_size=[480, 832],
        # extra_video=True
    )
    import random
    indices = [random.choice(range(len(dataset))) for _ in range(100)]
    for idx in indices:
        data = dataset[idx]
        import pdb;pdb.set_trace()
        # log.info(
        #     (
        #         f"{idx=} "
        #         f"{data['video'].sum()=}\n"
        #         f"{data['video'].shape=}\n"
        #         f"{data['video_name']=}\n"
        #         f"{data['t5_text_embeddings'].shape=}\n"
        #         "---"
        #     )
        # )