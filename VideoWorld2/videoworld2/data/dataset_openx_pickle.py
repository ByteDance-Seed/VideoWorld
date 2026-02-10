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
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.v2 import UniformTemporalSubsample
import warnings
import traceback
from PIL import Image
import random
import io
import json
from tqdm import tqdm
import cv2
from decord import VideoReader, cpu

class PreserveAspectRatioResize:
    def __init__(self, size, antialias=True):
       
        self.size = size
        self.antialias = antialias
    
    def __call__(self, img):

        _, h, w = img.shape
        target_h, target_w = self.size
        

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        

        resized = T.Resize((new_h, new_w), antialias=self.antialias)(img)
        
       
        result = torch.zeros((3, target_h, target_w), dtype=img.dtype)
        

        y_start = (target_h - new_h) // 2
        x_start = (target_w - new_w) // 2
        

        if new_h <= target_h and new_w <= target_w:
            result[:, y_start:y_start+new_h, x_start:x_start+new_w] = resized

        else:
 
            y_crop_start = (new_h - target_h) // 2
            x_crop_start = (new_w - target_w) // 2
            cropped = resized[:, y_crop_start:y_crop_start+target_h, x_crop_start:x_crop_start+target_w]
            result = cropped
        
        return result

class RandomResizeStrategy:
    def __init__(self, size, antialias=True, aspect_ratio_prob=0.5):

        self.size = size
        self.antialias = antialias
        self.aspect_ratio_prob = aspect_ratio_prob
        self.preserve_aspect_ratio_transform = PreserveAspectRatioResize(size, antialias)
        self.direct_resize_transform = T.Resize(size, antialias=True)
    
    def __call__(self, img, keep_aspect_ratio=True):

        return self.direct_resize_transform(img)



def save_video_as_gif(video_tensor, save_path, fps=24):

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    images = []

    video_np = video_tensor.permute(1, 2, 3, 0).numpy()
    
    if video_np.max() <= 1.1:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = video_np.astype(np.uint8)
    
    for frame in video_np:

        img = Image.fromarray(frame)
        images.append(img)
    

    duration = int(1000 / fps)  
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  
    )
    
    print(f"GIF has been saved to: {save_path}")


def save_video_as_frames(video_tensor, save_dir, frame_prefix="frame_", format="png"):

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    video_np = video_tensor.permute(1, 2, 3, 0).numpy()
    

    if video_np.max() <= 1.1:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = video_np.astype(np.uint8)
    

    total_frames = video_np.shape[0]
    for i, frame in enumerate(video_np):

        img = Image.fromarray(frame)

        filename = f"{frame_prefix}{i:04d}.{format}"
        file_path = os.path.join(save_dir, filename)

        img.save(file_path)
    
    print(f"The total number of frames ({total_frames}) has been saved to the directory: {save_dir}")





def generate_folder_frame_meta(dataset_dir, output_meta_path='./folder_frame_meta.json', image_keys=None, sample_size=50):

    if image_keys is None:
        image_keys = ['image', 'rgb', 'front_rgb', 'hand_image', 'image_1', 'rgb_static', 'agentview_rgb']
    
    folder_frame_meta = {
        'dataset_dir': dataset_dir,
        'timestamp': str(np.datetime64('now')),
        'subfolders': {}
    }
    
    for subfolder in tqdm(os.listdir(dataset_dir)):
        subfolder_path = os.path.join(dataset_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        print(f"Counting frames in the subfolder {subfolder}...")
        
        pickle_files = []
        for file in os.listdir(subfolder_path):
            if file.endswith('.pickle') or file.endswith('.pkl'):
                pickle_files.append(os.path.join(subfolder_path, file))
        
        total_files = len(pickle_files)
        
        if total_files == 0:
            folder_frame_meta['subfolders'][subfolder] = {
                'total_files': 0,
                'estimated_total_frames': 0,
                'frames_per_file': 0
            }
            continue
        
        sampled_files = random.sample(pickle_files, min(sample_size, total_files)) if total_files > 0 else []
        total_sampled_frames = 0
        
        for file_path in sampled_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                steps_key = None
                if 'steps' in data:
                    steps_key = 'steps'
                elif 'step' in data:
                    steps_key = 'step'
                else:
                    continue
                
                steps = data[steps_key]
                if not steps or not isinstance(steps, list):
                    continue
                
                file_frames = 0
                for step in steps:
                    if not isinstance(step, dict) or 'observation' not in step:
                        continue
                    
                    observation = step['observation']
                    if not isinstance(observation, dict):
                        continue
                    
        
                    has_valid_image = False
                    for key in image_keys:
                        if key in observation:
                            image_data = observation[key]
                            if isinstance(image_data, bytes) or isinstance(image_data, np.ndarray):
                                has_valid_image = True
                                break
                    
                    if has_valid_image:
                        file_frames += 1
                
                total_sampled_frames += file_frames
            except Exception as e:
                warnings.warn(f"Error when processing {file_path}: {str(e)}")
                continue
        
        if len(sampled_files) > 0:
            avg_frames_per_file = total_sampled_frames / len(sampled_files)
            estimated_total_frames = avg_frames_per_file * total_files
        else:
            avg_frames_per_file = 100
            estimated_total_frames = 100 * total_files
        

        folder_frame_meta['subfolders'][subfolder] = {
            'total_files': total_files,
            'sampled_files': len(sampled_files),
            'frames_per_file': avg_frames_per_file,
            'estimated_total_frames': estimated_total_frames
        }
    
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(folder_frame_meta, f, ensure_ascii=False, indent=2)
    
    print(f"The subfolder frame count statistics have been saved to: {output_meta_path}")
    return folder_frame_meta


class EnhancedOpenXPickleDataset(Dataset):
    def __init__(
        self,
        dataset_dir="datasets/openx_untar/",
        num_frames=93,
        video_size=[256, 256],
        subfolders=None, 
        ctrl_type=None,
        extra_video=False,
        ctrl_path="",
        aspect_ratio_prob=0.5,  
        image_keys=None,  
        meta_path="datasets/openx_videocraft_meta.json",  
        use_frame_based_sampling=True,  
        cache_path=None,  
        use_cache=True,  
        frame_interval=1,  
        is_infer_mode=False, 
        fps_variation=True  
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.video_size = video_size
        self.subfolders = subfolders
        self.ctrl_type = ctrl_type
        self.extra_video = extra_video
        self.ctrl_path = ctrl_path
        self.aspect_ratio_prob = aspect_ratio_prob
        self.use_frame_based_sampling = use_frame_based_sampling
        self.use_cache = use_cache
        self.frame_interval = frame_interval  
        self.is_infer_mode = is_infer_mode
        self.fps_variation = fps_variation  
        
        if cache_path is None:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            subfolder_suffix = "_" + "_" .join(subfolders) if subfolders else "all"
            cache_filename = f"pickle_paths_cache_{dataset_name}_{subfolder_suffix}.json"
            cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_filename)
        self.cache_path = cache_path
        
  
        self.image_keys = image_keys if image_keys else [
            'image', 'rgb', 'front_rgb', 'hand_image', 'image_1', 'rgb_static', 'agentview_rgb'
        ]
        

        if meta_path is None:
            meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folder_frame_meta.json')
        

        if os.path.exists(meta_path):
            print(f"Loading meta dictionary: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.folder_meta = json.load(f)
        else:
            print(f"Meta dictionary does not exist, generating: {meta_path}")
            self.folder_meta = generate_folder_frame_meta(
                dataset_dir, 
                output_meta_path=meta_path, 
                image_keys=self.image_keys
            )
        

        self.pickle_paths = []
        self.pickle_paths_by_folder = {}

        if self.use_cache and os.path.exists(self.cache_path):
            print(f"Loading file paths from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                    if self.subfolders and 'pickle_paths_by_folder' in cache_data:
                        self.pickle_paths_by_folder = {}
                        self.pickle_paths = []
                        
             
                        for subfolder in self.subfolders:
            
                            if subfolder in cache_data['pickle_paths_by_folder']:
                     
                                folder_files = cache_data['pickle_paths_by_folder'][subfolder]
                                self.pickle_paths_by_folder[subfolder] = folder_files
                                self.pickle_paths.extend(folder_files)
                                print(f"Loaded {len(folder_files)} file paths from cache for subfolder {subfolder}")
                    else:
                        
                        self.pickle_paths = cache_data['pickle_paths']
                        self.pickle_paths_by_folder = cache_data.get('pickle_paths_by_folder', {})
                        print(f"Successfully loaded {len(self.pickle_paths)} file paths from cache")
            except Exception as e:
                print(f"Failed to load from cache: {str(e)}")
                print("Re-collecting file paths")
                self._collect_pickle_files()
        else:

            self._collect_pickle_files()
        
 
        self.preprocess = T.Compose([
            T.ToTensor(),  
            RandomResizeStrategy(tuple(video_size), antialias=True, aspect_ratio_prob=aspect_ratio_prob)
        ])
        

        self.folder_probs = None
        self.folder_list = None
        if self.use_frame_based_sampling:
            self._compute_sampling_probs()
        
        print(f"{len(self.pickle_paths)} pickle files loaded")
    
    def _load_single_frame(self, pkl_data, frame_idx):
        
    
        if 'obs_images' in pkl_data:
            
            video = pkl_data['obs_images'].permute(1, 0, 2, 3)  # [C, T_original, H, W]
            
            video = video.permute(1, 0, 2, 3)  # [T_original, C, H, W]
            
           
            for frame_tensor in video:
    
                frame_np = frame_tensor.permute(1, 2, 0).numpy()
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
                
            image = Image.fromarray(frame_np)
        else:
            
            steps_key = None
            if 'steps' in pkl_data:
                steps_key = 'steps'
            elif 'step' in pkl_data:
                steps_key = 'step'
            
            empty_image = Image.new('RGB', (self.video_size[1], self.video_size[0]), color='white')
            if steps_key:
                steps = pkl_data[steps_key]
                if steps and isinstance(steps, list):
       
                    step = steps[frame_idx]
                    
                    try:
      
                        if not isinstance(step, dict) or 'observation' not in step:
                            return empty_image
                        
                        observation = step['observation']
                        if not isinstance(observation, dict):
                            return empty_image
                        
     
                        image_bytes = None
                        for key in self.image_keys:
                            if key in observation:
                                image_bytes = observation[key]
                                break
                        
                        if image_bytes is None:
            
                            return empty_image
                        
       
                        try:
                            if isinstance(image_bytes, bytes):
                               
                                image = Image.open(io.BytesIO(image_bytes))
                            elif isinstance(image_bytes, np.ndarray):
                                
                                if image_bytes.max() <= 1.0:
                                    image_bytes = (image_bytes * 255).astype(np.uint8)
                                else:
                                    image_bytes = image_bytes.astype(np.uint8)
                                image = Image.fromarray(image_bytes)
                            else:
                                return empty_image
                            
                        except Exception:
                            return empty_image
                    except Exception:
                        return empty_image
      
        return image

    def _collect_pickle_files(self):
        
        for subfolder in tqdm(os.listdir(self.dataset_dir)):
            print(f"Load {subfolder}")
            subfolder_path = os.path.join(self.dataset_dir, subfolder)
            

            if not os.path.isdir(subfolder_path):
                continue
            
            if self.subfolders is not None and subfolder not in self.subfolders:
                continue
            
           
            folder_files = []
            for file in os.listdir(subfolder_path):
                if file.endswith('.pickle') or file.endswith('.pkl'):
                    file_path = os.path.join(subfolder_path, file)
                    self.pickle_paths.append(file_path)
                    folder_files.append(file_path)
            

            if folder_files:
                self.pickle_paths_by_folder[subfolder] = folder_files
        

        if self.use_cache:
            try:

                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                

                cache_data = {
                    'pickle_paths': self.pickle_paths,
                    'pickle_paths_by_folder': self.pickle_paths_by_folder,
                    'timestamp': str(np.datetime64('now')),
                    'dataset_dir': self.dataset_dir,
                    'subfolders': self.subfolders
                }
                
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
                print(f"File paths have been saved to cache: {self.cache_path}")
            except Exception as e:
                print(f"Failed to save cache: {str(e)}")

    def _compute_sampling_probs(self):

        if not hasattr(self, 'folder_meta') or 'subfolders' not in self.folder_meta:
            print("Warning: No valid folder_meta data available, unable to perform frame-based sampling")
            self.use_frame_based_sampling = False
            return

        self.folder_list = []
        folder_frames = []
        
        for subfolder, folder_files in self.pickle_paths_by_folder.items():
            if folder_files and subfolder in self.folder_meta['subfolders']:
                self.folder_list.append(subfolder)
                folder_frames.append(self.folder_meta['subfolders'][subfolder]['estimated_total_frames'])
        
  
        if folder_frames:
            total_frames = sum(folder_frames)
            if total_frames > 0:
                self.folder_probs = [frames / total_frames for frames in folder_frames]
                return
        
        print("Warning: Unable to compute frame-based sampling probabilities, falling back to uniform distribution")
        self.folder_probs = None
    
    def __len__(self):
        return len(self.pickle_paths)
    
    

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
        selected_folder = ''
        try:
            data = dict()
            if self.use_frame_based_sampling and self.folder_list and self.folder_probs:
                np.random.seed(index)
                random.seed(index)
                selected_folder = np.random.choice(self.folder_list, p=self.folder_probs)
                selected_file = random.choice(self.pickle_paths_by_folder[selected_folder])
            else:
                selected_file = self.pickle_paths[index]
            
            frames = []
            is_mp4 = False
            if selected_file.endswith('.mp4'):
                vr = VideoReader(selected_file, ctx=cpu(0), num_threads=2)
                frame_ids = np.arange(0, len(vr)).tolist()
                frame_ids = frame_ids[::2]
                vr.seek(0)
                frame_data = vr.get_batch(frame_ids).asnumpy()
                for frame in frame_data:
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)
                is_mp4 = True
                original_length = len(frames)
            else:
                with open(selected_file, 'rb') as f:
                    pkl_data = pickle.load(f)

                if 'obs_images' in pkl_data:
                    original_length = pkl_data['obs_images'].shape[0]
                else:
                    original_length = len(pkl_data['steps'])

            frames = [i for i in range(original_length)] if not is_mp4 else frames
            if self.is_infer_mode:
                all_clips = []
                all_t5_embeddings = []
                all_t5_masks = []
                
              
                num_clips = 0
                current_start = 0
                while True:
                    current_end = current_start + self.sequence_length
                    num_clips += 1
                   
                    if current_end - 1 < original_length:
                        current_start = current_end - 1  
                    else:
                        break
                
                clip_ranges = []
                current_start = 0
                for _ in range(num_clips):
                    current_end = current_start + self.sequence_length
                    clip_ranges.append((current_start, current_end))
                    current_start = current_end - 1 
                for start_idx, end_idx in clip_ranges:
                    actual_end = min(end_idx, original_length)
                    clip_frames = frames[start_idx:actual_end]
                    
                    if len(clip_frames) < self.sequence_length:
                        pad_length = self.sequence_length - len(clip_frames)
                        last_frame = clip_frames[-1] if clip_frames else Image.new('RGB', (self.video_size[1], self.video_size[0]), color='white')
                        clip_frames += [last_frame for _ in range(pad_length)]
                    
                    processed_clip = []
                    for frame in clip_frames:
                        processed = self.preprocess(frame)
                        processed_clip.append(processed)
                    
                    clip_video = torch.stack(processed_clip).permute(1, 0, 2, 3)
                    if clip_video.max() < 1.5:
                        clip_video = torch.clamp(clip_video, 0, 1)
                    clip_video = (clip_video * 255).to(torch.uint8)
                    
                    all_clips.append(clip_video)
                
                video = torch.stack(all_clips)
                
        
                instruction = ""
                
                t5_embedding = np.zeros((512, 1024), dtype=np.float32)
                n_tokens = 0  
                t5_text_mask = torch.zeros(512, dtype=torch.int64)
                t5_text_mask[:n_tokens] = 1
                
                t5_embeddings = [torch.from_numpy(t5_embedding) for _ in range(num_clips)]
                t5_masks = [t5_text_mask.clone() for _ in range(num_clips)]
                
                data["t5_text_embeddings"] = torch.stack(t5_embeddings)
                data["t5_text_mask"] = torch.stack(t5_masks)
            else:
                min_start_idx = 0  
                max_start_idx = max(0, original_length - 30) 
                
                if min_start_idx <= max_start_idx:
                    start_idx = random.randint(min_start_idx, max_start_idx)
                else:
                    start_idx = 0 
                
                
                # start_idx = 0
                # import pdb;pdb.set_trace()
                if is_mp4 or original_length < self.sequence_length:
                    start_idx = 0
                if is_mp4:
                    print(selected_file, start_idx)

                sampled_frames = []

                if self.fps_variation and self.sequence_length > 1:
                    fps_scale = random.uniform(0.25, 1.0)
                    
                    if fps_scale <= 0.5:
                        remaining_frames = self.sequence_length - 1
                        group_size = int(1 / fps_scale)
                        num_groups = (remaining_frames + group_size - 1) // group_size
                        
                        current_idx = start_idx
                        if current_idx < original_length:
                            sampled_frames.append(frames[current_idx])
                            current_idx += self.frame_interval
                        
                       
                        for i in range(num_groups):
                            if current_idx < original_length:
                               
                                current_frame = frames[current_idx]
                                frames_to_add = min(group_size, self.sequence_length - len(sampled_frames))
                                sampled_frames.extend([current_frame for _ in range(frames_to_add)])
                                current_idx += self.frame_interval
                    else:
                
                        adjusted_interval = max(1, int(self.frame_interval / fps_scale))
                        current_idx = start_idx
                        while len(sampled_frames) < self.sequence_length and current_idx < original_length:
                            sampled_frames.append(frames[current_idx])
                            current_idx += adjusted_interval

                else:
                    
                    current_idx = start_idx
                    while len(sampled_frames) < self.sequence_length and current_idx < original_length:
                        sampled_frames.append(frames[current_idx])
                        current_idx += self.frame_interval
                
                if len(sampled_frames) < self.sequence_length:
                    pad_length = self.sequence_length - len(sampled_frames)
                    last_frame = frames[-1:] if frames else [Image.new('RGB', (self.video_size[1], self.video_size[0]), color='white')]
                    for _ in range(pad_length):
                        sampled_frames.append(last_frame[0])
                
                if self.sequence_length == 2 and len(sampled_frames) >= 2:
                    first_frame = sampled_frames[0]
                    second_frame = sampled_frames[1]
                    sampled_frames = [first_frame] + [second_frame for _ in range(4)]
                
                if not is_mp4:
                    _sampled_frames = sampled_frames
                    sampled_frames = []
                    for idx in _sampled_frames:
                        _frame = self._load_single_frame(pkl_data, idx)
                        sampled_frames.append(_frame)


                processed_frames = []
                keep_aspect_ratio = random.random() < 0.5
                for frame in sampled_frames:
                    processed = self.preprocess(frame)
                    processed_frames.append(processed)
                

                video = torch.stack(processed_frames).permute(1, 0, 2, 3)
                if video.max() < 1.5:
                    video = torch.clamp(video, 0, 1)
                video = (video * 255).to(torch.uint8)
            data['selected_folder'] = selected_folder 
            data["video"] = video
            data["video_name"] = {
                "video_path": selected_file,
                "start_idx": start_idx,
                "t5_embedding_path": selected_file.replace('.pkl', '.pickle').replace('.pickle', '.pickle'),
            }
            

            instruction = ""
            
            t5_embedding = np.zeros((512, 1024), dtype=np.float32)
            n_tokens = 0  
            t5_text_mask = torch.zeros(512, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1
            
            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = 24 
            data["image_size"] = torch.tensor([self.video_size[0], self.video_size[1], self.video_size[0], self.video_size[1]])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, self.video_size[0], self.video_size[1])
            

            # data["robot_and_gripper"] = pkl_data.get('robot_and_gripper', [])
            data["instruction"] = instruction
            
  
            if self.ctrl_type is not None:
                pass
            
            return data
            
        except Exception as e:
            warnings.warn(f"Invalid data encountered: {self.pickle_paths[index] if index < len(self.pickle_paths) else 'unknown path'}. Skipped.")
            warnings.warn(f"Error: {str(e)}")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            
            return self[np.random.randint(len(self))]


if __name__ == "__main__":  

    # import pdb;pdb.set_trace()
    dataset = EnhancedOpenXPickleDataset(
        video_size=[240, 416],
        aspect_ratio_prob=0.5, 
        num_frames=93,
        use_frame_based_sampling=True, 
        use_cache=True,
        cache_path="datasets/openx_videocraft_cache.json",
        meta_path="datasets/openx_videocraft_meta.json",
        frame_interval=1,
        subfolders=['bridge'],
        is_infer_mode=False,
        fps_variation=False        

    )
    
    print(f"Dataset size: {len(dataset)}")
    for i in range(100):
        sample = dataset[i]
        # import pdb;pdb.set_trace()
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Video shape: {sample['video'].shape}")
        print(f"Instruction: {sample.get('instruction', 'No instruction')}")
        
        if sample['video'].ndim == 5:
            for ci, vid in enumerate(sample['video']):

                save_path = f"datasets/openx_out/enhanced_output_video_{i}_part{ci}.gif"
                save_video_as_gif(vid, save_path, fps=16)

        else:
            save_path = f"datasets/openx_out/enhanced_output_video_clip_{i}.gif"
            save_video_as_gif(sample['video'], save_path, fps=16)
            frames_dir = f"datasets/openx_out/enhanced_output_frames_{i}"
            save_video_as_frames(sample['video'], frames_dir, frame_prefix=f"video_{i}_frame_")

            print(sample['selected_folder'], sample['video'].max(), sample['video'].min())


