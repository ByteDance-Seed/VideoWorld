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
import sys
import random
import torch
from tqdm import tqdm

from dataset_openx_pickle import EnhancedOpenXPickleDataset, save_video_as_gif

def generate_subfolder_gifs():

    base_save_dir = 'datasets/openx_out'
    os.makedirs(base_save_dir, exist_ok=True)
    
    dataset = EnhancedOpenXPickleDataset(
        video_size=[256, 256],
        aspect_ratio_prob=0.5,
        num_frames=93,
        use_frame_based_sampling=False,  
        use_cache=True,
        cache_path="datasets/openx_videocraft_cache.json",
        subfolders=['bridge']
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Subfolders found: {list(dataset.pickle_paths_by_folder.keys())}")
    

    for subfolder, file_paths in dataset.pickle_paths_by_folder.items():
        print(f"Processing subfolder: {subfolder}")
        

        subfolder_save_dir = os.path.join(base_save_dir, subfolder)
        os.makedirs(subfolder_save_dir, exist_ok=True)
        

        num_files_to_sample = min(50, len(file_paths))
        if num_files_to_sample == 0:
            print(f"No files found in subfolder {subfolder}, skipping...")
            continue
        

        sampled_files = random.sample(file_paths, num_files_to_sample)
 
        for i, file_path in enumerate(tqdm(sampled_files, desc=f"Generating GIFs for {subfolder}")):
            try:
      
                file_index = dataset.pickle_paths.index(file_path)
                
          
                sample = dataset[file_index]
                
       
                file_name = os.path.basename(file_path).split('.')[0]
                save_path = os.path.join(subfolder_save_dir, f"{file_name}_gif_{i}.gif")
                
       
                save_video_as_gif(sample['video'], save_path, fps=16)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
    
    print(f"All GIFs have been generated and saved to {base_save_dir}")


if __name__ == "__main__":
    generate_subfolder_gifs()