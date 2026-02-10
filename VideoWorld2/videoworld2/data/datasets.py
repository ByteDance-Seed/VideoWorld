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

import pandas as pd
import torch


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, debug_without_randomness=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        if debug_without_randomness:
            self.path = self.path[:1]
            print(f"debug_without_randomness mode: only {len(self.path)} tensors will be used: {self.path}")

        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")

        # Batch dimension will be added when collating the data
        if "prompt_emb" in data:
            data["prompt_emb"]["context"] = data["prompt_emb"]["context"][0]
        if "image_emb" in data:
            data["image_emb"]["clip_feature"] = data["image_emb"]["clip_feature"][0]
            data["image_emb"]["y"] = data["image_emb"]["y"][0]

        return data

    def __len__(self):
        return self.steps_per_epoch
