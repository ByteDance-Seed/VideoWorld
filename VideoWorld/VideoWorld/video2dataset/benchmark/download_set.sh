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
#!/bin/bash

video2dataset --url_list="video_sets/mp4.parquet" \
        --input_format="parquet" \
        --output_folder="dataset/mp4" \
        --output-format="webdataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --enable_wandb=False \
        --video_size=360 \
        --number_sample_per_shard=10 \
        --processes_count 10  \
