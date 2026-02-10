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
IMAGE_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1024": {"1:1": (1024, 1024), "4:3": (1168, 880), "3:4": (880, 1168), "16:9": (1360, 768), "9:16": (768, 1360)},
    "720": {"1:1": (960, 960), "4:3": (960, 704), "3:4": (704, 960), "16:9": (1280, 704), "9:16": (704, 1280)},
    "480": {"1:1": (480, 480), "4:3": (640, 480), "3:4": (480, 640), "16:9": (768, 432), "9:16": (432, 768)},
}

# "width:height"
VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "720": {"1:1": (960, 960), "4:3": (960, 704), "3:4": (704, 960), "16:9": (1280, 704), "9:16": (704, 1280)},
    "480": {"1:1": (480, 480), "4:3": (640, 480), "3:4": (480, 640), "16:9": (768, 432), "9:16": (432, 768)},
}
