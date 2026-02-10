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
import os

import pandas as pd
import pytest
import tempfile
import subprocess
import ffmpeg

from video2dataset.data_reader import VideoDataReader


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_data_reader(input_file):
    encode_formats = {"video": "mp4", "audio": "mp3"}
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]

    reading_config = {
        "yt_args": {
            "download_size": 360,
            "download_audio_rate": 12000,
        },
        "timeout": 60,
        "sampler": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        video_data_reader = VideoDataReader(
            encode_formats=encode_formats,
            tmp_dir=tmpdir,
            reading_config=reading_config,
        )
        for i, url in enumerate(url_list):
            key, streams, yt_meta_dict, error_message = video_data_reader((i, url))

            assert len(streams.get("audio", [])) > 0
            assert len(streams.get("video", [])) > 0
