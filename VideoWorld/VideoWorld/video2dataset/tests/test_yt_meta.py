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
from video2dataset.data_reader import get_yt_meta


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_meta(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "get_info": True,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert type(yt_meta_dict["info"]) == dict
        assert "id" in yt_meta_dict["info"].keys()
        assert "title" in yt_meta_dict["info"].keys()


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_no_meta(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert yt_meta_dict["info"] == None


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_subtitles(input_file):
    yt_metadata_args = {
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert type(yt_meta_dict["subtitles"]) == dict
        assert type(yt_meta_dict["subtitles"]["en"]) == list
        assert type(yt_meta_dict["subtitles"]["en"][0]) == dict


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_no_subtitles(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert yt_meta_dict["subtitles"] == None
