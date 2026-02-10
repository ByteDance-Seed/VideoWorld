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
"""
Test for dataloader with and without return all
"""
import pytest
import webdataset as wds
from video2dataset.dataloader import get_video_dataset

# Benchmark videos are the WebVid validation split (5000 videos)
SHARDS = "tests/test_files/return_all_test.tar"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_return_all(batch_size):
    decoder_kwargs = {"n_frames": 10, "fps": None, "num_threads": 1}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=None,
        crop_size=None,
        return_always=True,
        keys_to_remove=["m4a"],
        repeat=1,
        handler=wds.warn_and_continue,
    )
    dl = wds.WebLoader(dset, batch_size=None, num_workers=0)

    expected_keys = {
        "000000014": True,
        "000000025": True,
        "000000356": True,
        "0000008_00001": False,
        "0000030_00005": False,
        "0000038_00003": False,
    }
    received_keys = []
    for samp in dl:
        for i in range(len(samp["__key__"])):
            key = samp["__key__"][i]
            received_keys.append(key)
            assert expected_keys[key] == samp["__corrupted__"][i].item()

    assert set(received_keys) == set(expected_keys)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_default(batch_size):
    decoder_kwargs = {"n_frames": 10, "fps": None, "num_threads": 1}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=None,
        crop_size=None,
        return_always=False,
        keys_to_remove=["m4a"],
        repeat=1,
        handler=wds.warn_and_continue,
    )
    dl = wds.WebLoader(dset, batch_size=None, num_workers=0)

    expected_keys = ["0000008_00001", "0000030_00005", "0000038_00003"]

    received_keys = []
    for samp in dl:
        assert "__corrupted__" not in samp
        for i in range(len(samp["__key__"])):
            key = samp["__key__"][i]
            received_keys.append(key)

    assert set(received_keys) == set(expected_keys)


@pytest.mark.parametrize(
    "batch_size, expected_keys",
    [(1, ["0000008_00001", "0000030_00005", "0000038_00003"]), (2, ["0000008_00001", "0000030_00005"])],
)
def test_drop_last(batch_size, expected_keys):
    decoder_kwargs = {"n_frames": 10, "fps": None, "num_threads": 1}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=None,
        crop_size=None,
        return_always=False,
        keys_to_remove=["m4a"],
        drop_last=True,
        repeat=1,
        handler=wds.warn_and_continue,
    )
    dl = wds.WebLoader(dset, batch_size=None, num_workers=0)

    received_keys = []
    for samp in dl:
        assert "__corrupted__" not in samp
        for i in range(len(samp["__key__"])):
            key = samp["__key__"][i]
            received_keys.append(key)

    assert set(received_keys) == set(expected_keys)
