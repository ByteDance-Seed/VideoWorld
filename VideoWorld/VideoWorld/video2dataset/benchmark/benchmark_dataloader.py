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
Benchmark dataloader speed
"""
import time

# from video2dataset.dataloader import get_bytes_dataloader
from video2dataset.dataloader import get_video_dataset

# Benchmark videos are the WebVid validation split (5000 videos)
SHARDS = "examples/dataset/{00000..00004}.tar"


def benchmark_train_dl(num_frames, num_workers, bs=1, num_threads=4, resize_size=None, crop_size=None):
    from argparse import Namespace
    from webdataset import WebLoader

    decoder_kwargs = {"n_frames": num_frames, "fps": None, "num_threads": num_threads}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=bs,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
    )
    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
    count = 0
    t0 = time.time()
    for samp in dl:
        count += 1
    tf = time.time()
    return count / (tf - t0)


if __name__ == "__main__":
    # print("Benchmarking bytes dataloader...")
    # print(benchmark_bytes_dl())

    print("# benchmarking without resizing")
    for nf in [8, 16, 32]:
        for nw in [6, 8, 10, 12]:
            print(f"Benchmarking train dataloader with {nf} decoded frames and {nw} dl workers...")
            throughput = benchmark_train_dl(num_frames=nf, num_workers=nw)
            print(f"Got {throughput} samples/sec")

    print("# benchmarking with resizing and centercropping")
    for nf in [8, 16, 32]:
        for nw in [6, 8, 10, 12]:
            print(f"Benchmarking train dataloader with {nf} decoded frames and {nw} dl workers...")
            throughput = benchmark_train_dl(num_frames=nf, num_workers=nw, resize_size=256, crop_size=256)
            print(f"Got {throughput} samples/sec")
