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
import sys
import cv2
import time
import tempfile
import itertools
import platform
import psutil
import json
import yaml
from omegaconf import OmegaConf
from webdataset import WebLoader

from video2dataset import subsamplers
from video2dataset.dataloader import get_video_dataset


# Add this function to gather system information
def gather_system_info():
    cpu_count = os.cpu_count()
    cpu_info = platform.processor()
    gpu_info = [None]

    # TODO: how do you get GPU info on non linux
    if platform.system() == "Linux":
        try:
            gpu_info = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().splitlines()
            gpu_count = len(gpu_info)
        except KeyError:
            pass

    return {
        "platform": platform.system(),
        "cpu_count": cpu_count,
        "cpu_info": cpu_info,
        "gpu_info": gpu_info[0],
        "gpu_count": len(gpu_info),
    }


# Add this function to load and parse the config file
def load_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def create_parameter_grid(params):
    keys, values = zip(*params.items())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]


def make_fake_clips(time, n):
    clip_duration = time / n
    return [(i * clip_duration, (i + 1) * clip_duration) for i in range(n)]


def main(config_file="subsamplers_config.yaml"):
    # Gather system information
    system_info = gather_system_info()

    # Load config
    benchmark_config = load_config(config_file)

    # Dataloader
    ds = get_video_dataset(
        urls=benchmark_config.video_set[0].path,
        batch_size=1,
        decoder_kwargs={},  # load bytes
    )
    ds = WebLoader(ds, batch_size=None, num_workers=12)

    # Initialize all configs
    benchmarks = {}
    for subsampler_config in benchmark_config.subsamplers:
        subsampler_name = subsampler_config.name
        params_grid = create_parameter_grid(OmegaConf.to_container(subsampler_config.parameters, resolve=True)[0])
        benchmarks[subsampler_name] = [
            {
                "subsampler": getattr(subsamplers, subsampler_name)(**cfg),
                "config": cfg,
                "metrics": {"time": 0.0},
            }
            for cfg in params_grid
        ]

    for subsampler, bm_cfgs in benchmarks.items():
        print(f"Benchmarking {subsampler}...")
        size_metrics = {
            "samples": 0,
            "frames": 0,
            "bytes": 0.0,
        }
        for sample in ds:
            # TODO: parallelize this in a safe way i.e. each benchmarker gets certain amount of cores (no interference)
            # TODO: report per-core metrics
            # Update size metrics:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_vid:
                temp_vid.write(sample["mp4"])
                vid = cv2.VideoCapture(temp_vid.name)
                size_metrics["samples"] += 1
                size_metrics["bytes"] += len(sample["mp4"])
                size_metrics["frames"] += int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                seconds = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) / int(vid.get(cv2.CAP_PROP_FPS))

            # TODO: construct streams based on subsampler (maybe we want audio or sth
            streams = {"video": [sample["mp4"]]}
            metadata = {
                "key": sample["__key__"],
                "clips": make_fake_clips(seconds, 5),  # TODO: parameterize this, dense/sparse
            }

            for cfg in bm_cfgs:
                t0 = time.time()
                strm, md, err_msg = cfg["subsampler"](streams, metadata)
                tf = time.time()

                if err_msg is not None:
                    print(err_msg)

                cfg["metrics"]["time"] += tf - t0

                metadata["clips"] = make_fake_clips(seconds, 5)  # gets popped

        # TODO: Normalize metrics for core count
        # Update and normalize cfg metrics
        for cfg in bm_cfgs:
            for m in size_metrics:
                cfg["metrics"][m + "/s"] = size_metrics[m] / cfg["metrics"]["time"]

    # TODO: visualize in nice way - visual repr but also raw numbers in some json or something
    # For now just output JSON
    data = {
        "system_info": system_info,
        "dataset_info": size_metrics,
        "subsamplers": {},
    }

    for ss, bm_cfgs in benchmarks.items():
        cfg_metrics = [{"config": bm["config"], "metrics": bm["metrics"]} for bm in bm_cfgs]
        data["subsamplers"][ss] = sorted(cfg_metrics, key=lambda cfg: cfg["metrics"]["time"])

    with open("subsampler_results.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 1:
        main()
    else:
        print("Usage: python yourscript.py [config_file]")
