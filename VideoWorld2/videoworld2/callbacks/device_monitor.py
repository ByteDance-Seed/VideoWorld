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
from typing import Any, Dict, List, Tuple

import pandas as pd
import psutil
import pynvml
import torch

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log


def log_prof_data(
    data_list: List[Dict[str, Any]],
    iteration: int,
) -> Tuple[pd.DataFrame]:
    # Create a table to log data with rank information
    columns = ["iteration", "rank"] + list(data_list[0].keys())
    data = []

    # Initialize dictionaries to store min and max values for each metric
    min_values = {key: float("inf") for key in columns[2:]}
    max_values = {key: float("-inf") for key in columns[2:]}
    sum_values = {key: 0.0 for key in columns[2:]}

    count = 0

    for _rank, prof_data in enumerate(data_list):
        row = [iteration, _rank] + [prof_data[key] for key in columns[2:]]
        data.append(row)
        count += 1

        # Update min, max, and sum values
        for key in columns[2:]:
            min_values[key] = min(min_values[key], prof_data[key])
            max_values[key] = max(max_values[key], prof_data[key])
            sum_values[key] += prof_data[key]

    # Calculate average values
    avg_values = {key: sum_values[key] / count for key in columns[2:]}

    df = pd.DataFrame(data, columns=columns)
    summary_df = pd.DataFrame({"Avg": avg_values, "Max": max_values, "Min": min_values})
    return df, summary_df


class DeviceMonitor(EveryN):
    """
    A callback to monitor device (CPU/GPU) usage and log it at regular intervals.

    Args:
        every_n (int, optional): The frequency at which the callback is invoked. Defaults to 200.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        log_memory_detail (bool, optional): Whether to log the memory detail. Defaults to True.
    """

    def __init__(
        self,
        every_n: int = 200,
        step_size: int = 1,
        log_memory_detail: bool = True,
    ):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.log_memory_detail = log_memory_detail

    def on_train_start(self, model, iteration=0):
        torch.cuda.reset_peak_memory_stats()
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"{self.name} callback: local_dir: {self.local_dir}")

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        cur_process = psutil.Process(os.getpid())
        cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
        cpu_mem_gb = cpu_memory_usage / (1024**3)

        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        temp = torch.cuda.temperature()
        try:
            power = torch.cuda.power_draw()
        except Exception as e:
            log.warning(f"Failed to get power draw with error {e}")
            power = 0
        util = torch.cuda.utilization()
        clock = torch.cuda.clock_rate()

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
        nvml_free_gpu_mem_gb = memory_info.free / (1024**3)

        prof_data = {
            "cpu_mem_gb": cpu_mem_gb,
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "peak_gpu_mem_reserved_gb": peak_gpu_mem_reserved_gb,
            "nvml_used_gpu_mem_gb": nvml_used_gpu_mem_gb,
            "nvml_free_gpu_mem_gb": nvml_free_gpu_mem_gb,
            "temp": temp,
            "power": power,
            "util": util,
            "clock": clock,
        }

        data_list = [prof_data] * self.world_size
        # this is blocking by default
        if self.world_size > 1:
            torch.distributed.all_gather_object(data_list, prof_data)
            torch.distributed.barrier()

        df, summary_df = log_prof_data(data_list, iteration)
        if self.rank == 0:
            log.info(f"{self.name} Stats:\n{summary_df.to_string()}")
            if self.log_memory_detail:
                memory_stats = torch.cuda.memory_stats()

        torch.cuda.reset_peak_memory_stats()
