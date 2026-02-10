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
"""Create dataset from video links and metadata."""
import os
import sys
import signal
import fire
import fsspec

from omegaconf import OmegaConf
from typing import List, Optional, Any
import numpy as np  # pylint: disable=unused-import

from video2dataset.logger import LoggerProcess
from video2dataset.data_writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from video2dataset.input_sharder import InputSharder
from video2dataset.output_sharder import OutputSharder
from video2dataset.distributor import (
    no_distributor,
    multiprocessing_distributor,
    pyspark_distributor,
    SlurmDistributor,
    SlurmShardSampler,
)
from video2dataset.workers import DownloadWorker, SubsetWorker, OpticalFlowWorker, CaptionWorker, WhisperWorker
from video2dataset.configs import CONFIGS
from video2dataset.types import EncodeFormats


def identity(x):
    return x


# pylint: disable=unused-argument
# pylint: disable=eval-used
# pylint: disable=broad-except
def video2dataset(
    url_list: str,
    output_folder: str = "dataset",
    output_format: str = "files",
    input_format: str = "csv",
    encode_formats: Optional[EncodeFormats] = None,
    stage: str = "download",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    save_additional_columns: Optional[List[str]] = None,
    enable_wandb: bool = False,
    wandb_project: str = "video2dataset",
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    tmp_dir: str = "/tmp",
    config: Any = "default",
):
    """
    Create datasets from video/audio links

    Args:
    url_list: list of input urls - can be any of the supported input formats
        (csv, parquet, braceexpand tar paths etc.)
    output_folder: Desired location of output dataset
    output_format: Format of output dataset, can be
        - files, samples saved in subdirectory for each shard (useful for debugging)
        - webdataset, samples saved in tars (useful for efficient loading)
        - parquet, sampels saved in parquet (as bytes)
        - tfrecord, samples saved in tfrecord (as bytes)
        - dummy, does not save (useful for benchmarks)
    input_format: Format of the input, can be
        - txt, text file with a url in each line
        - csv, csv file with urls, (and captions + metadata)
        - tsv, tsv - || -
        - tsv.gz, - || - but compressed gzip
        - json, loads urls and metadata as json
        - parquet, loads urls and metadata as parquet
        - webdataset, usually braceexpand format of mutliple paths to tars to re-process
    encode_formats: Dict that specifies what extension each modality should use
        f.e. {"video": "mp4", "audio": "m4a"}
    stage: String that tells video2dataset what stage of processing is being performed. Can be
        WARNING: To be depracated soon (this information should be deduced based on config)
        - download, when input is some tabular format and data must be downloaded first
        - subset, tar files are already written and we would like to re-process (input_format == "webdataset")
        - optical_flow, tar files are written and we would like to compute optical_flow and save to md shards
        - caption, tar files are written and we would like to generate caption and save to md shards
    url_col: Column in input (if has columns) that contains the url
    caption_col: Column in input (if has columns) that contains captions (to be written as txt)
    clip_col: Column in input (if has columns) that contains timeframes of clips for how to split video
    save_additional_columns: List of column names to save to json component of a sample
    enable_wandb: Whether or not to log info to wandb
    wandb_project: Name of wandb project to log runs to
    incremental_mode: Decides how to handle restarting, Can be
        - incremental, checks which shards are done and skips those
        - overwrite, deletes and reprocesses shards as it goes
    max_shard_retry: Maximum amount of attempts to retry a failed shard
    tmp_dir: Path to temporary directory on your file system
    config: Path to your config of choice or the config itself (more info on configs in API doc)
    """
    local_args = dict(locals())
    if isinstance(config, str):
        config = CONFIGS[config] if config in CONFIGS else OmegaConf.load(config)
        config = OmegaConf.to_container(config)
    for arg_type in ["subsampling", "reading", "storage", "distribution"]:
        assert arg_type in config

    if config["reading"]["sampler"] is None:
        config["reading"]["sampler"] = identity

    called_from_slurm = "CALLED_FROM_SLURM" in os.environ
    if called_from_slurm:
        global_task_id = int(os.environ["GLOBAL_RANK"])
        num_tasks = (
            config["distribution"]["distributor_args"]["n_nodes"]
            * config["distribution"]["distributor_args"]["tasks_per_node"]
        )
        config["reading"]["sampler"] = SlurmShardSampler(global_task_id=global_task_id, num_tasks=num_tasks)
        config["distribution"]["distributor"] = "multiprocessing"

        # Only log from master
        enable_wandb = enable_wandb and (global_task_id == 0)

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
        if config["reading"]["yt_args"]["yt_metadata_args"] is None:
            config["reading"]["yt_args"]["yt_metadata_args"] = {}
        if not config["reading"]["yt_args"]["yt_metadata_args"].get("writesubtitles", None):  # type: ignore
            config["reading"]["yt_args"]["yt_metadata_args"]["writesubtitles"] = "all"  # type: ignore

    if encode_formats is None:
        encode_formats = {"video": "mp4"}

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, local_args)
    tmp_path = output_folder + "/_tmp"
    fs, run_tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(run_tmp_dir):
        fs.mkdir(run_tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(run_tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {output_format}")

    if input_format == "webdataset":
        shard_iterator = OutputSharder(  # type: ignore
            url_list, input_format, done_shards, sampler=config["reading"]["sampler"]
        )
    else:
        shard_iterator = InputSharder(  # type: ignore
            url_list,
            input_format,
            url_col,
            caption_col,
            clip_col,
            save_additional_columns,
            config["storage"]["number_sample_per_shard"],
            done_shards,
            tmp_path,
            config["reading"]["sampler"],
        )

    if stage == "download":
        worker = DownloadWorker(
            sample_writer_class=sample_writer_class,
            save_caption=save_caption,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "subset":
        worker = SubsetWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "optical_flow":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = OpticalFlowWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "caption":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = CaptionWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    elif stage == "whisper":
        is_slurm_task = "GLOBAL_RANK" in os.environ and config["distribution"]["distributor"] == "multiprocessing"
        worker = WhisperWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")

    print("Starting the downloading of this file")
    if config["distribution"]["distributor"] == "multiprocessing" or called_from_slurm:
        distributor_fn = multiprocessing_distributor if stage not in ["whisper", "caption"] else no_distributor
        called_from_slurm = "GLOBAL_RANK" in os.environ
    elif config["distribution"]["distributor"] == "pyspark":
        distributor_fn = pyspark_distributor
    elif config["distribution"]["distributor"] == "slurm":
        worker_args = {key: local_args[key] for key in local_args if not key.startswith("slurm")}
        slurm_args = config["distribution"]["distributor_args"]

        distributor_fn = SlurmDistributor(worker_args=worker_args, **slurm_args)
    else:
        raise ValueError(f"Distributor {config['distribution']['distributor']} not supported")

    distributor_fn(
        config["distribution"]["processes_count"],
        worker,
        shard_iterator,
        config["distribution"]["subjob_size"],
        max_shard_retry,
    )
    logger_process.join()
    if not called_from_slurm:
        fs.rm(run_tmp_dir, recursive=True)


def main():
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
