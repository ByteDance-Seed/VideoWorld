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
Worker for caption stage
"""
import time
import pyarrow as pa
import traceback
import numpy as np
import torch
import fsspec
import webdataset as wds

from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import CaptionSubsampler
from video2dataset.dataloader import get_video_dataset


class CaptionWorker:
    """
    A class to read shards, process them using CaptionSubsampler, and write the output.

    Attributes:
        sample_writer_class (type): The class used to write samples.
        output_folder (str): The folder to write the output.
        thread_count (int): The number of threads.
        number_sample_per_shard (int): The number of samples per shard.
        oom_shard_count (int): The number of out-of-memory shards.
        encode_formats (dict): The encoding formats.
        fps (int): The target frames per second.
        caption_subsampler (CaptionSubsampler): The CaptionSubsampler instance.
    """

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        encode_formats,
        is_slurm_task,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.encode_formats = encode_formats
        self.save_caption = True
        self.config = config

        self.caption_subsampler = CaptionSubsampler(
            **self.config["subsampling"]["CaptionSubsampler"]["args"],
            is_slurm_task=is_slurm_task,
        )

    def __call__(
        self,
        row,
    ):
        try:
            self.process_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def process_shard(
        self,
        row,
    ):
        """
        Process a video shard using the CaptionSubsampler.

        Args:
            row (tuple): A tuple containing the shard and shard_id.

        Raises:
            Except
        """
        shard, shard_id = row
        start_time = time.time()

        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")

            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception as e:  # pylint: disable=broad-except,unused-variable
            fields = [
                pa.field("key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("error_message", pa.string()),
            ]
            schema = pa.schema(fields)

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            self.encode_formats,
        )

        successes = 0
        failed_to_subsample = 0

        dset = get_video_dataset(
            urls=shard,
            batch_size=self.config["reading"]["dataloader_args"]["batch_size"],
            decoder_kwargs=self.config["reading"]["dataloader_args"]["decoder_kwargs"],
            resize_size=self.config["reading"]["dataloader_args"]["resize_size"],
            crop_size=None,
            enforce_additional_keys=[],
            return_always=True,
            handler=wds.warn_and_continue,
        )
        count = 0
        for sample in dset:
            batch_size = len(sample["__key__"])
            count += batch_size

            bad_batch_idx = []
            for batch_idx in range(batch_size):
                corrupted = sample["__corrupted__"][batch_idx]
                key = sample["__key__"][batch_idx]
                if corrupted:
                    bad_batch_idx.append(batch_idx)
                    url = sample["__url__"][batch_idx]
                    meta = {}
                    meta["url"] = url
                    meta["key"] = key
                    error_message = "corrupted sample"
                    failed_to_subsample += 1
                    status = "failed_to_subsample"
                    status_dict.increment(error_message)
                    meta["status"] = status
                    meta["error_message"] = error_message
                    meta["__corrupted__"] = True
                    sample_writer.write(
                        {},
                        key,
                        None,
                        meta,
                    )
                    continue

            if len(bad_batch_idx) != 0:
                bad_batch_idx.sort(reverse=True)
                for key in sample:
                    if isinstance(sample[key], list):
                        for i in bad_batch_idx:
                            del sample[key][i]
                    elif isinstance(sample[key], np.ndarray):
                        mask = np.ones(len(sample[key]), dtype=bool)
                        mask[bad_batch_idx] = False
                        sample[key] = sample[key][mask]
                    elif isinstance(sample[key], torch.Tensor):
                        mask = torch.ones(len(sample[key]), dtype=bool)
                        mask[bad_batch_idx] = False
                        sample[key] = sample[key][mask]
                    else:
                        raise TypeError(f"Unsupported data type: {type(sample[key])}")

            batch_size = len(sample["__key__"])
            # incase all elems are corrupted
            if batch_size == 0:
                continue

            meta = sample["json"]
            streams = {}
            caption, error_message = self.caption_subsampler(sample.get("mp4"))

            if error_message is not None:
                for batch_idx in range(batch_size):
                    meta = sample["json"][batch_idx]
                    key = sample["__key__"][batch_idx]
                    failed_to_subsample += 1
                    status = "failed_to_subsample"
                    status_dict.increment(error_message)
                    meta["status"] = status
                    meta["error_message"] = error_message
                    meta["__corrupted__"] = True
                    sample_writer.write(
                        streams,
                        key,
                        None,
                        meta,
                    )
                    # continue
                continue

            for batch_idx in range(batch_size):
                sample["json"][batch_idx]["vblip"] = caption[0][batch_idx]

                meta = sample["json"][batch_idx]
                key = sample["__key__"][batch_idx]
                successes += 1
                status = "success"
                status_dict.increment(status)
                meta["status"] = status
                meta["__corrupted__"] = False

                sample_writer.write(
                    streams,
                    key,
                    caption[0][batch_idx],
                    meta,
                )

        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            count,  # count
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.config["storage"]["oom_shard_count"],
        )
