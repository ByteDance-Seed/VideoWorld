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
Worker for optical flow stage
"""
import time
import math
import traceback
import json
import fsspec
import pyarrow as pa
import webdataset as wds

from multiprocessing.pool import ThreadPool
from threading import Semaphore

from video2dataset.data_reader import VideoDataReader
from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import WhisperSubsampler
from video2dataset.dataloader import get_video_dataset


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class WhisperWorker:
    """
    A class to read shards, process them using WhisperSubsampler, and write the output.

    Attributes:
        sample_writer_class (type): The class used to write samples.
        output_folder (str): The folder to write the output.
        thread_count (int): The number of threads.
        number_sample_per_shard (int): The number of samples per shard.
        oom_shard_count (int): The number of out-of-memory shards.
        encode_formats (dict): The encoding formats.
        detector (str): The optical flow detector type.
        fps (int): The target frames per second.
        optical_flow_subsampler (WhisperSubsampler): The WhisperSubsampler instance.
    """

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        column_list,
        tmp_dir,
        encode_formats,
        is_slurm_task,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.column_list = column_list
        self.encode_formats = encode_formats
        self.save_caption = False
        self.config = config

        self.data_reader = VideoDataReader(encode_formats, tmp_dir, config["reading"])

        if config["distribution"]["distributor"] != "slurm":
            self.whisper_subsampler = WhisperSubsampler(
                **self.config["subsampling"]["WhisperSubsampler"]["args"],
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
        Process a video shard using the WhisperSubsampler.

        Args:
            row (tuple): A tuple containing the shard and shard_id.

        Raises:
            Except
        """
        shard, shard_id = row
        start_time = time.time()

        _ = from_wds = shard.endswith(".tar")

        if from_wds:
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
            semaphore = None
        else:
            fs, shard_path = fsspec.core.url_to_fs(shard)
            with fs.open(shard_path, "rb") as f:
                df = pa.ipc.open_file(f).read_all()
            schema = df.schema
            schema = (
                schema.append(pa.field("key", pa.string()))
                .append(pa.field("status", pa.string()))
                .append(pa.field("error_message", pa.string()))
            )

            pydict = df.select(self.column_list).to_pydict()
            shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))
            del pydict
            del df
            url_indice = self.column_list.index("url")
            key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]
            semaphore = Semaphore(self.config["distribution"]["thread_count"])

            def data_generator():
                for e in key_url_list:
                    semaphore.acquire()  # pylint: disable=(consider-using-with)
                    yield e

            loader = data_generator()

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
        oom_sample_per_shard = math.ceil(math.log10(self.config["storage"]["number_sample_per_shard"]))

        successes = 0
        failed_to_subsample = 0

        if from_wds:
            dset = get_video_dataset(
                urls=shard,
                batch_size=1,
                decoder_kwargs={},
                video_key=self.encode_formats["audio"],
                enforce_additional_keys=[],
                return_always=True,
                handler=wds.warn_and_continue,
            )
        else:

            def create_dset():
                with ThreadPool(self.config["distribution"]["thread_count"]) as thread_pool:
                    for (
                        key,
                        streams,
                        yt_meta_dict,
                        error_message,
                    ) in thread_pool.imap_unordered(self.data_reader, loader):
                        str_key = compute_key(
                            key,
                            shard_id,
                            oom_sample_per_shard,
                            self.config["storage"]["oom_shard_count"],
                        )
                        sample = {
                            "__key__": str_key,
                            "__url__": shard,
                            "__corrupted__": error_message is not None,
                        }
                        meta = [
                            {
                                "key": str_key,
                                "status": None,
                                "error_message": error_message,
                                "yt_meta_dict": yt_meta_dict,
                            }
                        ]
                        sample.update(streams)
                        sample["meta"] = meta
                        yield sample

            dset = create_dset()

        count = 0
        for sample in dset:
            count += 1
            corrupted = sample["__corrupted__"]
            key = sample["__key__"]
            if corrupted:
                url = sample["__url__"]
                meta = {}
                meta["url"] = url
                meta["key"] = key
                # error_message = "corrupted sample"
                error_message = sample["meta"][0]["error_message"]
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
                _ = semaphore.release() if not from_wds else None
                continue
            meta = [json.loads(sample.get("json", b"{}").decode("utf-8"))] if from_wds else sample.pop("meta")

            streams = {"audio": [sample[self.encode_formats["audio"]]]} if from_wds else {"audio": [sample["audio"]]}
            streams, meta, error_message = self.whisper_subsampler(streams, meta)
            if error_message is not None:
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta = meta[0]
                meta["key"] = key
                meta["url"] = sample["__url__"]
                meta["status"] = status
                meta["error_message"] = error_message
                meta["__corrupted__"] = True
                sample_writer.write(
                    {},
                    key,
                    None,
                    meta,
                )
                _ = semaphore.release() if not from_wds else None
                continue

            streams.pop("audio")  # only write metadata shards
            meta = meta[0]  # remove when unifying workers (needs to be list)

            successes += 1
            status = "success"
            status_dict.increment(status)
            meta["url"] = sample["__url__"]
            meta["status"] = status
            meta["__corrupted__"] = False

            sample_writer.write(
                streams,
                key,
                None,
                meta,
            )
            _ = semaphore.release() if not from_wds else None

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
