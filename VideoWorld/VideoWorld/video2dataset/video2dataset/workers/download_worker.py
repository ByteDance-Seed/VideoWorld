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
"""the downloader module handles the downloading"""

import math
import time
import pyarrow as pa
import traceback

import fsspec

from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Any
import numpy as np

from video2dataset.data_reader import VideoDataReader
from video2dataset.logger import CappedCounter
from video2dataset.logger import write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class DownloadWorker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        save_caption,
        output_folder,
        column_list,
        tmp_dir,
        encode_formats,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = save_caption
        self.output_folder = output_folder
        self.column_list = column_list
        self.encode_formats = encode_formats
        self.config = config

        self.data_reader = VideoDataReader(encode_formats, tmp_dir, config["reading"])

        self.clipping_subsampler = ClippingSubsampler(
            5,  # oom_clip_count
            encode_formats,
            **self.config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
        )
        need_keyframes = self.clipping_subsampler.precision == "keyframe_adjusted"

        self.ffprobe_subsampler = None
        if "FFProbeSubsampler" in self.config["subsampling"] or need_keyframes:
            self.ffprobe_subsampler = FFProbeSubsampler(
                **self.config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"]
            )
            self.ffprobe_subsampler.extract_keyframes |= need_keyframes

        self.cut_detector = None
        self.cuts_are_clips = False
        if "CutDetectionSubsampler" in self.config["subsampling"]:
            if "args" in self.config["subsampling"]["CutDetectionSubsampler"]:
                self.cut_detector = CutDetectionSubsampler(
                    **self.config["subsampling"]["CutDetectionSubsampler"]["args"]
                )
            self.cuts_are_clips = self.config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

        self.noop_subsampler = NoOpSubsampler()

        video_subsamplers: List[Any] = []
        if "ResolutionSubsampler" in self.config["subsampling"]:
            video_subsamplers.append(ResolutionSubsampler(**self.config["subsampling"]["ResolutionSubsampler"]["args"]))
        if "FrameSubsampler" in self.config["subsampling"]:
            video_subsamplers.append(FrameSubsampler(**self.config["subsampling"]["FrameSubsampler"]["args"]))

        audio_subsamplers: List[Any] = []
        if "AudioRateSubsampler" in self.config["subsampling"]:
            audio_subsamplers.append(AudioRateSubsampler(**self.config["subsampling"]["AudioRateSubsampler"]["args"]))

        self.subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    def __call__(
        self,
        row,
    ):
        try:
            self.download_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def download_shard(
        self,
        row,
    ):
        """Function to start an video downloading in one process"""

        # shard_id, shard_file = row
        shard_file, shard_id = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
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

        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed = {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
        bytes_downloaded = 0
        url_indice = self.column_list.index("url")
        caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        semaphore = Semaphore(self.config["distribution"]["thread_count"])

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()  # pylint: disable=(consider-using-with)
                yield e

        loader = data_generator()

        # The subsamplers might change the output format, so we need to update the writer
        writer_encode_formats = self.encode_formats.copy()
        if self.subsamplers["audio"]:
            writer_encode_formats["audio"] = self.subsamplers["audio"][0].encode_formats["audio"]
        if self.subsamplers["video"]:
            writer_encode_formats["video"] = self.subsamplers["video"][0].encode_formats["video"]

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            writer_encode_formats,
        )
        oom_sample_per_shard = math.ceil(math.log10(self.config["storage"]["number_sample_per_shard"]))

        with ThreadPool(self.config["distribution"]["thread_count"]) as thread_pool:
            for key, streams, yt_meta_dict, error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                loader,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(
                        key, shard_id, oom_sample_per_shard, self.config["storage"]["oom_shard_count"]
                    )
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                        "yt_meta_dict": yt_meta_dict,
                    }

                    if error_message is not None:
                        print(error_message)
                        if "[youtube]" in error_message:  # video-specific error, remove videoID
                            error_message = "ERROR: [youtube]:" + error_message.split(":")[-1]
                        raise ValueError("failed_to_download")

                    for stream in streams.values():
                        bytes_downloaded += len(stream)
                    for mod in streams:
                        streams[mod] = [streams[mod]]

                    if self.ffprobe_subsampler is not None:
                        streams, meta, error_message = self.ffprobe_subsampler(streams, meta)
                        if error_message is not None:
                            raise ValueError("failed_to_subsample")

                    if self.config["storage"]["captions_are_subtitles"]:  # create clips
                        # all langs have same start and end times
                        subtitles = meta["yt_meta_dict"]["subtitles"][list(meta["yt_meta_dict"]["subtitles"].keys())[0]]
                        meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                    elif self.cut_detector is not None:  # apply cut detection to get clips
                        streams, cuts, error_message = self.cut_detector(streams)

                        if error_message is not None:
                            raise ValueError("failed_to_subsample")

                        meta["cuts"] = cuts

                    if self.cuts_are_clips:
                        cuts = meta["cuts"]["cuts_original_fps"]
                        native_fps = meta["cuts"]["original_fps"]
                        meta["clips"] = (np.array(cuts) / native_fps).tolist()

                    # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                    broadcast_subsampler = (
                        self.clipping_subsampler
                        if (
                            "clips" in self.column_list
                            or self.config["storage"]["captions_are_subtitles"]
                            or self.cuts_are_clips
                        )
                        else self.noop_subsampler
                    )
                    subsampled_streams, metas, error_message = broadcast_subsampler(streams, meta)

                    for modality in subsampled_streams:
                        for modality_subsampler in self.subsamplers[modality]:
                            subsampled_streams, metas, error_message = modality_subsampler(subsampled_streams, metas)

                    if error_message is not None:
                        meta["clips"] = []
                        raise ValueError("failed_to_subsample")

                    successes += 1
                    status = "success"
                    status_dict.increment(status)
                    subsampled_streams_list = [
                        dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())
                    ]
                    for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                        meta["status"] = status

                        text_caption = sample_data[caption_indice] if caption_indice is not None else None
                        if self.config["storage"]["captions_are_subtitles"]:
                            text_caption = meta.get("clip_subtitles")[0]["lines"]

                        sample_writer.write(
                            subsampled_streams,
                            meta["key"],
                            text_caption,
                            meta,
                        )
                except Exception as err:  # pylint: disable=broad-except
                    status = str(err)
                    if status.startswith("failed_to_"):
                        failed[status] += 1
                        status_dict.increment(error_message)
                        meta["status"] = status
                        meta["error_message"] = error_message
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                    else:
                        traceback.print_exc()
                        print(f"Sample {key} failed to download: {err}")

                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed["failed_to_download"],
            failed["failed_to_subsample"],
            bytes_downloaded,
            start_time,
            end_time,
            status_dict,
            self.config["storage"]["oom_shard_count"],
        )
        fs.rm(shard_path)
