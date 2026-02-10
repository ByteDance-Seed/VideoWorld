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
"""Reader is module to read the url list and return shards"""
import braceexpand
import fsspec


class OutputSharder:
    """
    The reader class reads a shard list and returns shards

    It provides an iter method
    It provides attributes:
    - shard_list: a list of shards to read
    - input_format: the format of the input dataset
    - done_shards: a set of already done shards
    - group_shards: the number of shards to group together
    """

    def __init__(self, shard_list, input_format, done_shards, sampler=lambda x: x) -> None:
        self.input_format = input_format
        self.done_shards = done_shards
        self.column_list = None
        fs, url_path = fsspec.core.url_to_fs(shard_list)

        if fs.isdir(url_path):
            self.shard_list = sorted(fs.glob(url_path + "/*.tar"))
            if "s3://" in shard_list:
                self.shard_list = ["s3://" + s for s in self.shard_list]
            if len(self.shard_list) == 0:
                raise ValueError(f"No file found at path {url_path} with extension {input_format}")
        else:
            self.shard_list = list(braceexpand.braceexpand(shard_list))

        num_shards = len(self.shard_list)
        print(f"Found a total of {num_shards} shards!")

        if self.input_format == "webdataset":
            self.shard_ids = [s.split("/")[-1][: -len(".tar")] for s in self.shard_list]
        elif self.input_format == "files":
            self.shard_ids = [s.split("/")[-1] for s in self.shard_list]

        self.shards = sampler(
            [(s_id, s) for s_id, s in zip(self.shard_ids, self.shard_list) if int(s_id) not in self.done_shards]
        )

        num_shards = len(self.shards)
        print(f"Processing a total of {num_shards} shards!")

    def __iter__(self):
        """
        Iterate over shards, yield shards of size group_shards size
        Each shard is a tuple (shard_id, shard)
        """
        for s_id, s in self.shards:
            yield (s, s_id)
