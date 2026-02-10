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
import pickle
from collections.abc import Iterable
from functools import partial
from typing import Callable

import omegaconf
import webdataset as wds
from webdataset.handlers import reraise_exception

from imaginaire.datasets.webdataset.config.schema import AugmentorConfig, DatasetConfig, DatasetInfo, TarSample, Wdinfo
from imaginaire.datasets.webdataset.utils.iterators import WebDataset
from imaginaire.datasets.webdataset.utils.misc import remove_extensions_from_keys, skip_keys, update_url
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log
from imaginaire.utils.distributed import get_world_size


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


class Dataset:
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            world_size: Total number of GPUs
        """
        super().__init__()

        self.config = config

        self.world_size = get_world_size()

        dataset_info = config.dataset_info
        self.streaming_download = config.streaming_download

        self.data_keys = config.keys

        # Parse the metadata
        self.wdinfo = Wdinfo([], 0, 0)
        self.parse_dataset_info(dataset_info=dataset_info)
        self.handler = handler
        self.augmentors = dict()

    def parse_dataset_info(self, dataset_info: list[DatasetInfo]):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
        """

        for dset_num, dset_info in enumerate(dataset_info):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.

            dset_id = "dset: {}".format(dset_num)

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path in dset_info.wdinfo:
                with open(wdinfo_path, "rb") as fp:
                    cur_dset_info = pickle.load(fp)

                # No need for sample_keys_full_list since we removed object store functionality
                sample_keys_full_list_per_tar = [None] * len(cur_dset_info["data_list"])

                data_root = cur_dset_info["root"]
                tar_files_list = cur_dset_info["data_list"]
                tar_files = [
                    TarSample(
                        path=tar_file,
                        root=data_root,
                        keys=dset_info.per_dataset_keys
                        if dset_info.per_dataset_keys
                        else self.data_keys,  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        sample_keys_full_list=sample_keys_full_list,
                    )
                    for tar_file, sample_keys_full_list in zip(
                        tar_files_list, sample_keys_full_list_per_tar, strict=True
                    )
                ]

                # Update the master winfo
                self.wdinfo.tar_files.extend(tar_files)
                self.wdinfo.total_key_count += cur_dset_info["total_key_count"]
                self.wdinfo.chunk_size = cur_dset_info["chunk_size"]

    @staticmethod
    # This is the function that calls each augmentor in sequence.
    def augmentor_fn(data, augmentations):
        # Build augmentor chain
        for aug_fn in augmentations:
            # Use generator function as augmentor
            # (recommended, allows skipping or replicating samples inside the augmentor)
            if getattr(aug_fn, "is_generator", False):
                data = aug_fn(data)
            else:  # Use regular function as augmentor (backward compatibility)
                data = wrap_augmentor_func_as_generator(aug_fn, data)
        yield from data

    def build_data_augmentor(self, augmentor_cfg: dict[str, AugmentorConfig]) -> Callable:
        r"""Function for building data augmentors from augmentor config."""
        augmentations = []
        for aug in augmentor_cfg.keys():
            augmentations.append(instantiate(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(Dataset.augmentor_fn, augmentations=augmentations)

    def build_dataset(self, **kwargs) -> WebDataset:
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if shuffle_buffer_size > 0:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(decoder), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(
            augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig)
        ), f"getting type: {type(augmentor_cfg)}"
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        log.info("Total number of training shards: %d" % num_tars)
        log.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset
