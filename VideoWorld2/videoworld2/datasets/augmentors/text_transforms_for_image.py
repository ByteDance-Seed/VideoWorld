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
import random
from typing import Optional

import numpy as np
import torch

from imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from imaginaire.utils import log

# For the qwen captions, we have 3 variants: short, medium, long
# In addition, for synthetic data, we create prompt embeddings as well.
# There is quite a bit of entropy in the way prompt data is saved.
# Captions are saved as "prompts", while the corresponding embeddings are saved as "original_prompt"
# This part will be cleaned after synthetic data is cleaned to be in the same format as real data.
_AVAILABLE_QWEN_CAPTIONS = ["qwen2p5_7b_short", "qwen2p5_7b_medium", "qwen2p5_7b_long"]
_CAPTION_EMBEDDING_MAPPING = {
    "qwen2p5_7b_short": "qwen2p5_7b_short",
    "qwen2p5_7b_medium": "qwen2p5_7b_medium",
    "qwen2p5_7b_long": "qwen2p5_7b_long",
    "prompts": "original_prompt",
}


def pad_and_resize(
    arr_np: np.ndarray, ntokens: int, is_mask_all_ones: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Function for padding and resizing a numpy array.
    Args:
        arr (np.ndarray): Input array
        ntokens (int): Number of output tokens after padding
        is_mask_all_ones (bool): if true, set mask to ones
    Returns:
        arr_padded (torch.Tensor): Padded output tensor
        mask (torch.Tensor): Padding mask
    """

    if isinstance(arr_np, np.ndarray):
        arr = torch.from_numpy(arr_np)
    elif isinstance(arr_np, torch.Tensor):
        arr = arr_np.clone().detach()
    else:
        raise TypeError("`arr_np` should be a numpy array or torch tensor.")
    embed_dim = arr.shape[1]

    arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

    # If the input text is larger than num_text_tokens, clip it.
    if arr.shape[0] > ntokens:
        arr = arr[0:ntokens]

    mask = torch.LongTensor(ntokens).zero_()
    if len(arr.shape) > 1:
        mask[0 : arr.shape[0]] = 1

    if len(arr.shape) > 1:
        arr_padded[0 : arr.shape[0]] = arr

    if is_mask_all_ones:
        mask.fill_(1)

    return arr_padded, mask


class TextTransformForImage(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs camera transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        embedding_key_in_dict = _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES[caption_type]
        embedding_type = self.args["embedding_type"]
        embedding_input_key_prefix = "" if embedding_type == "t5_xxl" else "umt5_"

        captions_key, embeddings_key = (
            f"captions_{caption_type}",
            f"{embedding_input_key_prefix}embeddings_captions_{embedding_key_in_dict}",
        )
        decoded_captions_ai = data_dict[captions_key]
        decoded_embeddings_ai = data_dict[embeddings_key]

        try:
            # Hotfix: Some captions are labeled as "captions" and some are labeled as "caption"
            # This issue needs to be fixed in the synthetic data. This is a hack and will be removed
            # once the data is cleaned.
            caption_key = "captions" if "captions" in decoded_captions_ai else "caption"
            embedding_key = "t5_xxl_fp8" if embedding_type == "t5_xxl" else "umt5_xxl"
            if caption_type == "qwen2p5_7b_v4":
                selected_caption_type = random.choice(_AVAILABLE_QWEN_CAPTIONS)
                data_dict["raw_captions"] = decoded_captions_ai[caption_key][selected_caption_type]
                t5_embedding = decoded_embeddings_ai[selected_caption_type]["embeddings"][embedding_key]
                data_dict["selected_caption_type"] = selected_caption_type
            elif caption_type == "prompts":
                data_dict["raw_captions"] = decoded_captions_ai["caption"]["prompt"]
                t5_embedding = decoded_embeddings_ai[_CAPTION_EMBEDDING_MAPPING[caption_type]]["embeddings"][
                    embedding_key
                ]
                data_dict["selected_caption_type"] = caption_type

            out_t5, out_t5_mask = pad_and_resize(
                t5_embedding,
                self.args["t5_tokens"]["num"],
                is_mask_all_ones=self.args["is_mask_all_ones"],
            )
            data_dict["t5_text_embeddings"] = out_t5
            data_dict["t5_text_mask"] = out_t5_mask
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}\n error {e}",
                rank0_only=False,
            )
            return None

        del data_dict[captions_key]
        del data_dict[embeddings_key]

        return data_dict
