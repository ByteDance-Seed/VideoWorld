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
import cosmos_predict2.datasets.augmentors.append_fps_frames_for_image as append_fps_frames_for_image
import cosmos_predict2.datasets.augmentors.merge_datadict as merge_datadict
import cosmos_predict2.datasets.augmentors.text_transforms_for_image as text_transforms_for_image
import cosmos_predict2.datasets.augmentors.text_transforms_for_video as text_transforms_for_video
import cosmos_predict2.datasets.augmentors.video_parsing as video_parsing
import imaginaire.datasets.webdataset.augmentors.image.normalize as normalize
import imaginaire.datasets.webdataset.augmentors.image.padding as padding
import imaginaire.datasets.webdataset.augmentors.image.resize as resize
from videoworld2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import log

AUGMENTOR_OPTIONS = {}


def augmentor_register(key):
    log.info(f"registering {key}...")

    def decorator(func):
        AUGMENTOR_OPTIONS[key] = func
        return func

    return decorator


def get_video_text_transform(
    caption_type: str,
    embedding_type: str,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
):
    del num_video_frames
    if caption_type == "t2w_qwen2p5_7b":
        log.info(
            f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
        )
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "t2w_windows",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "t2w_qwen2p5_7b",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        video_text_transform = L(text_transforms_for_video.TextTransformForVideo)(
            input_keys=[],
            args={
                "captions_key": "metas",
                "embeddings_key": embedding_type,
                "caption_windows_key": "i2w_windows_later_frames",
                "caption_type": "qwen2p5_7b_caption",
                "embedding_caption_type": "i2w_qwen2p5_7b_later_frames",
                "t5_tokens": {"num": 512},
                "is_mask_all_ones": True,
                "caption_probs": {
                    "long": long_caption_ratio,
                    "medium": medium_caption_ratio,
                    "short": short_caption_ratio,
                    "user": user_caption_ratio,
                },
            },
        )
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

    return video_text_transform


@augmentor_register("video_basic_augmentor_v2")
def get_video_augmentor_v2(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
):
    """
    num_video_frames: -1 means use all frames, otherwise use the number of frames specified.

    Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(
        caption_type=caption_type,
        embedding_type=embedding_type,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
    )
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    assert embedding_type in ("t5_xxl", "umt5_xxl"), f"Unsupported embeddings type ({embedding_type}) for video data"

    return {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
    }


@augmentor_register("image_basic_augmentor")
def get_image_augmentor(
    resolution: str,
    caption_type: str = "ai_v3p1",
    embedding_type: str = "t5_xxl",
):
    augmentation = {
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["images"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["images"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "text_transform": L(text_transforms_for_image.TextTransformForImage)(
            input_keys=[],
            args={
                "caption_type": caption_type,
                "embedding_type": embedding_type,
                "weight_captions_gt": 0.05,
                "caption_probs": {"ground_truth": 1},
                "t5_tokens": {"num": 512, "dim": 1024},
                "is_mask_all_ones": True,
            },
        ),
        "append_fps_frames": L(append_fps_frames_for_image.AppendFPSFramesForImage)(),
    }

    return augmentation


@augmentor_register("video_basic_augmentor_v2_with_edge_control")
def get_video_augmentor_v2_with_edge_control(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    num_video_frames: int = -1,
    **kwargs,
):
    """Video augmentor V2. It works with a naive video decoder ("video_naive_bytes") that does nothing.
    Augmentors here include:
    - a basic video decoder that fetches frames within a window and delegates further subsampling or duplication to the modeling code to produce videos with the required number of frames.
    - resize the video
    - add reflection padding
    - extract captions and embeddings.

    Supported caption_type include t2w_qwen2p5_7b and i2w_qwen2p5_7b_later_frames.
    Supported embedding_type include t5_xxl and umt5_xxl.
    """
    video_text_transform = get_video_text_transform(caption_type=caption_type, embedding_type=embedding_type)
    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        f"Unsupported caption type ({caption_type}) for video data"
    assert embedding_type in ("t5_xxl", "umt5_xxl"), f"Unsupported embeddings type ({embedding_type}) for video data"

    return {
        "video_parsing": L(video_parsing.VideoParsing)(
            input_keys=["metas", "video"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 4,
                "num_video_frames": num_video_frames,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=["video"],
            output_keys=[
                "video",
                "fps",
                "num_frames",
                "chunk_index",
                "frame_start",
                "frame_end",
                "n_orig_video_frames",
            ],
        ),
        "resize_largest_side_aspect_ratio_preserving": L(resize.ResizeLargestSideAspectPreserving)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "reflection_padding": L(padding.ReflectionPadding)(
            input_keys=["video"],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "text_transform": video_text_transform,
    }
