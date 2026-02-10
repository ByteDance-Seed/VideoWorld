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
"""test video2dataset subsamplers"""
import os
import subprocess
import pytest
import ffmpeg
import tempfile
import numpy as np
import cv2

from video2dataset.subsamplers import (
    ClippingSubsampler,
    _get_seconds,
    _split_time_frame,
    Streams,
    FFProbeSubsampler,
    ResolutionSubsampler,
    FrameSubsampler,
    AudioRateSubsampler,
    CutDetectionSubsampler,
    OpticalFlowSubsampler,
    WhisperSubsampler,
)


SINGLE = [[50.0, 60.0]]
MULTI = [
    ["00:00:09.000", "00:00:13.500"],
    ["00:00:13.600", "00:00:24.000"],
    ["00:00:45.000", "00:01:01.230"],
    ["00:01:01.330", "00:01:22.000"],
    ["00:01:30.000", "00:02:00.330"],
]


@pytest.mark.parametrize("clips", [SINGLE, MULTI])
def test_clipping_subsampler(clips):
    current_folder = os.path.dirname(__file__)
    # video lenght - 2:02
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()
    audio = os.path.join(current_folder, "test_files/test_audio.mp3")
    with open(audio, "rb") as aud_f:
        audio_bytes = aud_f.read()

    min_length = 5.0 if clips == MULTI else 2.0
    max_length = 999999.0 if clips == MULTI else 3.0
    subsampler = ClippingSubsampler(
        oom_clip_count=3,
        encode_formats={"video": "mp4", "audio": "mp3"},
        min_length=min_length,
        max_length=max_length,
        max_length_strategy="all",
        precision="low",
    )

    metadata = {
        "key": "000",
        "clips": clips,
    }

    streams: Streams = {"video": [video_bytes], "audio": [audio_bytes]}
    stream_fragments, meta_fragments, error_message = subsampler(streams, metadata)
    video_fragments = stream_fragments["video"]
    audio_fragments = stream_fragments["audio"]
    assert error_message is None
    # first one is only 4.5s
    assert len(audio_fragments) == len(video_fragments) == len(meta_fragments)
    if clips == SINGLE:
        assert len(video_fragments) == 3
    else:
        assert len(video_fragments) == 4

    for vid_frag, meta_frag in zip(video_fragments, meta_fragments):
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(vid_frag)
            key_ind = int(meta_frag["key"].split("_")[-1])
            s, e = meta_frag["clips"][0]

            if clips == MULTI:
                key_ind += 1
            else:
                key_ind = 0

            s_target, e_target = clips[key_ind]
            s_target, e_target = _get_seconds(s_target), _get_seconds(e_target)
            expected_clips = _split_time_frame(s_target, e_target, min_length, max_length)
            assert [_get_seconds(s), _get_seconds(e)] in expected_clips
            assert _get_seconds(e) - _get_seconds(s) >= min_length

            s_s, e_s = _get_seconds(s), _get_seconds(e)
            probe = ffmpeg.probe(tmp.name)
            video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
            frag_len = float(video_stream["duration"])

            assert abs(frag_len - (e_s - s_s)) < 5.0


@pytest.mark.parametrize("size,resize_mode", [(144, ["scale"]), (1620, ["scale", "crop", "pad"])])
def test_resolution_subsampler_video_size(size, resize_mode):
    current_folder = os.path.dirname(__file__)
    # video lenght - 2:02, 1080x1920
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ResolutionSubsampler(video_size=size, resize_mode=resize_mode)

    streams = {"video": [video_bytes]}
    subsampled_streams, _, error_message = subsampler(streams)
    assert error_message is None
    subsampled_videos = subsampled_streams["video"]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        h_vid, w_vid = video_stream["height"], video_stream["width"]

        assert h_vid == size
        if resize_mode == ["scale"]:
            assert w_vid == 256  # 1920 / (1080/144)
        else:
            assert w_vid == size


@pytest.mark.parametrize("height,width,resize_mode", [(-1, 128, ["scale"]), (1620, 1620, ["scale", "crop", "pad"])])
def test_resolution_subsampler_height_and_width(height, width, resize_mode):
    current_folder = os.path.dirname(__file__)
    # video lenght - 2:02, 1080x1920
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ResolutionSubsampler(height=height, width=width, resize_mode=resize_mode)

    streams = {"video": [video_bytes]}
    subsampled_streams, _, error_message = subsampler(streams)
    assert error_message is None
    subsampled_videos = subsampled_streams["video"]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        h_vid, w_vid = video_stream["height"], video_stream["width"]

        if resize_mode == ["scale"]:
            assert h_vid == 72
            assert w_vid == 128
        else:
            assert h_vid == height
            assert w_vid == width


@pytest.mark.parametrize("target_frame_rate", [6, 15, 30])
def test_frame_rate_subsampler(target_frame_rate):
    current_folder = os.path.dirname(__file__)
    # video length - 2:02, 1080x1920, 30 fps
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = FrameSubsampler(target_frame_rate)

    streams = {"video": [video_bytes]}
    subsampled_streams, _, error_message = subsampler(streams)
    assert error_message is None
    subsampled_videos = subsampled_streams["video"]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        frame_rate = int(video_stream["r_frame_rate"].split("/")[0])

        assert frame_rate == target_frame_rate


@pytest.mark.parametrize("sample_rate,n_audio_channels", [(44100, 1), (24000, 2)])
def test_audio_rate_subsampler(sample_rate, n_audio_channels):
    current_folder = os.path.dirname(__file__)
    audio = os.path.join(current_folder, "test_files/test_audio.mp3")
    with open(audio, "rb") as aud_f:
        audio_bytes = aud_f.read()

    streams = {"audio": [audio_bytes]}
    subsampler = AudioRateSubsampler(sample_rate, "mp3", n_audio_channels)

    subsampled_streams, _, error_message = subsampler(streams)
    assert error_message is None
    subsampled_audios = subsampled_streams["audio"]

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(subsampled_audios[0])

        out = subprocess.check_output(f"file {tmp.name}".split()).decode("utf-8")
        assert "Audio file with ID3 version" in out

        result = ffmpeg.probe(tmp.name)
        read_sample_rate = result["streams"][0]["sample_rate"]
        read_num_channels = result["streams"][0]["channels"]
        assert int(read_sample_rate) == sample_rate
        assert int(read_num_channels) == n_audio_channels


@pytest.mark.parametrize(
    "cut_detection_mode,framerates", [("longest", []), ("longest", [1]), ("all", []), ("all", [1])]
)
def test_cut_detection_subsampler(cut_detection_mode, framerates):
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = CutDetectionSubsampler(cut_detection_mode, framerates, threshold=5)

    streams = {"video": [video_bytes]}
    streams, cuts, err_msg = subsampler(streams)
    if cut_detection_mode == "longest":
        assert len(cuts["cuts_original_fps"]) == 1
        assert cuts["cuts_original_fps"][0] == [0, 2096]

        if len(framerates) > 0:
            assert cuts["cuts_1"][0] == [0, 2100]

    if cut_detection_mode == "all":
        assert len(cuts["cuts_original_fps"]) > 1
        assert cuts["cuts_original_fps"][-1] == [3015, 3678]

        if len(framerates) > 0:
            assert cuts["cuts_1"][-1] == [3420, 3678]


@pytest.mark.parametrize(
    "detector,fps,params", [("cv2", 1, None), ("cv2", 2, None), ("cv2", 1, (0.25, 2, 10, 3, 5, 1, 0))]
)
def test_optical_flow_subsampler(detector, fps, params):
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")

    cap = cv2.VideoCapture(video)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    step = int(native_fps / fps)
    raw_frames = np.array(frames)[::step]

    subsampler = OpticalFlowSubsampler(detector, detector_args=params)

    optical_flow, metrics, error_message = subsampler(raw_frames)
    mean_magnitude, _ = metrics
    assert error_message is None

    first_frame = optical_flow[0]
    assert first_frame.shape == (1080, 1920, 2)

    if fps == 1:
        if params is None:
            assert np.isclose(mean_magnitude, 0.00225532218919966, rtol=1e-3)  # verified independently on colab
        elif params is not None:
            assert np.isclose(
                mean_magnitude, 0.0034578320931094217, rtol=1e-3
            )  # np.isclose due to potential numerical precision issues
    elif fps == 2:  # fps = 2, params = None
        assert np.isclose(mean_magnitude, 0.0011257734728123598, rtol=1e-3)


@pytest.mark.parametrize("extract_keyframes", [False, True])
def test_ffprobe_subsampler(extract_keyframes):
    current_folder = os.path.dirname(__file__)
    # video length - 2:02, 1080x1920, 30 fps
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = FFProbeSubsampler(extract_keyframes)

    streams = {"video": [video_bytes]}
    metadata = {}
    subsampled_streams, metadata, error_message = subsampler(streams, metadata)
    assert error_message is None
    assert metadata is not None
    assert "video_metadata" in metadata

    video_metadata = metadata["video_metadata"]

    # check some basic metadata
    assert "format" in video_metadata
    assert "duration" in video_metadata["format"]
    assert "streams" in video_metadata
    video_stream_info = next(stream for stream in video_metadata["streams"] if stream["codec_type"] == "video")

    assert "width" in video_stream_info
    assert "height" in video_stream_info
    assert "r_frame_rate" in video_stream_info

    if extract_keyframes:
        assert "keyframe_timestamps" in video_metadata
        assert isinstance(video_metadata["keyframe_timestamps"], list)
        assert len(video_metadata["keyframe_timestamps"]) > 0
    else:
        assert "keyframe_timestamps" not in metadata


def test_whisper_subsampler():
    current_folder = os.path.dirname(__file__)
    audio = os.path.join(current_folder, "test_files/test_audio.mp3")
    with open(audio, "rb") as aud_f:
        audio_bytes = aud_f.read()

    subsampler = WhisperSubsampler("small", 4, "float32")
    streams = {"audio": [audio_bytes]}
    metadata = [{"key": "000"}]

    _, metadata, error_message = subsampler(streams, metadata)
    assert error_message is None
    transcript = metadata[0]["whisper_transcript"]
    assert transcript["segments"][0]["text"].startswith(" Bob Jones University in Greenville, South Carolina")
    assert len(transcript["segments"]) == 28
