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
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .audio_rate_subsampler import AudioRateSubsampler
from .clipping_subsampler import ClippingSubsampler, _get_seconds, _split_time_frame, Streams
from .frame_subsampler import FrameSubsampler
from .ffprobe_subsampler import FFProbeSubsampler
from .noop_subsampler import NoOpSubsampler
from .resolution_subsampler import ResolutionSubsampler
from .cut_detection_subsampler import CutDetectionSubsampler
from .optical_flow_subsampler import OpticalFlowSubsampler
from .whisper_subsampler import WhisperSubsampler
from .caption_subsampler import CaptionSubsampler
