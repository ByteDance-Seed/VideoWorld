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
Whisper subsampler - transcribes audio using the Whisper model from OAI using WhisperX API

code: https://github.com/m-bain/whisperX
"""
import os
import time
import tempfile

try:
    import whisperx
    import torch
except:  # pylint: disable=broad-except,bare-except
    pass

from .subsampler import Subsampler


class WhisperSubsampler(Subsampler):
    """
    Transcribes audio samples using the OAI Whisper Model via WhisperX API

    Params:
        model_name: https://github.com/guillaumekln/faster-whisper/blob/20d4e9418b5efb69ec5aa4819a39e3fb0e772a2a/faster_whisper/transcribe.py#LL90C1-L90C1
        batch_size: batch size used during inference (try to maximize this for perf)
        compute_type: accuracy/mem tradeoff (float16, float32, int8)
    """

    def __init__(
        self,
        model_name="large-v2",
        batch_size=16,
        compute_type="float16",
        download_root=None,
        is_slurm_task=False,
    ):
        if is_slurm_task:
            global_rank = os.environ["GLOBAL_RANK"]
            if global_rank != 0:
                time.sleep(20)  # let master worker download model

            device, device_index = "cuda", int(os.environ["LOCAL_RANK"])
            while True:
                try:
                    self.model = whisperx.load_model(
                        model_name,
                        device=device,
                        device_index=device_index,
                        compute_type=compute_type,
                        download_root=download_root,
                    )
                    print("model_loaded", os.environ["GLOBAL_RANK"], flush=True)
                    break
                except Exception as e:  # pylint: disable=(broad-except)
                    print(str(e), flush=True)
                    print(
                        "loading failed, retrying...",
                        os.environ["GLOBAL_RANK"],
                        flush=True,
                    )
                    continue
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )

        self.batch_size = batch_size

    def __call__(self, streams, metadata=None):
        audio_bytes = streams.get("audio")

        for i, aud_bytes in enumerate(audio_bytes):
            # TODO: .m4a not always
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmpfile:
                    tmpfile.write(aud_bytes)
                    tmpfile.flush()  # ensure all data is written
                    audio = whisperx.load_audio(tmpfile.name)
                    result = self.model.transcribe(audio, batch_size=self.batch_size)
                    metadata[i]["whisper_transcript"] = result
            except Exception as err:  # pylint: disable=broad-except
                return [], metadata, str(err)

        return streams, metadata, None
