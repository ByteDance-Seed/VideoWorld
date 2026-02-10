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
import argparse
import json
import os
from typing import Iterable, Tuple, Union

import torch
from PIL import Image

from videoworld2.auxiliary.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from videoworld2.auxiliary.guardrail.common.io_utils import get_video_filepaths, read_video
from videoworld2.auxiliary.guardrail.video_content_safety_filter.model import ModelConfig, VideoSafetyModel
from videoworld2.auxiliary.guardrail.video_content_safety_filter.vision_encoder import SigLIPEncoder
from imaginaire.utils import log, misc

# Define the class index to class name mapping for multi-class classification
CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    2: "Violence",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}


class VideoContentSafetyFilter(ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_dir: str,
        offload_model_to_cpu: bool = True,
    ) -> None:
        """Video content safety filter model.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
        """
        self.offload_model = offload_model_to_cpu
        self.dtype = torch.float32

        self.checkpoint_dir = os.path.join(checkpoint_dir, "nvidia/Cosmos-Guardrail1/video_content_safety_filter")

        # Use ModelConfig directly for inference configuration
        model_config = ModelConfig(input_size=1152, num_classes=7)

        # Load the multi-class classifier and initialize the SigLIP encoder
        self.model = VideoSafetyModel(model_config)
        safety_filter_local_path = os.path.join(self.checkpoint_dir, "safety_filter.pt")
        checkpoint = torch.load(safety_filter_local_path, map_location=torch.device("cpu"), weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.encoder = SigLIPEncoder(checkpoint_dir=self.checkpoint_dir, device="cuda", dtype=self.dtype)
        if offload_model_to_cpu:
            self.encoder.to("cpu")
            self.model = self.model.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved video content safety filter to CPU")
        else:
            self.encoder.to("cuda")
            self.model = self.model.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved video content safety filter to GPU")

    @torch.inference_mode()
    def __infer(self, pil_image: Image.Image) -> int:
        """Infer the class of the image."""
        image_embs = self.encoder.encode_image(pil_image)
        logits = self.model.network(image_embs)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class

    def _to_cuda_if_offload(self):
        if self.offload_model:
            self.encoder = self.encoder.to("cuda")
            self.model = self.model.to("cuda")
            log.debug("Move video content safety filter to GPU")

    def _to_cpu_if_offload(self):
        if self.offload_model:
            self.encoder = self.encoder.to("cpu")
            self.model = self.model.to("cpu")
            log.debug("Offload video content safety filter to CPU")

    def is_safe_file(self, filepath: str) -> bool:
        """Check if the video file is safe."""
        video_data = read_video(filepath)

        # Sample frames at 2 FPS
        sample_rate = 2  # frames per second
        frame_interval = int(video_data.fps / sample_rate)
        frame_numbers = list(range(0, int(video_data.fps * video_data.duration), frame_interval))

        is_safe = True
        frame_scores = []

        self._to_cuda_if_offload()
        for frame_number in frame_numbers:
            try:
                frame = video_data.frames[frame_number]
                pil_image = Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark the video as unsafe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                log.warning(f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}")
                continue

        # Prepare data for JSON
        video_data = {
            "filepath": filepath,
            "is_safe": is_safe,
            "video_length": video_data.duration,
            "fps": video_data.fps,
            "frame_scores": frame_scores,
        }
        self._to_cpu_if_offload()
        log.info(f"Video {filepath} is {'SAFE' if is_safe else 'UNSAFE'}.")
        log.debug(f"Video data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def is_safe_frames(self, frames: Iterable) -> bool:
        """Check if the video frames are safe."""
        is_safe = True
        frame_scores = []

        self._to_cuda_if_offload()
        for frame_number, frame in enumerate(frames):
            try:
                pil_image = Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark as not safe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                log.warning(f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}")
                continue

        video_data = {
            "is_safe": is_safe,
            "frame_scores": frame_scores,
        }
        self._to_cpu_if_offload()
        log.debug(f"Frames data: {json.dumps(video_data, indent=4)}")
        return is_safe

    def is_safe(self, input: Union[str, Iterable]) -> Tuple[bool, str]:
        if isinstance(input, str):
            is_safe = self.is_safe_file(input)
            return is_safe, "safe video detected" if is_safe else "unsafe video detected"
        elif isinstance(input, Iterable):
            is_safe = self.is_safe_frames(input)
            return is_safe, "safe frames detected" if is_safe else "unsafe frames detected"
        else:
            raise ValueError(f"Input type {type(input)} not supported.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing input videos")
    return parser.parse_args()


def main(args):
    filepaths = get_video_filepaths(args.input_dir)
    if not filepaths:
        log.error(f"No video files found in directory: {args.input_dir}")
        return

    video_filter = VideoContentSafetyFilter(
        checkpoint_dir="checkpoints/nvidia/Cosmos-Guardrail1/video_content_safety_filter"
    )
    runner = GuardrailRunner(safety_models=[video_filter], generic_safe_msg="Video is safe")

    for filepath in filepaths:
        with misc.timer("video content safety filter"):
            _ = runner.run_safety_check(filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
