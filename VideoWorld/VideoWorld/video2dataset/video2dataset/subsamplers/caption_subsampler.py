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
video captioner
"""
import os

import torch
from torch import nn
from einops import rearrange
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Blip2QFormerModel,
    Blip2VisionModel,
)

from transformers.modeling_outputs import BaseModelOutputWithPooling

from .subsampler import Subsampler


class AttrDict(dict):
    """
    Lets us access dict keys with <dict>.key
    """

    # pylint: disable=super-with-arguments
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VideoBlipVisionModel(Blip2VisionModel):
    """
    A simple, augmented version of Blip2VisionModel to handle videos.
    Source: https://github.com/yukw777/VideoBLIP
    """

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward method for vision blip model"""
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        batch, _, time, _, _ = pixel_values.size()
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)
        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(batch, time * seq_len, -1)
        pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
        hidden_states = (
            tuple(hidden.view(batch, time * seq_len, -1) for hidden in vision_outputs.hidden_states)
            if vision_outputs.hidden_states is not None
            else None
        )
        attentions = (
            tuple(hidden.view(batch, time, -1, seq_len, seq_len) for hidden in vision_outputs.attentions)
            if vision_outputs.attentions is not None
            else None
        )
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return (last_hidden_state, pooler_output, hidden_states, attentions)


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    """
    A simple, augmented version of Blip2ForConditionalGeneration to handle videos.
    Source: https://github.com/yukw777/VideoBLIP
    """

    def __init__(self, config: Blip2Config) -> None:
        super(Blip2ForConditionalGeneration, self).__init__(config)  # pylint: disable=E1003
        self.vision_model = VideoBlipVisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model
        self.post_init()


class VideoCaptioner:
    """Synthetic video captioner using the VideoBLIP model."""

    def __init__(self, args):
        assert args.video_captioner in ["kpyu/video-blip-flan-t5-xl-ego4d", "kpyu/video-blip-opt-2.7b-ego4d"]
        model = args.video_captioner
        if args.prompt is None:
            self.prompt = "Question: Can you give a detailed description of what is happening in this video? Answer:"
        else:
            self.prompt = args.prompt

        self.device = args.get("device", "cuda")
        self.processor = Blip2Processor.from_pretrained(model)

        if ":" in self.device:
            device_map = {"": int(self.device.split(":")[-1])}
        else:
            device_map = {"": 0}
        self.model = VideoBlipForConditionalGeneration.from_pretrained(model, load_in_8bit=True, device_map=device_map)

    def __call__(self, video):
        video = video * 0.00392156862745098
        video = (video - torch.Tensor([0.48145466, 0.4578275, 0.40821073])) / torch.Tensor(
            [0.26862954, 0.26130258, 0.27577711]
        )
        video = rearrange(video, "b t h w c -> b c t h w").to(torch.float16)

        inputs = self.processor(images=None, text=[self.prompt] * video.shape[0], return_tensors="pt")
        inputs["pixel_values"] = video
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=10,
                early_stopping=True,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [x.strip() for x in generated_text]
        return generated_text


class CaptionSubsampler(Subsampler):
    """A class to generate synthetic text caption from video frames."""

    def __init__(
        self,
        captioner_args=None,
        is_slurm_task=False,
    ):
        if is_slurm_task:
            local_rank = os.environ["LOCAL_RANK"]
            device = f"cuda:{local_rank}"
            captioner_args["device"] = device
        else:
            captioner_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(captioner_args, AttrDict):
            captioner_args = AttrDict(captioner_args)

        self.captioner = VideoCaptioner(captioner_args)

    def __call__(self, frames):
        try:
            combined_caption = self.captioner(frames)
            return [
                combined_caption,
            ], None
        except Exception as err:  # pylint: disable=broad-except
            return [], str(err)
