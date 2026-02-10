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
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)

from imaginaire.utils import log

SYSTEM_PROMPT_REFINE = (
    """You are a prompt optimization specialist whose goal is to rewrite the user's input prompts into high-quality English prompts by referring to the details of the user's input images, making them more complete and expressive while maintaining the original meaning. You need to integrate the content of the user's photo with the input prompt for the rewrite, strictly adhering to the formatting of the examples provided.\n"""
    """Task Requirements:\n"""
    """1. For overly brief user inputs, reasonably infer and supplement details without changing the original meaning, making the image more complete and visually appealing;\n"""
    """2. Improve the characteristics of the main subject in the user's description (such as appearance, actions, expression, quantity, ethnicity, posture, surrounding environment etc.), rendering style, spatial relationships, and camera angles;\n"""
    """3. The overall output should be in English, retaining original text in quotes and book titles as well as important input information without rewriting them;\n"""
    """4. The prompt should match the user’s intent and provide a precise and detailed style description. If the user has not specified a style, you need to carefully analyze the style of the user's provided photo and use that as a reference for rewriting;\n"""
    """5. You need to emphasize movement information in the input and different camera angles;\n"""
    """6. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible;\n"""
    """7. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo;\n"""
    """8. Control the rewritten prompt to around 80-100 words.\n"""
    """9. No matter what language the user inputs, you must always output in English.\n"""
    """Example of the rewritten English prompt:\n"""
    """1. A Japanese fresh film-style video of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The video has a vintage film texture. A medium shot of a seated portrait.\n"""
    """2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "Ziyang". The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.\n"""
    """3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.\n"""
    """4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.\n"""
    """Directly output the rewritten English text."""
)

SYSTEM_PROMPT_CRITIC = """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video.
Analyze the video carefully and answer the question according to the following template:

<think>
<overview>
[Brief description of the video.]
</overview>

<component name="Component 1 Name">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<component name="Component 2 Name">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<!-- Add more components as needed -->
</think>

<answer>
[Whether the video contains anomalies or artifacts. Answer "Yes" or "No".]
</answer>"""

USER_PROMPT_CRITIC = "Does the video contain any anomalies or artifacts?"


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_mask = torch.zeros(max(allowed_token_ids) + 1, dtype=torch.bool)
        self.allowed_mask[allowed_token_ids] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] > self.allowed_mask.shape[0]:
            self.allowed_mask = torch.nn.functional.pad(
                self.allowed_mask, (0, scores.shape[-1] - self.allowed_mask.shape[0]), value=False
            )

        # Vectorized operation: set all disallowed tokens to -inf at once
        device = scores.device

        # Move mask to the same device as scores if needed
        if self.allowed_mask.device != device:
            self.allowed_mask = self.allowed_mask.to(device)

        # Create mask for disallowed tokens and apply it vectorized
        disallowed_mask = ~self.allowed_mask[: scores.shape[-1]]
        scores[:, disallowed_mask] = float("-inf")
        return scores


class CosmosReason1(torch.nn.Module):
    def __init__(self, checkpoint_dir: str, offload_model_to_cpu: bool = True, enabled: bool = True):
        """Cosmos Reason1 model for prompt refinement.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
            enabled (bool, optional): Whether to enable the model. Defaults to True.
        """
        super().__init__()
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_dir, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            use_cache=True,
        )
        self.offload_model = offload_model_to_cpu
        self.enabled = enabled
        self._compute_allowed_token_ids()
        # move model to GPU if not offload_model_to_cpu and enabled
        if not offload_model_to_cpu and self.enabled:
            self.model = self.model.to("cuda")
            log.debug("Move Reason1 model to GPU")

    def _compute_allowed_token_ids(self):
        """Pre-compute allowed token IDs for ASCII characters to avoid repeated computation."""
        log.debug("Pre-computing allowed token IDs for ASCII characters...")
        # Get all token IDs
        all_token_ids = list(range(self.processor.tokenizer.vocab_size))
        # Batch decode all tokens at once (much faster than individual decodes)
        try:
            # Try batch decoding first (fastest)
            decoded_tokens = self.processor.tokenizer.batch_decode(
                [[i] for i in all_token_ids], skip_special_tokens=False
            )
            # Filter for ASCII-only tokens
            self.allowed_token_ids = [
                token_id
                for token_id, decoded in zip(all_token_ids, decoded_tokens)
                if all(ord(c) < 128 for c in decoded)
            ]
        except Exception:
            # Fallback to individual decoding if batch decode fails
            log.warning("Batch decode failed, falling back to individual token decoding...")
            self.allowed_token_ids = [
                i for i in all_token_ids if all(ord(c) < 128 for c in self.processor.tokenizer.decode([i]))
            ]
        # Add special tokens
        self.allowed_token_ids.extend(self.processor.tokenizer.all_special_ids)
        self.allowed_token_ids = list(set(self.allowed_token_ids))  # Remove duplicates
        log.debug(f"Computed {len(self.allowed_token_ids)} allowed token IDs")

    def prepare_dialog(self, image_or_video_path: str, prompt: str) -> list[dict]:
        log.debug(f"Preparing dialog for {image_or_video_path} with prompt {prompt}")
        if image_or_video_path.endswith(".mp4"):
            type = "video"
        elif image_or_video_path.endswith(".jpg") or image_or_video_path.endswith(".png"):
            type = "image"
        else:
            raise ValueError(f"Unsupported file type: {image_or_video_path}")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_REFINE,
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": type, type: image_or_video_path}]},
        ]
        return messages

    def refine_prompt(self, image_or_video_path: str, prompt: str) -> str:
        # skip prompt refinement if not enabled
        if not self.enabled:
            return prompt

        # prompt refinement
        dialog = self.prepare_dialog(image_or_video_path, prompt)
        if self.offload_model:
            self.model = self.model.to("cuda")
            log.debug("Move Reason1 model to GPU")
        text = self.processor.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(dialog)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Inference: Generation of the output
        inputs = inputs.to("cuda")
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(self.allowed_token_ids)])
        generated_ids = self.model.generate(**inputs, max_new_tokens=512, logits_processor=logits_processor)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        log.debug(f"Upsampled prompt: {output_text}")
        if self.offload_model:
            self.model = self.model.to("cpu")
            log.debug("Offload Reason1 model to CPU")
        return output_text

    def analyze_video(self, video_path: str, num_trials: int = 1, seed: int | None = None) -> str:
        if self.offload_model:
            self.model = self.model.to("cuda")
            log.debug("Move Reason1 model to GPU")
        dialog = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_CRITIC,
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 16, "total_pixels": 8192 * 28 * 28},
                    {"type": "text", "text": USER_PROMPT_CRITIC},
                ],
            },
        ]
        text = self.processor.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(dialog, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        # Inference: Generation of the output
        inputs = inputs.to("cuda")
        if seed is not None:
            set_seed(seed)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            num_return_sequences=num_trials,
            max_new_tokens=4096,
        )
        in_len = len(inputs.input_ids[0])
        generated_ids_trimmed = [out_ids[in_len:] for out_ids in generated_ids]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        if self.offload_model:
            self.model = self.model.to("cpu")
            log.debug("Offload Reason1 model to CPU")
        return output_text


if __name__ == "__main__":
    model = CosmosReason1("checkpoints/nvidia/Cosmos-Reason1-7B")
    print(model.refine_prompt("assets/video2world/input0.jpg", "A bus terminal in the city."))
