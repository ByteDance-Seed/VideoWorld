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
import pdb

import mediapy as mp
import numpy as np

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from videoworld2.configs.action_conditioned.config_action_conditioned import (
    ACTION_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
)
from videoworld2.pipelines.action_video2world import ActionConditionedVideo2WorldPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video


def get_action_sequence(annotation_path):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    # rescale the action to the original scale
    action_ee = np.array(data["action"])[:, :6] * 20
    gripper = np.array(data["continuous_gripper_state"])[1:, None]

    # concatenate the end-effector displacement and gripper width
    action = np.concatenate([action_ee, gripper], axis=1)
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--input_annotation",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
        help="Chunk size",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = ACTION_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pth"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    # text_encoder_path = "checkpoints/google-t5/t5-11b"
    text_encoder_path = ""

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = ActionConditionedVideo2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=True,
    )

    return pipe


def read_first_frame(video_path):
    video = mp.read_video(video_path)  # Returns (T, H, W, C) numpy array
    return video[0]  # Return first frame as numpy array


def process_single_generation(
    pipe, input_path, input_annotation, output_path, guidance, seed, chunk_size, autoregressive
):
    actions = get_action_sequence(input_annotation)
    first_frame = read_first_frame(input_path)

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}")

    if autoregressive:
        log.info("Using autoregressive mode")
        video_chunks = []
        for i in range(0, len(actions), chunk_size):
            if actions[i : i + chunk_size].shape[0] < chunk_size:
                log.info("Reached end of actions")
                break
            video = pipe(
                first_frame,
                actions[i : i + chunk_size],
                num_conditional_frames=1,
                guidance=guidance,
                seed=i,
            )
            first_frame = ((video[0, :, -1].permute(1, 2, 0).cpu().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            video_chunks.append(video)
        video = torch.cat([video_chunks[0]] + [chunk[:, :, :-1] for chunk in video_chunks[1:]], dim=2)
    else:
        video = pipe(
            first_frame,
            actions[:chunk_size],
            num_conditional_frames=1,
            guidance=guidance,
            seed=seed,
        )

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=4)
        log.success(f"Successfully saved video to: {output_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: ActionConditionedVideo2WorldPipeline) -> None:
    process_single_generation(
        pipe=pipe,
        input_path=args.input_video,
        input_annotation=args.input_annotation,
        output_path=args.save_path,
        guidance=args.guidance,
        seed=args.seed,
        chunk_size=args.chunk_size,
        autoregressive=args.autoregressive,
    )
    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
