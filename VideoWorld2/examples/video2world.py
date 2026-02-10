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

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import torch
from megatron.core import parallel_state

from videoworld2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B,
)
from videoworld2.pipelines.video2world import _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS, Video2WorldPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def validate_input_file(input_path: str, num_conditional_frames: int) -> bool:
    if not os.path.exists(input_path):
        log.warning(f"Input file does not exist, skipping: {input_path}")
        return False

    ext = os.path.splitext(input_path)[1].lower()

    if num_conditional_frames == 1:
        # Single frame conditioning: accept both images and videos
        if ext not in _IMAGE_EXTENSIONS and ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file with unsupported extension for single frame conditioning: {input_path} "
                f"(expected: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS})"
            )
            return False
    elif num_conditional_frames == 5:
        # Multi-frame conditioning: only accept videos
        if ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file for multi-frame conditioning (requires video): {input_path} "
                f"(expected: {_VIDEO_EXTENSIONS}, got: {ext})"
            )
            return False
    else:
        log.error(f"Invalid num_conditional_frames: {num_conditional_frames} (must be 1 or 5)")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B", "14B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--resolution",
        choices=["480", "720"],
        default="720",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=[10, 16],
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video-to-world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=["1:1", "4:3", "3:4", "16:9", "9:16"],
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1, 5],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'input_video', 'prompt', and 'output_video' fields.",
    )
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
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the text2image inference.")
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace, text_encoder=None):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B

        config.resolution = args.resolution
        if args.fps == 10:  # default is 16 so no need to change config
            config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-{args.resolution}p-{args.fps}fps.pt"
    elif args.model_size == "14B":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_14B

        config.resolution = args.resolution
        if args.fps == 10:  # default is 16 so no need to change config
            config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-{args.resolution}p-{args.fps}fps.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    # Only set up text encoder path if no encoder is provided
    text_encoder_path = None if text_encoder is not None else "checkpoints/google-t5/t5-11b"
    log.info(f"Using dit_path: {dit_path}")
    if text_encoder is not None:
        log.info("Using provided text encoder")
    else:
        log.info(f"Using text encoder from: {text_encoder_path}")

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

        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        else:
            log.info("Distributed environment already initialized, skipping initialization")
            # Check if we need to reinitialize with different context parallel size
            current_cp_size = parallel_state.get_context_parallel_world_size()
            if current_cp_size != args.num_gpus:
                log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                log.warning("Using existing context parallel configuration")
            else:
                log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False
    config.prompt_refiner_config.offload_model_to_cpu = args.offload_prompt_refiner

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=True,
    )

    # Set the provided text encoder if one was passed
    if text_encoder is not None:
        pipe.text_encoder = text_encoder

    return pipe


def process_single_generation(
    pipe: Video2WorldPipeline,
    input_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
    benchmark: bool = False,
    use_cuda_graphs: bool = False,
) -> bool:
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {prompt}")

    num_repeats = 4 if benchmark else 1
    time_sum = 0
    for i in range(num_repeats):
        if benchmark and i > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            input_path=input_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_sum += elapsed
            log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")
    if benchmark:
        time_avg = time_sum / (num_repeats - 1)
        log.critical(f"Average generation time for Video2WorldPipeline is {time_avg:.1f} seconds.")

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        if pipe.config.state_t == 16:
            fps = 10
        else:
            fps = 16
        save_image_or_video(video, output_path, fps=fps)
        log.success(f"Successfully saved video to: {output_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldPipeline) -> None:
    if args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
        )
    # Video-to-World
    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json, "r") as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(batch_inputs):
            log.info(f"Processing batch item {idx + 1}/{len(batch_inputs)}")
            input_video = item.get("input_video", "")
            prompt = item.get("prompt", "")
            output_video = item.get("output_video", f"output_{idx}.mp4")

            if not input_video or not prompt:
                log.warning(f"Skipping item {idx}: Missing input_video or prompt")
                continue

            process_single_generation(
                pipe=pipe,
                input_path=input_video,
                prompt=prompt,
                output_path=output_video,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=args.seed,
                benchmark=args.benchmark,
                use_cuda_graphs=args.use_cuda_graphs,
            )
    else:
        process_single_generation(
            pipe=pipe,
            input_path=args.input_path,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            benchmark=args.benchmark,
            use_cuda_graphs=args.use_cuda_graphs,
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
