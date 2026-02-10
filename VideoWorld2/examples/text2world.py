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

import torch
import torch.distributed

from videoworld2.pipelines.text2image import Text2ImagePipeline
from videoworld2.pipelines.video2world import Video2WorldPipeline

# Import functionality from other example scripts
from examples.text2image import process_single_generation as process_single_image_generation
from examples.text2image import setup_pipeline as setup_text2image_pipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, cleanup_distributed
from examples.video2world import process_single_generation as process_single_video_generation
from examples.video2world import setup_pipeline as setup_video2world_pipeline
from imaginaire.utils import log

_DEFAULT_POSITIVE_PROMPT = "An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to World Generation with Cosmos Predict2")
    # Common arguments between text2image and video2world
    parser.add_argument(
        "--model_size",
        choices=["2B", "14B"],
        default="2B",
        help="Size of the model to use for text2world generation",
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_POSITIVE_PROMPT, help="Text prompt for generation")
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'prompt' and 'output_video' fields.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video2world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=["1:1", "4:3", "3:4", "16:9", "9:16"],
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference for both text2image and video2world parts",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )

    # Text2image specific arguments
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the text2image inference.")

    # Video2world specific arguments
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
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value for video generation")
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory"
    )
    return parser.parse_args()


def generate_first_frames(text2image_pipe: Text2ImagePipeline, args: argparse.Namespace) -> list:
    """
    Generate first frames using the text2image pipeline.
    Returns a list of batch items containing prompt, output video path, and temp image path.
    """
    from megatron.core import parallel_state

    from imaginaire.utils.distributed import barrier, get_rank

    batch_items = []

    # Check if we're in a multi-GPU distributed environment
    is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()
    rank = get_rank() if is_distributed else 0

    # Only rank 0 should run text2image generation to avoid OOM when CP is disabled
    if rank == 0 and text2image_pipe is not None:
        if args.batch_input_json is not None:
            # Process batch inputs from JSON file
            log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
            with open(args.batch_input_json, "r") as f:
                batch_inputs = json.load(f)

            # Generate all the first frames first
            for idx, item in enumerate(batch_inputs):
                log.info(f"Generating first frame {idx + 1}/{len(batch_inputs)}")
                prompt = item.get("prompt", "")
                output_video = item.get("output_video", f"output_{idx}.mp4")

                if not prompt:
                    log.warning(f"Skipping item {idx}: Missing prompt")
                    continue

                # Save the generated first frame with a temporary name based on the output video path
                temp_image_name = os.path.splitext(output_video)[0] + "_temp.jpg"

                # Use the imported process_single_image_generation function
                if process_single_image_generation(
                    pipe=text2image_pipe,
                    prompt=prompt,
                    output_path=temp_image_name,
                    negative_prompt=args.negative_prompt,
                    aspect_ratio=args.aspect_ratio,
                    seed=args.seed,
                    use_cuda_graphs=args.use_cuda_graphs,
                    benchmark=args.benchmark,
                ):
                    # Save the item for the second stage
                    batch_items.append(
                        {"prompt": prompt, "output_video": output_video, "temp_image_path": temp_image_name}
                    )
        else:
            # Single item processing
            temp_image_path = os.path.splitext(args.save_path)[0] + "_temp.jpg"

            if args.use_cuda_graphs:
                log.warning(
                    "Using CUDA Graphs for a single inference call may not be beneficial because of overhead of Graphs creation."
                )

            # Use the imported process_single_image_generation function
            if process_single_image_generation(
                pipe=text2image_pipe,
                prompt=args.prompt,
                output_path=temp_image_path,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                use_cuda_graphs=args.use_cuda_graphs,
                benchmark=args.benchmark,
            ):
                # Add single item to batch_items for consistent processing
                batch_items.append(
                    {"prompt": args.prompt, "output_video": args.save_path, "temp_image_path": temp_image_path}
                )

        log.info(f"Rank 0: Generated {len(batch_items)} first frames")
    else:
        # Non-rank-0 processes: just wait for broadcast
        log.info(f"Rank {rank}: Waiting for batch_items from rank 0")
        batch_items = []  # Initialize empty list for non-rank-0 processes

    # Broadcast batch_items from rank 0 to all other ranks using PyTorch's broadcast_object_list
    if is_distributed:
        batch_items_list = [batch_items]  # Wrap in list for broadcast_object_list
        torch.distributed.broadcast_object_list(batch_items_list, src=0)
        batch_items = batch_items_list[0]  # Extract the broadcasted list

        if rank != 0:
            log.info(f"Rank {rank}: Received {len(batch_items)} batch items from rank 0")

        barrier()
        log.info(f"Rank {rank}: Synchronized after batch_items broadcast")

    return batch_items


def generate_videos(video2world_pipe: Video2WorldPipeline, batch_items: list, args: argparse.Namespace) -> None:
    """
    Generate videos from first frames using the video2world pipeline.
    """
    # Process all items for video generation
    for idx, item in enumerate(batch_items):
        log.info(f"Generating video from first frame {idx + 1}/{len(batch_items)}")
        prompt = item["prompt"]
        output_video = item["output_video"]
        temp_image_path = item["temp_image_path"]

        # Use the imported process_single_video_generation function
        process_single_video_generation(
            pipe=video2world_pipe,
            input_path=temp_image_path,
            prompt=prompt,
            output_path=output_video,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=1,  # Always use 1 frame for text2world
            guidance=args.guidance,
            seed=args.seed,
            benchmark=args.benchmark,
            use_cuda_graphs=args.use_cuda_graphs,
        )

        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            log.success(f"Cleaned up temporary image: {temp_image_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        from megatron.core import parallel_state

        from imaginaire.utils.distributed import get_rank

        if args.benchmark:
            log.warning(
                "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
            )

        # Check if we're in a multi-GPU distributed environment
        is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()
        rank = get_rank() if is_distributed else 0

        # Step 1: Initialize text2image pipeline and generate all first frames
        # Only rank 0 initializes the text2image pipeline to avoid OOM
        text2image_pipe = None
        text_encoder = None

        log.info("Step 1: Initializing text2image pipeline...")
        text2image_pipe = setup_text2image_pipeline(args)

        # Handle the case where setup_text2image_pipeline returns None for non-rank-0 processes
        if text2image_pipe is not None:
            # Store text encoder for later use (only on rank 0)
            text_encoder = text2image_pipe.text_encoder
            log.info("Rank 0: Text2image pipeline initialized successfully")
        else:
            # Non-rank-0 processes get None
            text_encoder = None
            log.info(f"Rank {rank}: Text2image pipeline setup returned None (expected for non-rank-0)")

        # Generate first frames (only rank 0 does actual generation)
        log.info("Step 1: Generating first frames...")
        batch_items = generate_first_frames(text2image_pipe, args)

        # Clean up text2image pipeline on rank 0
        if text2image_pipe is not None:
            log.info("Step 1 complete. Cleaning up text2image pipeline to free memory...")
            del text2image_pipe
            torch.cuda.empty_cache()

        # Step 2: Initialize video2world pipeline and generate videos
        log.info("Step 2: Initializing video2world pipeline...")

        # For non-rank-0 processes, let video2world create its own text encoder
        # This avoids the complexity of broadcasting the text encoder object across ranks
        if is_distributed and rank != 0:
            text_encoder = None
            log.info(f"Rank {rank}: Will create new text encoder for video2world pipeline")

        # Pass all video2world relevant arguments and the text encoder
        video2world_pipe = setup_video2world_pipeline(args, text_encoder=text_encoder)

        # Generate videos
        log.info("Step 2: Generating videos from first frames...")
        generate_videos(video2world_pipe, batch_items, args)

        # Clean up video2world pipeline
        log.info("All done. Cleaning up video2world pipeline...")
        del video2world_pipe
        torch.cuda.empty_cache()

    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
