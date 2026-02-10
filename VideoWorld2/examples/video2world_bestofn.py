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
import base64
import functools
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional

import torch

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from videoworld2.auxiliary.cosmos_reason1 import CosmosReason1
from videoworld2.pipelines.video2world import Video2WorldPipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, cleanup_distributed
from examples.video2world import process_single_generation as process_single_generation_default
from examples.video2world import setup_pipeline as setup_pipeline_default
from examples.video2world import validate_input_file
from examples.video2world_gr00t import process_single_generation as process_single_generation_gr00t
from examples.video2world_gr00t import setup_pipeline as setup_pipeline_gr00t
from imaginaire.utils import log
from imaginaire.utils.distributed import get_rank


def parse_response(response: str) -> Optional[Dict[str, Any]]:
    try:
        wrapped = f"<root>{response.strip()}</root>"
        root = ET.fromstring(wrapped)

        result = {"think": {}, "answer": None}

        # Parse <think> section
        think_element = root.find("think")
        if think_element is not None:
            # Parse overview
            overview = think_element.find("overview")
            result["think"]["overview"] = overview.text.strip() if overview is not None and overview.text else ""

            # Parse components
            result["think"]["components"] = []
            for comp in think_element.findall("component"):
                component_data = {"name": comp.get("name", "")}

                analysis = comp.find("analysis")
                component_data["analysis"] = analysis.text.strip() if analysis is not None and analysis.text else ""

                anomaly = comp.find("anomaly")
                component_data["anomaly"] = anomaly.text.strip() if anomaly is not None and anomaly.text else ""

                result["think"]["components"].append(component_data)

        # Parse <answer> section
        answer_element = root.find("answer")
        result["answer"] = answer_element.text.strip() if answer_element is not None and answer_element.text else ""

        return result
    except Exception:
        return None


def video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def build_html_report(video_path: str, responses: List[str]) -> str:
    # Convert video to base64
    video_base64 = video_to_base64(video_path)
    mime_type = "video/mp4"

    # Parse responses
    parsed_responses = [parse_response(response) for response in responses]
    valid_responses = [r for r in parsed_responses if r is not None]

    # Count answers
    yes_count = sum(1 for r in valid_responses if r.get("answer", "").lower() == "yes")
    no_count = sum(1 for r in valid_responses if r.get("answer", "").lower() == "no")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cosmos-Reason1 Video Analysis Report - {os.path.basename(video_path)}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        video {{ width: 100%; max-width: 600px; }}
        .red {{ background-color: #ffebee; color: #c62828; padding: 10px; margin: 5px 0; }}
        .green {{ background-color: #e8f5e8; color: #2e7d32; padding: 10px; margin: 5px 0; }}
        .trial {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 15px; }}
    </style>
</head>
<body>
    <h1>Cosmos-Reason1 Video Analysis Report</h1>
    <p>File: {os.path.basename(video_path)}</p>

    <h2>Video</h2>
    <video controls>
        <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
    </video>

    <h2>Summary</h2>
    <div class="stats">
        <div class="stat red">
            <div style="font-size: 24px; font-weight: bold;">{yes_count}</div>
            <div>Anomaly Detected</div>
        </div>
        <div class="stat green">
            <div style="font-size: 24px; font-weight: bold;">{no_count}</div>
            <div>No Anomaly</div>
        </div>
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{len(valid_responses)}</div>
            <div>Total Responses</div>
        </div>
    </div>

    <h2>Detailed Analysis ({len(responses)} trials)</h2>
"""

    for i, (response, parsed) in enumerate(zip(responses, parsed_responses), 1):
        if parsed is not None:
            answer = parsed.get("answer", "").lower()
            answer_class = "red" if answer == "yes" else "green"

            html += f"""
    <div class="trial">
        <h3>Trial {i}</h3>
"""

            # Overview
            if parsed.get("think", {}).get("overview"):
                html += f"""
        <p><strong>Overview:</strong> {parsed["think"]["overview"]}</p>
"""

            # Components
            if parsed.get("think", {}).get("components"):
                for comp in parsed["think"]["components"]:
                    anomaly = comp.get("anomaly", "").lower()
                    comp_class = "red" if anomaly == "yes" else "green"
                    html += f"""
        <div class="{comp_class}">
            <strong>{comp.get('name', 'Unknown Component')} - {comp.get('anomaly', '')}</strong>
            <p>{comp.get('analysis', 'No analysis provided')}</p>
        </div>
"""

            # Final answer
            html += f"""
        <div class="{answer_class}">
            <strong>Final Answer: {parsed.get("answer", "No answer")}</strong>
        </div>
    </div>
"""

    html += """
</body>
</html>"""

    return html


def count_answers(responses: List[str]) -> tuple[int, int]:
    no_count = 0
    total_parsed = 0
    for response in responses:
        # Look for <answer>Yes</answer> or <answer>No</answer> pattern (case insensitive)
        answer_match = re.search(r"<answer>\s*(yes|no)\s*</answer>", response, re.IGNORECASE)
        if answer_match:
            total_parsed += 1
            if answer_match.group(1).lower() == "no":
                no_count += 1
    return no_count, total_parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Best-of-N Video Generation with Cosmos Predict2")
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
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path", type=str, default="output/best-of-n", help="Directory to save the generated videos and reports"
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
    # GR00T-specific settings. Specify --gr00t_variant to enable
    parser.add_argument("--gr00t_variant", type=str, default="", help="GR00T variant to use", choices=["gr1", "droid"])
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix to add to all prompts"
    )
    # Rejection sampling settings
    parser.add_argument("--num_generations", type=int, default=2, help="Number of generations for the input")
    parser.add_argument(
        "--skip_generation", action="store_true", help="Skip video generation and only run the critic model"
    )
    parser.add_argument("--num_critic_trials", type=int, default=5, help="Number of critic trials for each generation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/nvidia/Cosmos-Reason1-7B",
        help="Path to the Cosmos-Reason1 checkpoint",
    )
    return parser.parse_args()


def generate_video(
    args: argparse.Namespace, pipe: Video2WorldPipeline, process_single_generation: Callable
) -> List[str]:
    if not validate_input_file(args.input_path, args.num_conditional_frames):
        log.error(f"Input file validation failed: {args.input_path}")
        return []

    prompt = args.prompt
    log.info(f"Running Video2WorldPipeline with \ninput: {args.input_path}\nprompt: {prompt}")

    video_paths = []
    input_basename = os.path.splitext(os.path.basename(args.input_path))[0]

    # Ensure the output directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Generate multiple videos with different seeds
    for generation_idx in range(args.num_generations):
        current_seed = args.seed + generation_idx
        output_path = os.path.join(args.save_path, f"{input_basename}_seed{current_seed}.mp4")

        log.info(f"Generating video {generation_idx + 1}/{args.num_generations} with seed {current_seed}")

        # Call the original process_single_generation function directly
        success = process_single_generation(
            pipe=pipe,
            input_path=args.input_path,
            prompt=prompt,
            output_path=output_path,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=current_seed,
        )

        if success:
            video_paths.append(output_path)
            log.success(f"Successfully generated video at {output_path}")
        else:
            log.warning(f"Failed to generate video for seed {current_seed}")

    log.info(f"Generated {len(video_paths)}/{args.num_generations} videos")
    return video_paths


def run_critic(args, video_paths):
    if not video_paths:
        log.warning("No videos to analyze")
        return []

    log.info(f"Initializing CosmosReason1 critic model from {args.checkpoint_dir}")
    reason1 = CosmosReason1(args.checkpoint_dir, offload_model_to_cpu=False, enabled=True)

    scores = []
    for video_path in video_paths:
        log.info(f"Analyzing video: {video_path}")
        responses = reason1.analyze_video(video_path, args.num_critic_trials, args.seed)

        # Count "No" (no anomaly) answers and total successful parses
        no_count, total_parsed = count_answers(responses)

        if total_parsed == 0:
            log.warning(f"No valid responses were parsed for {video_path}")
            score = 0.0
        else:
            score = no_count / total_parsed

        # Create output filename in the same directory as the video
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        score_pct = int(score * 100)
        output_filename = os.path.join(video_dir, f"{video_name}_score{score_pct:03d}")

        # Save raw responses to JSON
        with open(output_filename + ".json", "w") as f:
            json.dump(responses, f, indent=2)
        log.success(f"Saved {len(responses)} responses to {output_filename}.json")
        log.info(
            f"Analysis results: {no_count} 'No anomaly' answers out of {total_parsed} valid responses ({score:.2%})"
        )

        # Generate HTML report in the same directory as the video
        html_content = build_html_report(video_path, responses)
        with open(output_filename + ".html", "w", encoding="utf-8") as f:
            f.write(html_content)
        log.success(f"Generated HTML report: {output_filename}.html")

        scores.append(score)

    return scores


if __name__ == "__main__":
    args = parse_args()

    # Handle GR00T-specific settings if gr00t_variant is provided
    if args.gr00t_variant:
        setup_pipeline = setup_pipeline_gr00t
        process_single_generation = functools.partial(process_single_generation_gr00t, prompt_prefix=args.prompt_prefix)
    else:
        setup_pipeline = setup_pipeline_default
        process_single_generation = process_single_generation_default

    # Create output directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        log.info(f"Created output directory: {args.save_path}")

    rank = 0
    # Generate videos if not provided
    video_paths = []
    if args.skip_generation:
        log.info(f"Skipping generation, looking for existing videos in {args.save_path}")
        video_paths = glob.glob(os.path.join(args.save_path, "*.mp4"))
        if not video_paths:
            log.warning(f"No video files found in {args.save_path}")
    else:
        try:
            pipe = setup_pipeline(args)
            rank = get_rank()  # Get rank before destroying the process group
            video_paths = generate_video(args, pipe, process_single_generation)
            if not video_paths:
                log.error("Failed to generate any videos, exiting")
                exit(1)
        except Exception as e:
            log.error(f"Error during video generation: {e}")
            exit(1)
        finally:
            cleanup_distributed()

    del pipe  # Free up memory
    torch.cuda.empty_cache()

    # Run the critic model on rank 0 only
    if rank == 0:
        try:
            scores = run_critic(args, video_paths)

            # Print the final results
            if scores:
                log.info("\nFinal Results (highest score first):")
                log.info("-" * 80)

                # Sort videos by score (highest first)
                sorted_results = sorted(zip(scores, video_paths), key=lambda x: x[0], reverse=True)

                for i, (score, video_path) in enumerate(sorted_results, 1):
                    video_name = os.path.basename(video_path)
                    log.info(f"#{i}: Score: {score:.2%} - {video_name}")

                # Print the best video path
                best_score, best_video = sorted_results[0]
                log.info("-" * 80)
                log.success(f"Best video: {os.path.basename(best_video)} with score {best_score:.2%}")
                log.success(f"Full path: {best_video}")
            else:
                log.warning("No scores available")
        except Exception as e:
            log.error(f"Error during critic evaluation: {e}")
            exit(1)
