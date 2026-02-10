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
# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
import shutil
import cv2
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from utils.calvin_utils import print_and_save

logger = logging.getLogger(__name__)

# Path to calvin
CALVIN_ROOT = os.environ['CALVIN_ROOT']

EP_LEN = 360
NUM_SEQUENCES = 1000


def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    # import pdb;pdb.set_trace()
    for subtask_i, subtask in enumerate(eval_sequence):
        success, index = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    print(subtask)
    save_root = '/opt/tiger/gr1_rollout'
    temp_save_root = '/opt/tiger/temp_gr1_rollout'
    os.system(f'mkdir -p {temp_save_root}')
    start_index = len(os.listdir(save_root))
    cur_index = start_index
    for step in range(EP_LEN):
        npz_path = f'{temp_save_root}/episode_{cur_index:07d}.npz'
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        np.savez(npz_path, rel_actions = action.numpy(), robot_obs = obs['robot_obs'], rgb_static=obs['rgb_obs']['rgb_static'], rgb_gripper=obs['rgb_obs']['rgb_gripper'])
        vis_path = f'{temp_save_root}/episode_{cur_index:07d}.png'
        # import pdb;pdb.set_trace()
        # cv2.imwrite(vis_path, obs['rgb_obs']['rgb_static'][:, :, ::-1])
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        
        if len(current_task_info) > 0:
            for file in os.listdir(temp_save_root):
                os.system(f'mv {os.path.join(temp_save_root, file)} {save_root}')
            with open('/opt/tiger/gr1_rollout.txt', 'a') as f:
                f.write(f'{start_index} {cur_index} {subtask} \n')
            f.close()
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True, (start_index, cur_index)
        cur_index += 1
    os.system(f'rm -r {temp_save_root}')
    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False, (start_index, cur_index)


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_dir", default='task_ABCD_D/', type=str, help="Dataset directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument('--eval_dir', type=str, help="Directory to save evaluation results")
    parser.add_argument('--policy_ckpt_path', type=str, help="Path to the evaluating checkpoint")
    parser.add_argument('--mae_ckpt_path', type=str, help="Path to the MAE checkpoint")
    parser.add_argument('--configs_path', type=str, help="Path to the config json file")
    parser.add_argument('--device', default=0, type=int, help="CUDA device")    
    args = parser.parse_args()

    
    with open(args.configs_path, "r") as f:
        variant = json.load(f)
    device = torch.device('cuda', args.device)
    model = GR1CalvinEvaluation(
        mae_ckpt=args.mae_ckpt_path,
        policy_ckpt=args.policy_ckpt_path,
        variant=variant,
        device=device)
    
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    env = make_env(args.dataset_dir, observation_space, args.device)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(args.eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(args.eval_dir, "result.txt")
    evaluate_policy(
        model, 
        env,
        eval_sr_path, 
        eval_result_path, 
        args.eval_dir,
        debug=args.debug)


if __name__ == "__main__":
    main()