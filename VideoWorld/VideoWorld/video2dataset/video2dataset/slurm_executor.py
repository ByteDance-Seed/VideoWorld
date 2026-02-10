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
"""Executor for distribution via slurm"""

import os
import yaml
import fire

from video2dataset import video2dataset


def executor(worker_args, node_id, n_nodes, num_tasks_per_node, subtask_id):
    """
    Spins up the individual workers on the nodes
    :param worker_args: parameters to video2dataset on the respective node
    :param node_id: node id
    :param n_nodes: overall number of node
    :param num_tasks_per_node: number of distinct calls to video2dataset from the current node
    :param subtask_id: local id of current call to video2dataset
    :return:
    """

    num_tasks = n_nodes * num_tasks_per_node
    print("#" * 100)
    print("args:")
    print(worker_args)
    print("node id", node_id)
    print("n_nodes", n_nodes)
    print("num_tasks_per_node", num_tasks_per_node)
    print("subtask_id", subtask_id)
    print("num_tasks", num_tasks)
    print("#" * 100)

    global_task_id = node_id * num_tasks_per_node + subtask_id
    assert global_task_id < num_tasks, (
        f"global_task_id is {global_task_id} but must be less than "
        f"num_nodes*num_tasks_per_node={n_nodes * num_tasks_per_node}"
    )

    print(f"Starting task with id {global_task_id}")
    os.environ["GLOBAL_RANK"] = str(global_task_id)
    os.environ["LOCAL_RANK"] = str(subtask_id)

    # Read the worker args from the file
    with open(worker_args, "r", encoding="utf-8") as worker_args_file:
        worker_args = yaml.load(worker_args_file, Loader=yaml.SafeLoader)

    # call main script from every subprocess
    video2dataset(**worker_args)


if __name__ == "__main__":
    fire.Fire(executor)
