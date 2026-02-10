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
"""distributor defines the distribution strategies for img2dataset"""
import os
import time
import subprocess
import yaml
from datetime import datetime
from contextlib import contextmanager
from multiprocessing import get_context
from itertools import islice, chain

import fsspec
from tqdm import tqdm


def retrier(runf, failed_shards, max_shard_retry):
    # retry failed shards max_shard_retry times
    for i in range(max_shard_retry):
        if len(failed_shards) == 0:
            break
        print(f"Retrying {len(failed_shards)} shards, try {i+1}")
        failed_shards = runf(failed_shards)
    if len(failed_shards) != 0:
        print(
            f"Retried {max_shard_retry} times, but {len(failed_shards)} shards "
            "still failed. You may restart the same command to retry again."
        )


def no_distributor(process_count, worker, input_sharder, _, max_shard_retry):  # pylint: disable=unused-argument
    """Go through shards sequentially (useful for when things don't like multiprocessing)"""

    def run(gen):
        failed_shards = []
        for shard in gen:
            status, row = worker(shard)
            if status is False:
                failed_shards.append(row)
        return failed_shards

    failed_shards = run(input_sharder)
    retrier(run, failed_shards, max_shard_retry)


def multiprocessing_distributor(processes_count, worker, input_sharder, _, max_shard_retry):
    """Distribute the work to the processes using multiprocessing"""
    ctx = get_context("spawn")
    with ctx.Pool(processes_count, maxtasksperchild=5) as process_pool:

        def run(gen):
            failed_shards = []
            for status, row in tqdm(process_pool.imap_unordered(worker, gen)):
                if status is False:
                    failed_shards.append(row)
            return failed_shards

        failed_shards = run(input_sharder)

        retrier(run, failed_shards, max_shard_retry)

        process_pool.terminate()
        process_pool.join()
        del process_pool


def pyspark_distributor(processes_count, worker, input_sharder, subjob_size, max_shard_retry):
    """Distribute the work to the processes using pyspark"""

    with _spark_session(processes_count) as spark:

        def batcher(iterable, batch_size):
            iterator = iter(iterable)
            for first in iterator:
                yield list(chain([first], islice(iterator, batch_size - 1)))

        def run(gen):
            failed_shards = []
            for batch in batcher(gen, subjob_size):
                rdd = spark.sparkContext.parallelize(batch, len(batch))
                for status, row in rdd.map(worker).collect():
                    if status is False:
                        failed_shards.append(row)
            return failed_shards

        failed_shards = run(input_sharder)

        retrier(run, failed_shards, max_shard_retry)


@contextmanager
def _spark_session(processes_count: int):
    """Create and close a spark session if none exist"""

    from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel
    import pyspark  # pylint: disable=import-outside-toplevel

    spark_major_version = int(pyspark.version.__version__[0])
    if spark_major_version >= 3:
        spark = SparkSession.getActiveSession()
    else:
        spark = pyspark.sql.SparkSession._instantiatedSession  # pylint: disable=protected-access

    if spark is None:
        print("No pyspark session found, creating a new one!")
        owned = True
        spark = (
            SparkSession.builder.config("spark.driver.memory", "16G")
            .master("local[" + str(processes_count) + "]")
            .appName("spark-stats")
            .getOrCreate()
        )
    else:
        owned = False

    try:
        yield spark
    finally:
        if owned:
            spark.stop()


class SlurmShardSampler:
    """
    Should be callable to select samples based on the node_id
    :param global_task_id: The global task id for the current task
    :param num_tasks: The overall number of tasks
    :return:
    """

    def __init__(self, global_task_id, num_tasks):
        self.task_id = global_task_id
        print(global_task_id)
        self.num_tasks = num_tasks

    def __call__(self, shardfile_list):
        shardlist = [
            (full_shard_id, shard_id)
            for full_shard_id, shard_id in shardfile_list
            if int(full_shard_id) % self.num_tasks == self.task_id
        ]
        return shardlist


class SlurmDistributor:
    """Parallelism via slurm"""

    def __init__(
        self,
        worker_args,
        cpus_per_task,
        job_name,
        partition,
        n_nodes,
        account,
        gpus_per_node=0,
        tasks_per_node=1,
        nodelist=None,
        constraint=None,
        exclude=None,
        cache_path=None,
        timeout=None,
        verbose_wait=False,
    ):
        self.cpus_per_task = cpus_per_task
        self.job_name = job_name
        self.partition = partition
        self.n_nodes = n_nodes
        self.gpus_per_node = gpus_per_node
        self.account = account
        self.tasks_per_node = tasks_per_node
        self.nodelist = nodelist
        self.constraint = constraint
        self.exclude = exclude
        self.cache_path = cache_path
        if not cache_path:
            cache_path = ".video2dataset_cache/"
        self.timeout = timeout
        self.verbose_wait = verbose_wait

        self.fs, self.cache_path = fsspec.core.url_to_fs(cache_path)
        if not self.fs.exists(self.cache_path):
            self.fs.mkdir(self.cache_path)

        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # save worker args to file (this is written by the slurm_executor)
        self.worker_args_as_file = os.path.join(self.cache_path, f"{self.timestamp}_worker_args.yaml")
        with self.fs.open(self.worker_args_as_file, "w", encoding="utf-8") as f:
            yaml.dump(worker_args, f, default_flow_style=False)

        self.launcher_path = os.path.join(self.cache_path, self.timestamp + "_launcher.sh")
        with self.fs.open(self.launcher_path, "w", encoding="utf-8") as launcher_file:
            launcher_file.write(self._make_launch_cpu())

        self.sbatch_path = os.path.join(self.cache_path, self.timestamp + "_sbatch.sh")
        with self.fs.open(self.sbatch_path, "w", encoding="utf-8") as sbatch_file:
            sbatch_file.write(self._make_sbatch())

        print(f"Wrote launcher to {self.launcher_path}")
        print(f"Wrote sbatch to {self.sbatch_path}")

    def _make_sbatch(self):
        nodelist = ("#SBATCH --nodelist " + self.nodelist) if self.nodelist is not None else ""
        exclude = ("#SBATCH --exclude " + self.exclude) if self.exclude is not None else ""
        account = ("#SBATCH --account " + self.account) if self.account is not None else ""
        constraint = ("#SBATCH --constraint " + self.constraint) if self.constraint is not None else ""
        return f"""#!/bin/bash
#SBATCH --partition={self.partition}
#SBATCH --job-name={self.job_name}
#SBATCH --output={self.cache_path}/slurm-%x_%j.out
#SBATCH --nodes={self.n_nodes}
#SBATCH --ntasks-per-node={self.tasks_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --gpus-per-node={self.gpus_per_node}
#SBATCH --exclusive
{nodelist}
{exclude}
{account}
{constraint}
#SBATCH --open-mode append

srun --account {self.account} bash {self.launcher_path}

"""

    def _make_launch_cpu(
        self,
    ):
        """Create cpu launcher"""

        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            venv_activate = f"source {venv}/bin/activate"
        else:
            conda_env = os.environ.get("CONDA_ENV")
            if conda_env:
                venv_activate = f"conda activate {conda_env}"
            else:
                raise ValueError("You need to specify either a virtual environment or a conda environment.")

        cdir = os.path.abspath(os.path.dirname(__file__))
        script = os.path.join(cdir, "slurm_executor.py")
        project_root = os.path.abspath(os.path.join(cdir, ".."))
        return f"""#!/bin/bash
# mpi version for node rank
H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
export NODE_RANK=${{THEID}}
echo THEID=$THEID

# set new location for cache dir
export XDG_CACHE_HOME="{self.cache_path}"
#in case of accessing s3 disable ssl verification (for making torchdata s3 related functionality work)
export S3_VERIFY_SSL=0
export CALLED_FROM_SLURM=1

cd {project_root}
{venv_activate}

python {script} --worker_args {self.worker_args_as_file} --node_id $SLURM_NODEID --n_nodes $SLURM_JOB_NUM_NODES --num_tasks_per_node $SLURM_NTASKS_PER_NODE --subtask_id $SLURM_LOCALID
"""

    def __call__(self, *args, **kwargs):
        print(f"{self.__class__.__name__} starting the job")

        status = self._run_job()

        # interpret the results
        if status == "success":
            print("job succeeded")
            return True
        elif status == "failed":
            print("job failed")
            return False
        else:
            print("exception occurred")
            return False

    def _start_job(self, sbatch_file):
        """start job"""
        args = ["sbatch"]
        args.append(sbatch_file)
        sbatch_output = subprocess.check_output(args).decode("utf8")
        lines = sbatch_output.split("\n")

        lines = [line for line in lines if "Submitted" in line]
        if len(lines) == 0:
            raise ValueError(f"slurm sbatch failed: {sbatch_output}")

        parsed_sbatch = lines[0].split(" ")
        job_id = parsed_sbatch[3].strip()
        return job_id

    def _run_job(
        self,
    ):
        """
        Run a job and wait for it to finish.
        """
        try:
            job_id = self._start_job(self.sbatch_path)

            print(f"waiting for job {job_id}")

            timeout = self.timeout

            if timeout is None:
                print("You have not specified a timeout, defaulting to 2 weeks.")
                timeout = 1.21e6

            status = self._wait_for_job_to_finish(job_id=job_id, timeout=timeout)

            if not status:
                print(f"canceling {job_id}")
                subprocess.check_output(["scancel", job_id]).decode("utf8")
                status = self._wait_for_job_to_finish(job_id)
                print("job cancelled")
                return "failed"
            else:
                print("job succeeded")
                return "success"
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            return "exception occurred"

    def _wait_for_job_to_finish(self, job_id, timeout=30):
        t = time.time()
        while 1:
            if time.time() - t > timeout:
                return False
            time.sleep(1)
            if self._is_job_finished(job_id):
                return True

    def _is_job_finished(self, job_id):
        status = subprocess.check_output(["squeue", "-j", job_id]).decode("utf8")

        if self.verbose_wait:
            print(f"job status is {status}")

        return status == "slurm_load_jobs error: Invalid job id specified" or len(status.split("\n")) == 2
