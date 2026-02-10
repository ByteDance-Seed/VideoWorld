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
from __future__ import annotations

import collections
import os
import threading
from typing import TYPE_CHECKING

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from imaginaire.model import ImaginaireModel
from imaginaire.utils import callback, distributed, log, misc

if TYPE_CHECKING:
    from imaginaire.config import CheckpointConfig, JobConfig


class Checkpointer:
    """The checkpointer class. Supports checkpoint saving/loading to local disk."""

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        # Set the callback functions.
        self.callbacks = callbacks
        self.checkpoint_dir_local = f"{config_job.path_local}/checkpoints"
        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"

        # Handle optimizer state dict if FSDP is enabled
        is_fsdp = model.config.fsdp_shard_size != 0 and distributed.get_world_size() > 1
        is_fsdp = True
        if is_fsdp:
            optimizer_state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        else:
            optimizer_state_dict = optimizer.state_dict()

        # Gather all the state dicts to be saved
        state_dicts_to_save = {
            "model": model.state_dict(),
            "optim": optimizer_state_dict,
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }

        if distributed.get_rank() == 0:
            self.callbacks.on_save_checkpoint(model, state_dict=state_dicts_to_save)
            folders = state_dicts_to_save.keys()
            for folder in folders:
                state_dict = state_dicts_to_save[folder]
                state_dict = misc.to(state_dict, device="cpu")
                # Wait for previous saver thread to end.
                if self.save_thread:
                    self.save_thread.join()
                # Run the checkpoint saver in a separate thread.
                checkpoint_path = os.path.join(self.checkpoint_dir_local, folder, checkpoint_file)
                self.save_thread = threading.Thread(
                    target=self._save_worker_local,
                    daemon=False,
                    args=(state_dict, checkpoint_path, distributed.get_rank()),
                )
                self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(self, state_dict: dict[str, torch.Tensor], checkpoint_path: str, rank: int = 0) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_path (str): The path of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_file = os.path.basename(checkpoint_path)
        try:
            torch.save(state_dict, checkpoint_path)
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to save (local): {e}")

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        assert self.load_path is None, "load_path is not supported yet"
        self.callbacks.on_load_checkpoint_start(model)

        is_fsdp = model.config.fsdp_shard_size != 0 and distributed.get_world_size() > 1

        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = self.checkpoint_dir_local
            model_checkpoint_path = os.path.join(checkpoint_dir, "model", latest_checkpoint_file)
            optimizer_checkpoint_path = os.path.join(checkpoint_dir, "optim", latest_checkpoint_file)
            scheduler_checkpoint_path = os.path.join(checkpoint_dir, "scheduler", latest_checkpoint_file)
            trainer_checkpoint_path = os.path.join(checkpoint_dir, "trainer", latest_checkpoint_file)
            resume = True
            only_resume_scheduler = True
        else:
            model_checkpoint_path = None
            optimizer_checkpoint_path = None
            scheduler_checkpoint_path = None
            trainer_checkpoint_path = None
            resume = False
            only_resume_scheduler = False
        # import pdb;pdb.set_trace()
        # Load checkpoint.
        if latest_checkpoint_file is not None:
            state_dicts_paths = {
                "model": model_checkpoint_path,
                "optim": optimizer_checkpoint_path,
                "scheduler": scheduler_checkpoint_path,
                "trainer": trainer_checkpoint_path,
            }
            state_dicts_to_load = {}
            for key, checkpoint_path in state_dicts_paths.items():
                try:
                    self._check_checkpoint_exists(checkpoint_path)
                    log.info(f"Loading checkpoint (local): {checkpoint_path}")
                    state_dicts_to_load[key] = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                    log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
                except:
                    log.info(f"Loading checkpoint (local): {checkpoint_path}, Failed")

            # import pdb;pdb.set_trace()
            self.callbacks.on_load_checkpoint(model, state_dict=state_dicts_to_load)

            # Load the state dicts.
            log.info("- Loading the model...")
            is_fsdp = True
            if is_fsdp:
                # If a model is wrapped with FSDP, its underlying weights will be DTensor.
                # However, Transformer Engine cannot load weights (the useless `extra_state`) into DTensor.
                # So we need to first remove the attention operators from Transformer Engine.
                # It will work correctly as long as the attention operators do not have any weights.
                for block in model.pipe.dit.blocks:
                    block.self_attn.attn = None
                
                state_dicts_to_load_for_dit_ctrl = collections.OrderedDict()
                state_dicts_to_load_for_dit_reg = collections.OrderedDict()
                state_dicts_to_load_for_dit_ema = collections.OrderedDict()
                state_dicts_to_load_for_dit_ldm = collections.OrderedDict()
                for key, val in state_dicts_to_load["model"].items():
                    if key.startswith("net."):
                        state_dicts_to_load_for_dit_reg[key.replace("net.", "")] = val
                    elif key.startswith("net_ema."):
                        state_dicts_to_load_for_dit_ema[key.replace("net_ema.", "")] = val
                    elif key.startswith("ldm_net."):
                        state_dicts_to_load_for_dit_ldm[key.replace("ldm_net.", "")] = val
                    elif key.startswith("net_ctrl"):
                        state_dicts_to_load_for_dit_ctrl[key.replace("net_ctrl.", "")] = val
                # Load Regular weights.
               
                set_model_state_dict(
                    model.pipe.dit,
                    state_dicts_to_load_for_dit_reg,
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                        strict=False if model.config.train_architecture == "lora" else True,
                    ),
                )
                # Load Regular weights.
                try:
                    set_model_state_dict(
                        model.ldm_net,
                        state_dicts_to_load_for_dit_ldm,
                        options=StateDictOptions(
                            full_state_dict=True,
                            broadcast_from_rank0=True,
                            strict=False,
                        ),
                    )
                except:
                    pass

                try:
                    set_model_state_dict(
                        model.pipe.dit_ctrl,
                        state_dicts_to_load_for_dit_ctrl,
                        options=StateDictOptions(
                            full_state_dict=True,
                            broadcast_from_rank0=True,
                            strict=False,
                        ),
                    )
                except:
                    pass


                # Load EMA weights.
                if model.pipe.config.ema.enabled:
                    set_model_state_dict(
                        model.pipe.dit_ema,
                        state_dicts_to_load_for_dit_ema,
                        options=StateDictOptions(
                            full_state_dict=True,
                            broadcast_from_rank0=True,
                            strict=False if model.config.train_architecture == "lora" else True,
                        ),
                    )

                # Restore the attention operators.
                model.pipe.apply_cp()
            else:
                model.load_state_dict(state_dicts_to_load["model"], strict=self.strict_resume)
            
            try:
                if resume or only_resume_scheduler:
                    iteration = state_dicts_to_load["trainer"]["iteration"]
                    assert scheduler
                    log.info("- Loading the scheduler...")
                    scheduler.load_state_dict(state_dicts_to_load["scheduler"])
                    scheduler.last_epoch = iteration
                else:
                    iteration = 0
            except:
                iteration = 0
            
            try:
                if resume:
                    assert optimizer
                    log.info("- Loading the optimizer...")
                    if is_fsdp:
                        set_optimizer_state_dict(
                            model,
                            optimizer,
                            state_dicts_to_load["optim"],
                            options=StateDictOptions(
                                full_state_dict=True,
                                broadcast_from_rank0=True,
                            ),
                        )
                    else:
                        optimizer.load_state_dict(state_dicts_to_load["optim"])
                    log.info("- Loading the gradient scaler...")
                    grad_scaler.load_state_dict(state_dicts_to_load["trainer"]["grad_scaler"])
                    log.success(f"Done with loading the checkpoint (iteration {iteration}).")
                else:
                    log.success("Done with loading the checkpoint.")
            except:
                pass
        else:
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=model_checkpoint_path)

        return iteration

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        if os.path.isfile(latest_path):
            checkpoint_file = open(latest_path).read().strip()
        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File not found (local): {checkpoint_path}")

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()
