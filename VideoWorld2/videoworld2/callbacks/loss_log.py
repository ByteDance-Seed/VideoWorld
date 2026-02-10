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
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


@dataclass
class _LossRecord:
    iter_count: int = 0
    loss: float = 0

    def reset(self) -> None:
        self.iter_count = 0
        self.loss = 0

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            loss = self.loss / self.iter_count
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        else:
            loss = torch.ones(1)
        iter_count = self.iter_count
        self.reset()
        return loss.tolist(), iter_count


class LossLog(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
    ) -> None:
        super().__init__()
        self.logging_iter_multipler = logging_iter_multipler
        self.name = self.__class__.__name__

        self.train_video_log = _LossRecord()

    def on_before_backward(
        self,
        model: ImaginaireModel,
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        # Log this loss for aligning the curve with diffsyncstudio
        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0 and distributed.is_rank0():
            info = {
                "train_loss_step": loss.detach().item(),
            }

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ):
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            _loss = output_batch["loss"].detach().mean(dim=0)

            self.train_video_log.iter_count += 1
            self.train_video_log.loss += _loss

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            world_size = dist.get_world_size()
            loss, iter_count = self.train_video_log.get_stat()
            iter_count *= world_size

            if distributed.is_rank0():
                info = {}
                if iter_count > 0:
                    info[f"train@{self.logging_iter_multipler}/loss"] = loss
