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
import os

# import pathlib
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

# from filelock import FileLock
from torch import nn

from imaginaire.utils import distributed, log
from imaginaire.utils.easy_io import easy_io


def disabled_train(self: Any, mode: bool = True) -> Any:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def count_params(model: nn.Module, verbose=False) -> int:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def expand_dims_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x
