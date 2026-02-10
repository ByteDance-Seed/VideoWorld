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
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig
from torch import nn

from videoworld2.utils.fused_adam_dtensor import FusedAdam
from videoworld2.utils.optim_instantiate_dtensor import get_base_optimizer
from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.utils import log


def get_regular_param_group_simple(net: nn.Module):
    """
    seperate the parameters of the network into two groups: decay and no_decay.
    based on nano_gpt codebase.
    """
    param_dict = {pn: p for pn, p in net.named_parameters()}
    num_total_params = sum(p.numel() for p in param_dict.values())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    num_trainable_params = sum(p.numel() for p in param_dict.values())
    log.info(f"total num parameters : {num_total_params:,} | trainable num parameters : {num_trainable_params:,}")

    params = [p for n, p in param_dict.items()]
    return params


def get_base_optimizer_simple(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    net_param_total = get_regular_param_group_simple(model)

    param_group = [
        {
            "params": net_param_total,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    elif optim_type == "fusedadam":
        opt_cls = FusedAdam
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    for k, v in kwargs.items():
        if isinstance(v, ListConfig):
            kwargs[k] = list(v)

    return opt_cls(param_group, **kwargs)


FusedAdamWConfig: LazyDict = L(get_base_optimizer)(
    model=PLACEHOLDER,
    lr=1e-4,
    weight_decay=0.1,
    betas=[0.9, 0.99],
    optim_type="fusedadam",
    eps=1e-8,
    master_weights=True,
    capturable=True,
)


def register_optimizer():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)
