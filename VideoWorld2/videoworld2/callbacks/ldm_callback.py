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
import time

import torch
from torch import Tensor

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import log
from imaginaire.utils.distributed import rank0_only
from imaginaire.utils.easy_io import easy_io
import torch.distributed as dist

class LDMCallback(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # import pdb;pdb.set_trace()
        world_size = dist.get_world_size()
        _gather_data = [None for _ in range(world_size)]

        dist.all_gather_object(_gather_data, model.validation_results)
        if dist.get_rank() == 0:
            gather_data = []
            for data in _gather_data:
                gather_data.extend(data)
            

            latent_code_infos = []
            for item in gather_data:
               
                if 'latent_codes' in item:
                    for latent_code, video_path in zip(item['latent_codes'], item['video_path']):
                        latent_code_infos.append((latent_code.cpu().tolist(), video_path))



            if len(latent_code_infos) > 0:
                torch.save(latent_code_infos, f'latent_code_infos.pt')
            log.info(f"Latent Codes Saved")
            
        dist.barrier()
        model.validation_results = []
        return
