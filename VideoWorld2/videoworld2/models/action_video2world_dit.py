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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from videoworld2.conditioner import DataType
from videoworld2.models.video2world_dit import MinimalV1LVGDiT


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionConditionedMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs["action_dim"]
        del kwargs["action_dim"]
        # import pdb;pdb.set_trace()
        super().__init__(*args, **kwargs)

        self.action_embedder_B_D = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        action: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        # import pdb;pdb.set_trace()
        # NOTE: project action to action embedding
        assert action is not None, "action must be provided"
        # action = rearrange(action, "b t d -> b 1 (t d)") # For LDM
        action_emb_B_D = self.action_embedder_B_D(action)
        action_emb_B_3D = self.action_embedder_B_3D(action)

        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        # import pdb;pdb.set_trace()
        # NOTE: add action embedding to the timestep embedding and adaln_lora
        if t_embedding_B_T_D.shape[1] != 1: # For LDM
            rank = torch.distributed.get_rank()
            context_parallel_size = 2
            chunk_size = t_embedding_B_T_D.shape[1]
            # world_size = torch.distributed.get_world_size()
            chunk_idx = rank % context_parallel_size
            if chunk_idx == 0:
                t_embedding_B_T_D[:, 1:] = t_embedding_B_T_D[:, 1:] + action_emb_B_D[:, :chunk_size-1]
                adaln_lora_B_T_3D[:, 1:] = adaln_lora_B_T_3D[:, 1:] + action_emb_B_3D[:, :chunk_size-1]
            else:
                t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D[:, chunk_idx*chunk_size-1:(chunk_idx+1)*chunk_size-1]
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D[:, chunk_idx*chunk_size-1:(chunk_idx+1)*chunk_size-1]
        else:
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D

        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
