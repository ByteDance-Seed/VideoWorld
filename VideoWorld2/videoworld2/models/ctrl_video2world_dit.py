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
from einops import rearrange
import torch
from torch import nn
from videoworld2.conditioner import DataType
from videoworld2.models.video2world_dit import MinimalV1LVGDiT
from videoworld2.models.text2image_dit import PatchEmbed
from torchvision import transforms
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class CtrlEncoder(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        hint_channels = kwargs.pop("hint_channels", 16)
        control_weight = kwargs.pop("control_weight", 16)
        num_control_blocks = kwargs.pop("num_control_blocks", None)
        if num_control_blocks is not None:
            assert num_control_blocks > 0 and num_control_blocks <= kwargs["num_blocks"]
            kwargs["layer_mask"] = [False] * num_control_blocks + [True] * (kwargs["num_blocks"] - num_control_blocks)
        super().__init__(*args, **kwargs)
        num_blocks = self.num_blocks
        model_channels = self.model_channels
        layer_mask = kwargs.get("layer_mask", None)
        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        self.layer_mask = layer_mask
        self.hint_channels = hint_channels
        self.control_weight = control_weight
        self.build_hint_patch_embed()
        hint_nf = [16, 16, 32, 32, 96, 96, 256]
        nonlinearity = nn.SiLU()
        input_hint_block = [nn.Linear(model_channels, hint_nf[0]), nonlinearity]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
        self.input_hint_block = nn.Sequential(*input_hint_block)
        # Initialize weights
        self.init_weights()
        self.zero_blocks = nn.ModuleDict()
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            self.zero_blocks[f"block{idx}"] = zero_module(nn.Linear(model_channels, model_channels))
        self.input_hint_block.append(zero_module(nn.Linear(hint_nf[-1], model_channels)))

    def build_hint_patch_embed(self):
        # import pdb;pdb.set_trace()
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.hint_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        patch_spatial = patch_spatial if in_channels == 128 else 1
        in_channels = in_channels + 1 if (concat_padding_mask and in_channels == 128) else in_channels
        self.x_embedder2 = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def prepare_hint_embedded_sequence(
        self, x_B_C_T_H_W: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.concat_padding_mask and self.hint_channels==128:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder2(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None

    def encode_hint(
        self,
        hint: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
    ) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        # assert hint.size(1) <= self.hint_channels, f"Expected hint channels <= {self.hint_channels}, got {hint.size(1)}"
        if hint.size(1) < self.hint_channels:
            padding_shape = list(hint.shape)
            padding_shape[1] = self.hint_channels - hint.size(1)
            hint = torch.cat([hint, torch.zeros(*padding_shape, dtype=hint.dtype, device=hint.device)], dim=1)
        
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."

        hint_B_T_H_W_D, _ = self.prepare_hint_embedded_sequence(hint, fps=fps, padding_mask=padding_mask)

        hint = rearrange(hint_B_T_H_W_D, "B T H W D -> T H W B D")
  
        guided_hint = self.input_hint_block(hint)
        return guided_hint
    
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
        # For Ctrl
        hint_key: Optional[str] = None,
        # control_weight: Optional[float] = 1.0,
        num_layers_to_use: Optional[int] = -1,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        is_training_base_model: bool = False,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        # import pdb;pdb.set_trace()
        x_input = x_B_C_T_H_W
        hint = kwargs.pop(hint_key) #[1, 16, T, 60, 104]
        assert hint is not None
       
        guided_hints = self.encode_hint(hint, fps=fps, padding_mask=padding_mask, data_type=data_type) #T, 30, 56, 1, 2048
        guided_hints = torch.chunk(guided_hints, hint.shape[0] // x_input.shape[0], dim=3)
        # Only support multi-control at inference time
        assert len(guided_hints) == 1 or not torch.is_grad_enabled()
        

        B, C, T, H, W = x_input.shape
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        self.crossattn_emb = crossattn_emb
        outs = {}

        num_control_blocks = self.layer_mask.index(True)
        num_layers_to_use = num_control_blocks
        control_gate_per_layer = [i < num_layers_to_use for i in range(num_control_blocks)]

        is_training = torch.is_grad_enabled()
        if is_training and is_training_base_model:
            coin_flip = torch.rand(B).to(x_B_C_T_H_W.device) > self.dropout_ctrl_branch  # prob for only training base model
            if self.blocks["block0"].x_format == "THWBD":
                coin_flip = coin_flip[None, None, None, :, None]
            elif self.blocks["block0"].x_format == "BTHWD":
                coin_flip = coin_flip[:, None, None, None, None]
        else:
            coin_flip = 1

        # if isinstance(control_weight, torch.Tensor):
        #     if control_weight.ndim == 0:  # Single scalar tensor
        #         control_weight = [float(control_weight)] * len(guided_hints)
        #     elif control_weight.ndim == 1:  # List of scalar weights
        #         control_weight = [float(w) for w in control_weight]
        #     else:  # Spatial-temporal weight maps
        #         control_weight = [w for w in control_weight]  # Keep as tensor
        # else:
        control_weight = self.control_weight
        control_weight = [control_weight] * len(guided_hints)

        x_before_blocks = x_B_C_T_H_W.clone()
        for i, guided_hint in enumerate(guided_hints):
            x_B_C_T_H_W = x_before_blocks
            blocks = self.blocks
            zero_blocks = self.zero_blocks
            t_embedder = self.t_embedder
    

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
                    x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
                )   

            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
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

            block_kwargs = {
                "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
                "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            }
            for idx, block in enumerate(blocks):
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    **block_kwargs,
                )
                if guided_hint is not None:
                    x_B_T_H_W_D = x_B_T_H_W_D + guided_hint.permute(3, 0, 1, 2, 4)
                    guided_hint = None

                name = f"block{idx}"
                gate = control_gate_per_layer[idx]
                if isinstance(control_weight[i], (float, int)):
                    hint_val = zero_blocks[name](x_B_T_H_W_D) * control_weight[i] * coin_flip * gate
                else:  # Spatial-temporal weights [num_controls, B, 1, T, H, W]
                    control_feat = zero_blocks[name](x)
                    # Get current feature dimensions
                    weight_map = control_weight[i]  # [B, 1, T, H, W]
                    # Reshape to match THWBD format
                    weight_map = weight_map.permute(2, 3, 4, 0, 1)  # [T, H, W, B, 1]
                    weight_map = weight_map.view(T * H * W, 1, 1, B, 1)

                    hint_val = control_feat * weight_map * coin_flip * gate
                if name not in outs:
                    outs[name] = hint_val
                else:
                    outs[name] += hint_val


        return outs