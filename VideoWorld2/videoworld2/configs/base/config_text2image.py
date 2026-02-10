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
import attrs

from videoworld2.conditioner import ReMapkey, TextAttr, VideoConditioner
from videoworld2.models.text2image_dit import MiniTrainDIT
from videoworld2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 35
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class CosmosGuardrailConfig:
    checkpoint_dir: str
    offload_model_to_cpu: bool = True
    enabled: bool = True


@make_freezable
@attrs.define(slots=False)
class Text2ImagePipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict
    net: LazyDict
    tokenizer: LazyDict
    precision: str
    rectified_flow_t_scaling_factor: float
    resize_online: bool
    resolution: str
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder_class: str = "T5"
    guardrail_config: CosmosGuardrailConfig = attrs.field(factory=CosmosGuardrailConfig)


# Cosmos Predict2 Text2Image 2B
PREDICT2_TEXT2IMAGE_NET_2B = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    # cross attention settings
    crossattn_emb_channels=1024,
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

PREDICT2_TEXT2IMAGE_PIPELINE_2B = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(VideoConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=PREDICT2_TEXT2IMAGE_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="1024",
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth",
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Cosmos Predict2 Text2Image 14B
PREDICT2_TEXT2IMAGE_NET_14B = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=5120,
    num_blocks=36,
    num_heads=40,
    mlp_ratio=4.0,
    # cross attention settings
    crossattn_emb_channels=1024,
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

PREDICT2_TEXT2IMAGE_PIPELINE_14B = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(VideoConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=PREDICT2_TEXT2IMAGE_NET_14B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="1024",
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/tokenizer/tokenizer.pth",
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
)
