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
from copy import deepcopy
from enum import Enum

import attrs

from videoworld2.conditioner import BooleanFlag, ReMapkey, TextAttr
from videoworld2.configs.base.defaults.ema import EMAConfig
from videoworld2.configs.vid2vid.defaults.conditioner import Vid2VidConditioner
from videoworld2.models.text2image_dit import SACConfig
from videoworld2.models.video2world_dit import MinimalV1LVGDiT
from videoworld2.models.ctrl_video2world_dit import CtrlEncoder
from videoworld2.tokenizers.tokenizer import TokenizerInterface

from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict




"""
Video2WorldPipelineConfig
├── conditioner (Vid2VidConditioner)
│   ├── fps (ReMapkey)
│   ├── padding_mask (ReMapkey)
│   ├── text (TextAttr)
│   └── use_video_condition (BooleanFlag)
├── net (MinimalV1LVGDiT)
│   └── sac_config (SACConfig)
├── tokenizer (TokenizerInterface)
├── prompt_refiner_config (CosmosReason1Config)
├── guardrail_config (CosmosGuardrailConfig)
├── ema (EMAConfig)
└── timestamps (SolverTimestampConfig)
"""

class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames
    CHANNEL_CONCAT = "channel_concat"  # First few frames of the video are concatenated in the channel dimension

    def __str__(self) -> str:
        return self.value


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
class CosmosReason1Config:
    checkpoint_dir: str
    offload_model_to_cpu: bool = True
    enabled: bool = True


@make_freezable
@attrs.define(slots=False)
class CosmosGuardrailConfig:
    checkpoint_dir: str
    offload_model_to_cpu: bool = True
    enabled: bool = True


@make_freezable
@attrs.define(slots=False)
class Video2WorldPipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict
    conditioning_strategy: str
    min_num_conditional_frames: int
    max_num_conditional_frames: int
    sigma_conditional: float
    net: LazyDict
    tokenizer: LazyDict
    prompt_refiner_config: CosmosReason1Config
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder_class: str = "T5"
    input_data_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)

 
@make_freezable
@attrs.define(slots=False)
class Video2WorldWithCtrlPipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict
    conditioning_strategy: str
    min_num_conditional_frames: int
    max_num_conditional_frames: int
    sigma_conditional: float
    net: LazyDict
    net_ctrl: LazyDict
    tokenizer: LazyDict
    prompt_refiner_config: CosmosReason1Config
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder_class: str = "T5"
    input_data_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)
    hint_key: str = "control_input_depth"
    ldm_input_key: str = "video"
    


# Cosmos Predict2 Ctrl Branch 
PREDICT2_VIDEO2WORLD_CTRL_NET_2B = L(CtrlEncoder)(
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
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
    hint_channels=128,
    layer_mask=[True if (i >= 3) else False for i in range(28)],
    control_weight=1,
    
)

# Cosmos Predict2 Video2World 2B
PREDICT2_VIDEO2WORLD_NET_2B = L(MinimalV1LVGDiT)(
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
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
)


PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B = Video2WorldWithCtrlPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(Vid2VidConditioner)(
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
            dropout_rate=0.0,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=2,
    net=PREDICT2_VIDEO2WORLD_NET_2B,
    net_ctrl=PREDICT2_VIDEO2WORLD_CTRL_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="checkpoints/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=True,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
    hint_key='control_input_depth'
)

PREDICT2_VIDEO2WORLD_PIPELINE_2B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(Vid2VidConditioner)(
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
            dropout_rate=0.0,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=2,
    net=PREDICT2_VIDEO2WORLD_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="checkpoints/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=True,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Cosmos Predict2 Video2World 14B
PREDICT2_VIDEO2WORLD_NET_14B = L(MinimalV1LVGDiT)(
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
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=2.0,
    rope_w_extrapolation_ratio=2.0,
    rope_t_extrapolation_ratio=0.8333333333333334,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_14b_720",
    ),
)

PREDICT2_VIDEO2WORLD_PIPELINE_14B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(Vid2VidConditioner)(
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
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=2,
    net=PREDICT2_VIDEO2WORLD_NET_14B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/Cosmos-Predict2-14B-Video2World/tokenizer/tokenizer.pth",
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="checkpoints/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=True,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Cosmos Predict2 Video2World 2B pipeline config variants - resolution ["480", "720"] and fps [10, 16]
# 2B, resolution 480p, fps 10
PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_2B)
PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS.resolution = "480"
PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS.state_t = 16
# 2B, resolution 480p, fps 16
PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_2B)
PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS.resolution = "480"

PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B_480P_16FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B)
PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B_480P_16FPS.resolution = "480"

PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B_480P_16FPS_INFER = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B)
PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B_480P_16FPS_INFER.resolution = "480"
PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B_480P_16FPS_INFER.tokenizer.vae_pth = "checkpoints/VideoWorld2_dLDM_2B/tokenizer/tokenizer.pth"

# 2B, resolution 720p, fps 10
PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_10FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_2B)
PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_10FPS.state_t = 16
# 2B, resolution 720p, fps 16
PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_16FPS = PREDICT2_VIDEO2WORLD_PIPELINE_2B

# Cosmos Predict2 Video2World 14B pipeline config variants - resolution ["480", "720"] and fps [10, 16}]
# 14B, resolution 480p, fps 10
PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_10FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_14B)
PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_10FPS.resolution = "480"
PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_10FPS.state_t = 16
# 14B, resolution 480p, fps 16
PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_16FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_14B)
PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_16FPS.resolution = "480"
# 14B, resolution 720p, fps 10
PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_10FPS = deepcopy(PREDICT2_VIDEO2WORLD_PIPELINE_14B)
PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_10FPS.state_t = 16
# 14B, resolution 720p, fps 16
PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_16FPS = PREDICT2_VIDEO2WORLD_PIPELINE_14B
