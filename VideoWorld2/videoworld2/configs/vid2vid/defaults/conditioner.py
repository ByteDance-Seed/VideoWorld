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
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from videoworld2.conditioner import BooleanFlag, GeneralConditioner, ReMapkey, T2VCondition, TextAttr
from videoworld2.utils.context_parallel import broadcast_split_tensor
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@dataclass(frozen=True)
class Vid2VidCondition(T2VCondition):
    use_video_condition: bool = False
    # the following two attributes are used to set the video condition; during training, inference
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None
    control_input_depth: Optional[torch.Tensor] = None
    recon_image: Optional[torch.Tensor] = None
    decoder_feature: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        **extra_kwargs,
    ) -> "Vid2VidCondition":
        """
        Sets the video conditioning frames for video-to-video generation.

        This method creates a conditioning mask for the input video frames that determines
        which frames will be used as context frames for generating new frames. The method
        handles both image batches (T=1) and video batches (T>1) differently.

        Args:
            gt_frames: A tensor of ground truth frames with shape [B, C, T, H, W], where:
                B = batch size
                C = number of channels
                T = number of frames
                H = height
                W = width

            random_min_num_conditional_frames: Minimum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            random_max_num_conditional_frames: Maximum number of frames to use for conditioning
                when randomly selecting a number of conditioning frames.

            num_conditional_frames: Optional; If provided, all examples in the batch will use
                exactly this many frames for conditioning. If None, a random number of frames
                between random_min_num_conditional_frames and random_max_num_conditional_frames
                will be selected for each example in the batch.

        Returns:
            A new Vid2VidCondition object with the gt_frames and conditioning mask set.
            The conditioning mask (condition_video_input_mask_B_C_T_H_W) is a binary tensor
            of shape [B, 1, T, H, W] where 1 indicates frames used for conditioning and 0
            indicates frames to be generated.

        Notes:
            - For image batches (T=1), no conditioning frames are used (num_conditional_frames_B = 0).
            - For video batches:
                - If num_conditional_frames is provided, all examples use that fixed number of frames.
                - Otherwise, each example randomly uses between random_min_num_conditional_frames and
                random_max_num_conditional_frames frames.
            - The mask marks the first N frames as conditioning frames (set to 1) for each example.
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames
        for key in extra_kwargs:
            kwargs[key] = extra_kwargs[key]

        # condition_video_input_mask_B_C_T_H_W
        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:  # handle image batch
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:  # handle video batch
            if num_conditional_frames is not None:
                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames, random_max_num_conditional_frames + 1, size=(B,)
                )
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[idx, :, : num_conditional_frames_B[idx], :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        # import pdb;pdb.set_trace()
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Vid2VidCondition":
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:
            # Do not use classifier free guidance on conditional frames.
            # YB found that it leads to worse results.
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Vid2VidCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        control_input_depth = self.control_input_depth
        recon_image = self.recon_image
        decoder_feature = self.decoder_feature
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["recon_image"] = None
        kwargs["control_input_depth"] = None
        kwargs["decoder_feature"] = None
        new_condition = T2VCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                if control_input_depth is not None:
                    control_input_depth = broadcast_split_tensor(
                        control_input_depth, seq_dim=2, process_group=process_group
                    )
                if recon_image is not None:
                    recon_image = broadcast_split_tensor(
                        recon_image, seq_dim=2, process_group=process_group
                    )
                if decoder_feature is not None:
                    decoder_feature = broadcast_split_tensor(
                        decoder_feature, seq_dim=2, process_group=process_group
                    )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["control_input_depth"] = control_input_depth
        kwargs["recon_image"] = recon_image
        kwargs["decoder_feature"] = decoder_feature
        return type(self)(**kwargs)


class Vid2VidConditionV2(Vid2VidCondition):
    """
    compared to Vid2VidCondition, this class apply zero frames when use_video_condition is False~(unconditional generation in cfg)
    in the case, we do zero-out conditional frames in the video condition
    """

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
    ) -> "Vid2VidConditionV2":
        num_conditional_frames = 0 if not self.use_video_condition else num_conditional_frames
        return super().set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Vid2VidConditionV2":
        del is_cfg_conditional
        _condition = super().set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        return _condition


class Vid2VidConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Vid2VidCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Vid2VidCondition(**output)


class Vid2VidConditionerV2(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Vid2VidConditionV2:
        output = super()._forward(batch, override_dropout_rate)
        return Vid2VidConditionV2(**output)


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

VideoPredictionConditioner: LazyDict = L(Vid2VidConditioner)(
    **_SHARED_CONFIG,
)

VideoPredictionConditionerV2: LazyDict = L(Vid2VidConditionerV2)(
    **_SHARED_CONFIG,
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )
