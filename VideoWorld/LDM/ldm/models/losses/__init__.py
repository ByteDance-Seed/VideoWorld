# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .adv_loss import AdvLoss
from .clip_loss import CLIPLoss
from .composition_loss import (CharbonnierCompLoss, L1CompositionLoss,
                               MSECompositionLoss)
from .gan_loss import (DiscShiftLoss, GANLoss, GaussianBlur,
                       GradientPenaltyLoss, disc_shift_loss,
                       gen_path_regularizer, gradient_penalty_loss,
                       r1_gradient_penalty_loss)
from .gradient_loss import GradientLoss
from .loss_comps import (CLIPLossComps, DiscShiftLossComps,
                         GANLossComps, GeneratorPathRegularizerComps,
                         GradientPenaltyLossComps, R1GradientPenaltyComps)
from .loss_wrapper import mask_reduce_loss, reduce_loss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import (CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss,
                             PSNRLoss, tv_loss)
from .vqperceptual import LPIPS

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'L1CompositionLoss',
    'MSECompositionLoss', 'CharbonnierCompLoss', 'GANLoss', 'GaussianBlur',
    'GradientPenaltyLoss', 'PerceptualLoss', 'PerceptualVGG', 'reduce_loss',
    'mask_reduce_loss', 'DiscShiftLoss', 'MaskedTVLoss', 'GradientLoss',
    'TransferalPerceptualLoss', 'gradient_penalty_loss',
    'r1_gradient_penalty_loss', 'gen_path_regularizer',
    'CLIPLoss', 'CLIPLossComps', 'DiscShiftLossComps', 
    'GANLossComps', 'GeneratorPathRegularizerComps',
    'GradientPenaltyLossComps', 'R1GradientPenaltyComps', 'disc_shift_loss',
    'tv_loss', 'PSNRLoss', 'AdvLoss','LPIPS'
]
