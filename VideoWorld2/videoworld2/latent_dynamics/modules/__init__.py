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
from enum import Enum

from videoworld2.latent_dynamics.modules.distributions import GaussianDistribution, IdentityDistribution
# from videoworld2.latent_dynamics.modules.layers2d import Decoder, Encoder
from videoworld2.latent_dynamics.modules.layers3d import DecoderBase, DecoderFactorized, EncoderBase, EncoderFactorized, LDMDecoderFactorized, LDMEncoderFactorizedV2,  LDMEncoderFactorized, LDMDecoderFactorizedV2, LDMDecoderFactorizedV3
from videoworld2.latent_dynamics.modules.quantizers import FSQuantizer, LFQuantizer, ResidualFSQuantizer, VectorQuantizer

# from videoworld2.latent_dynamics.modules.qformer_ldm import (
#     QFormerMF,
#     QFormerMFSep,
#     QFormerMFSingleQ,
#     # QFormerAdjacentFSingleQ,
#     PositionEmbeddingRandom
# )

# class EncoderType(Enum):
#     Default = Encoder


# class DecoderType(Enum):
#     Default = Decoder


class Encoder3DType(Enum):
    BASE = EncoderBase
    FACTORIZED = EncoderFactorized
    LDM = LDMEncoderFactorized
    LDM_V2 = LDMEncoderFactorizedV2


class Decoder3DType(Enum):
    BASE = DecoderBase
    FACTORIZED = DecoderFactorized
    LDM = LDMDecoderFactorized
    LDM_V2 = LDMDecoderFactorizedV2
    LDM_V3 = LDMDecoderFactorizedV3
    


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


class DiscreteQuantizer(Enum):
    VQ = VectorQuantizer
    LFQ = LFQuantizer
    FSQ = FSQuantizer
    RESFSQ = ResidualFSQuantizer
