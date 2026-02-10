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
__all__ = [
    'FSQQuantizer',
    'FSQDQuantizer'
]

from abc import ABC
from typing import TYPE_CHECKING, MutableMapping, cast, Optional
from mmengine.model import BaseModule
import transformers
transformers.optimization.AdamW.__name__ = 'transformers_optimization_AdamW'
# import pdb;pdb.set_trace()
# import todd
import torch
from torch import Tensor, int32
import torch.distributed
# from todd.runners import Memo
from torch import nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack

from falcon.registry import MODELS



def exists(v):
    return v is not None
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

@MODELS.register_module()
class FSQQuantizer(BaseModule):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()
        # import pdb;pdb.set_trace()
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes.to(z.dtype))

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out, indices


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSQDQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        init_scale=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.init_scale=init_scale
        self.ada_param = nn.Parameter(torch.ones(codebook_dim)*self.init_scale,requires_grad=True)

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        scale = F.sigmoid(self.ada_param)*4
        return torch.tanh(scale*(z + shift)) * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSQLQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        init_scale=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.init_scale=init_scale
        self.ada_param = nn.Parameter(torch.ones(codebook_dim)*self.init_scale,requires_grad=True)

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return torch.tanh(self.ada_param*(z + shift)) * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        init_scale=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.init_scale=init_scale
        self.ada_param = nn.Parameter(torch.ones(codebook_dim)*self.init_scale,requires_grad=True)

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return torch.tanh(self.ada_param*(z + shift)) * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(z, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSSQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.bound(z)
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSQLSQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)



    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward



class TanhStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a,h,k,low,epsilon=0.001):
        ctx.save_for_backward(input, a, h, )
        ctx.epsilon =  epsilon
        ctx.k = k
        ctx.low = low
        # a = a  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 前向传播，计算逼近阶梯函数的值
        S = torch.sum(h * 0.5 * (torch.tanh(k * (input.unsqueeze(dim=-1) - a)) + 1), dim=-1) - (torch.sum(h,dim=-1) / 2)
        return S

    @staticmethod
    def backward(ctx, grad_output):
        input, a, h = ctx.saved_tensors
        epsilon=ctx.epsilon
        k=ctx.k
        low = ctx.low
        # a = a.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 后向传播，计算带epsilon的导数
        dS_dinput = torch.sum(h * 0.5 * k * (1 - torch.tanh(k * (input.unsqueeze(-1) - a)) ** 2), dim=-1) + epsilon
        # 将梯度小于1的值设置为1
        dS_dinput_clamped = torch.clamp(dS_dinput, min=low)
        grad_input = grad_output * dS_dinput_clamped
        return grad_input, None, None, None, None

class TanhStepMinMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a,h,k,low,high,epsilon=0.001):
        ctx.save_for_backward(input, a, h, )
        ctx.epsilon =  epsilon
        ctx.k = k
        ctx.low = low
        ctx.high = high
        # a = a  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 前向传播，计算逼近阶梯函数的值
        S = torch.sum(h * 0.5 * (torch.tanh(k * (input.unsqueeze(dim=-1) - a)) + 1), dim=-1) - (torch.sum(h,dim=-1) / 2)
        return S

    @staticmethod
    def backward(ctx, grad_output):
        input, a, h = ctx.saved_tensors
        epsilon=ctx.epsilon
        k=ctx.k
        low = ctx.low
        high = ctx.high
        # a = a.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 后向传播，计算带epsilon的导数
        dS_dinput = torch.sum(h * 0.5 * k * (1 - torch.tanh(k * (input.unsqueeze(-1) - a)) ** 2), dim=-1) + epsilon
        # 将梯度小于1的值设置为1
        dS_dinput_clamped = torch.clamp(dS_dinput, min=low,max=high)
        grad_input = grad_output * dS_dinput_clamped
        return grad_input, None, None, None, None,None


class TanhStepMerTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a,h,k,low,epsilon=0.001):
        ctx.save_for_backward(input, a, h, )
        ctx.epsilon =  epsilon
        ctx.k = k
        ctx.low = low
        # a = a  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 前向传播，计算逼近阶梯函数的值
        S = torch.sum(h * 0.5 * (torch.tanh(k * (input.unsqueeze(dim=-1) - a)) + 1), dim=-1) - (torch.sum(h,dim=-1) / 2)
        return S

    @staticmethod
    def backward(ctx, grad_output):
        input, a, h = ctx.saved_tensors
        epsilon=ctx.epsilon
        k=ctx.k
        low = ctx.low
        # a = a.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 后向传播，计算带epsilon的导数
        dS_dinput = torch.sum(h * 0.5 * k * (1 - torch.tanh(k * (input.unsqueeze(-1) - a)) ** 2), dim=-1) + epsilon
        tanh_grad = 1 - torch.tanh(input) ** 2
        # 将梯度小于1的值设置为1
        dS_dinput_clamped = torch.max(dS_dinput, tanh_grad)
        grad_input = grad_output * dS_dinput_clamped
        return grad_input, None, None, None, None


class TanhStepMerTanhCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a,h,k,low,epsilon=0.001):
        ctx.save_for_backward(input, a, h, )
        ctx.epsilon =  epsilon
        ctx.k = k
        ctx.low = low
        # a = a  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 前向传播，计算逼近阶梯函数的值
        S = torch.sum(h * 0.5 * (torch.tanh(k * (input.unsqueeze(dim=-1) - a)) + 1), dim=-1) - (torch.sum(h,dim=-1) / 2)
        return S

    @staticmethod
    def backward(ctx, grad_output):
        input, a, h = ctx.saved_tensors
        epsilon=ctx.epsilon
        k=ctx.k
        low = ctx.low
        # a = a.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # h = h.view(1, 1, 1, -1)  # shape: (1, 1, 1, d)
        # 后向传播，计算带epsilon的导数
        dS_dinput = torch.sum(h * 0.5 * k * (1 - torch.tanh(k * (input.unsqueeze(-1) - a)) ** 2), dim=-1) + epsilon
        tanh_grad = 1 - torch.tanh(input) ** 2
        # 将梯度小于1的值设置为1
        dS_dinput_clamped = torch.max(dS_dinput, tanh_grad)
        dS_dinput_clamped = torch.clamp(dS_dinput_clamped, max=3.0)
        grad_input = grad_output * dS_dinput_clamped
        return grad_input, None, None, None, None



@MODELS.register_module()
class FSQLRDQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        k=50,
        low=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        a_8 = torch.tensor([[-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          ])
        a_5 = torch.tensor([[-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            ])
        self.register_buffer("a_8", a_8, persistent=False)
        self.register_buffer("a_5", a_5, persistent=False)

        h_8 = torch.tensor([[1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            ])
        h_5 = torch.tensor([[1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            ])
        self.register_buffer("h_8", h_8, persistent=False)
        self.register_buffer("h_5", h_5, persistent=False)
        self.k=k
        self.low = low

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    # def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    #     """Bound `z`, an array of shape (..., d)."""
    #     half_l = (self._levels - 1) * (1 - eps) / 2
    #     offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
    #     shift = (offset / half_l).atanh()
    #     return (z + shift).tanh() * half_l - offset
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        # return (z + shift).tanh() * half_l - offset
        z_8 = z[:,:,:,:3]
        z_5 = z[:, :, :, 3:]
        out_8 = TanhStepFunction.apply(z_8, self.a_8, self.h_8, self.k,self.low)-0.5
        out_5 = TanhStepFunction.apply(z_5, self.a_5, self.h_5, self.k,self.low)+0.5
        out = torch.cat([out_8,out_5],dim=-1)
        return out


    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward


@MODELS.register_module()
class FSQLRDMinMaxQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        k=50,
        low=1.0,
        high=3.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        a_8 = torch.tensor([[-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          ])
        a_5 = torch.tensor([[-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            ])
        self.register_buffer("a_8", a_8, persistent=False)
        self.register_buffer("a_5", a_5, persistent=False)

        h_8 = torch.tensor([[1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            ])
        h_5 = torch.tensor([[1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            ])
        self.register_buffer("h_8", h_8, persistent=False)
        self.register_buffer("h_5", h_5, persistent=False)
        self.k=k
        self.low = low
        self.high = high

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    # def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    #     """Bound `z`, an array of shape (..., d)."""
    #     half_l = (self._levels - 1) * (1 - eps) / 2
    #     offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
    #     shift = (offset / half_l).atanh()
    #     return (z + shift).tanh() * half_l - offset
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        # return (z + shift).tanh() * half_l - offset
        z_8 = z[:,:,:,:3]
        z_5 = z[:, :, :, 3:]
        out_8 = TanhStepMinMaxFunction.apply(z_8, self.a_8, self.h_8, self.k,self.low,self.high)-0.5
        out_5 = TanhStepMinMaxFunction.apply(z_5, self.a_5, self.h_5, self.k,self.low,self.high)+0.5
        out = torch.cat([out_8,out_5],dim=-1)
        return out


    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward



@MODELS.register_module()
class FSQLRDMTQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        k=50,
        low=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        a_8 = torch.tensor([[-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          ])
        a_5 = torch.tensor([[-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            ])
        self.register_buffer("a_8", a_8, persistent=False)
        self.register_buffer("a_5", a_5, persistent=False)

        h_8 = torch.tensor([[1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            ])
        h_5 = torch.tensor([[1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            ])
        self.register_buffer("h_8", h_8, persistent=False)
        self.register_buffer("h_5", h_5, persistent=False)
        self.k=k
        self.low = low

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    # def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    #     """Bound `z`, an array of shape (..., d)."""
    #     half_l = (self._levels - 1) * (1 - eps) / 2
    #     offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
    #     shift = (offset / half_l).atanh()
    #     return (z + shift).tanh() * half_l - offset
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        # return (z + shift).tanh() * half_l - offset
        z_8 = z[:,:,:,:3]
        z_5 = z[:, :, :, 3:]
        out_8 = TanhStepMerTanhFunction.apply(z_8, self.a_8, self.h_8, self.k,self.low)-0.5
        out_5 = TanhStepMerTanhFunction.apply(z_5, self.a_5, self.h_5, self.k,self.low)+0.5
        out = torch.cat([out_8,out_5],dim=-1)
        return out


    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward

@MODELS.register_module()
class FSQLRDMTCLQuantizer(nn.Module):

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        k=50,
        low=1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _levels = torch.tensor(levels,dtype=int32)

        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks

        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        a_8 = torch.tensor([[-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          [-1.3540, -0.7332, -0.3942, -0.1257,  0.1257,  0.3942,  0.7332],
                          ])
        a_5 = torch.tensor([[-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            [-0.9730, -0.2554,  0.2554,  0.9730],
                            ])
        self.register_buffer("a_8", a_8, persistent=False)
        self.register_buffer("a_5", a_5, persistent=False)

        h_8 = torch.tensor([[1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            [1.,1.,1.,1.,1.,1.,1.],
                            ])
        h_5 = torch.tensor([[1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            ])
        self.register_buffer("h_8", h_8, persistent=False)
        self.register_buffer("h_5", h_5, persistent=False)
        self.k=k
        self.low = low

    @property
    def embedding_dim(self):
        return self.dim

    @property
    def num_embeddings(self):
        return self.codebook_size


    # def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    #     """Bound `z`, an array of shape (..., d)."""
    #     half_l = (self._levels - 1) * (1 - eps) / 2
    #     offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
    #     shift = (offset / half_l).atanh()
    #     return (z + shift).tanh() * half_l - offset
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        # return (z + shift).tanh() * half_l - offset
        z_8 = z[:,:,:,:3]
        z_5 = z[:, :, :, 3:]
        out_8 = TanhStepMerTanhCLFunction.apply(z_8, self.a_8, self.h_8, self.k,self.low)-0.5
        out_5 = TanhStepMerTanhCLFunction.apply(z_5, self.a_5, self.h_5, self.k,self.low)+0.5
        out = torch.cat([out_8,out_5],dim=-1)
        return out


    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        return codes



    def forward(
        self,
        z: torch.Tensor,
        memo: Memo | None,
    ) -> tuple[torch.Tensor, Memo]:
        if memo is None:
            memo = todd.Config()

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        # if is_img_or_video:
        #     z = rearrange(z, 'b d ... -> b ... d')
        #     z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # if is_img_or_video:
        #     out = unpack_one(out, ps, 'b * d')
        #     out = rearrange(out, 'b ... d -> b d ...')
        #
        #     indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        memo['quant'] = indices
        memo['input_z']=z

        return out,memo,indices,codes,z


    if TYPE_CHECKING:
        __call__ = forward



# def dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):
#
#     new_shape = [1] * len(x.shape)
#     new_shape[ch_axis] = x.shape[ch_axis]
#     scale = scale.reshape(new_shape)
#     zero_point = zero_point.reshape(new_shape)
#
#     tanh_scale = 1 / (1 - alpha)
#     tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))
#
#     x = x / scale + zero_point
#     x = torch.clamp(x, quant_min, quant_max)
#     x = x.floor() + (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
#     x = (x.round() - x).detach() + x
#     x = (x - zero_point) * scale
#
#     return x
