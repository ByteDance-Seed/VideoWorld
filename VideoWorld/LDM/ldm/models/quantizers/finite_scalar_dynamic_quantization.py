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
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from ldm.registry import MODELS


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

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class LTQSoft(nn.Module):
    def __init__(self,num_splits, input_channels,up=1.0,low=-1.0):
        super().__init__()
        init_range = up-low
        self.n_val = num_splits - 1
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.ones((1, input_channels)) * low, requires_grad=True)
        self.a = nn.Parameter(torch.ones(input_channels, self.n_val) * self.interval, requires_grad=True)
        self.scale1 = nn.Parameter(torch.ones((1, input_channels)), requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones((1, input_channels)), requires_grad=True)
        self.eps = 1e-3
        self.input_channels = input_channels

        step_right = torch.tensor(torch.ones(1, input_channels) * low)
        self.register_buffer("step_right", step_right, persistent=False)

        self.register_buffer("zero", torch.ones(1, input_channels) * low, persistent=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, input_channels), requires_grad=True)
        self.register_buffer("up", torch.Tensor([up]), persistent=False)
        self.len_wi = up - low
        self.low = low

    def locate_intervals(self, x):
        x = x * self.scale1
        a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        a_pos = F.softmax(a_pos) * self.len_wi
        thresholds = []  # 初始化阈值列表，包含起始阈值

        # 计算所有区间的阈值
        for i in range(self.n_val):
            # 更新阈值
            if i == 0:
                new_threshold = self.start + a_pos[:, 0] / 2
            else:
                new_threshold = thresholds[i - 1] + a_pos[:, i - 1] / 2 + a_pos[:, i] / 2
            thresholds.append(new_threshold)

        # 使用计算出的阈值确定每个元素所在的区间
        intervals = torch.zeros_like(x)  # 初始化区间索引矩阵
        for i, thre in enumerate(thresholds):
            intervals += (x > thre).float()  # 如果x中的元素大于阈值，则区间索引加1
        return intervals

    def indice_to_code(self,indices):
        code_array = torch.ones((self.input_channels, self.n_val + 1), device=indices.device) * self.low
        for i in range(1, self.n_val + 1):
            code_array[:, i] = code_array[:, i - 1] + self.interval

        b, l, c, d = indices.size()
        indices = rearrange(indices, "b l c d->( b l c ) d")

        code = torch.gather(code_array.unsqueeze(dim=0).expand(b * l * c, -1, -1), 2,
                            indices.to(torch.int64).unsqueeze(dim=2))
        code = code.squeeze(dim=-1).view(b, l, c, d)
        out = code * self.scale2
        out = out + self.bias
        return out

    def forward(self,x):
        x = x * self.scale1

        x_forward = x
        x_backward = x

        step_right = self.zero + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        a_pos = F.softmax(a_pos) * self.len_wi

        for i in range(self.n_val):
            step_right += self.interval
            if i ==0:
                thre_forward = self.start + a_pos[:, 0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.zero)
                x_backward = torch.where(x > thre_backward,
                                         self.interval / a_pos[:, i] * (x - thre_backward) + step_right - self.interval,
                                         self.zero)
            else:
                thre_forward += a_pos[:, i - 1] / 2 + a_pos[:, i] / 2
                thre_backward += a_pos[:, i - 1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(x > thre_backward,
                                         self.interval / a_pos[:, i] * (x - thre_backward) + step_right - self.interval,
                                         x_backward)

        thre_backward += a_pos[:, i]
        x_backward = torch.where(x > thre_backward, self.up, x_backward)
        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out * self.scale2
        out = out + self.bias
        return out


@MODELS.register_module()
class FDQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        use_tanh = False,
        up = 1.0,
        low = -1.0,
        drop_out = 0.0,
        drop_out_f = 0.0,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)

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

        # implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.lqt_8 = LTQSoft(num_splits=8, input_channels=3, up=up, low=low)
        self.lqt_5 = LTQSoft(num_splits=5, input_channels=3, up=up, low=low)
        self.use_tanh = use_tanh
        if drop_out == 0.0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=drop_out)

        if drop_out_f == 0.0:
            self.drop_f = nn.Identity()
        else:
            self.drop_f = nn.Dropout(p=drop_out_f)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # quantized = round_ste(self.bound(z))
        # half_width = self._levels // 2  # Renormalize to [-1, 1].
        # return quantized / half_width
        if self.use_tanh:
            z = F.tanh(z)

        z_8 = z[:,:,:,:3]
        z_5 = z[:,:,:,3:]
        code_8 = self.lqt_8(z_8)
        code_5 = self.lqt_5(z_5)
        out = torch.cat([code_8,code_5],dim=-1)
        return out

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim
        # zhat = self._scale_and_shift(zhat)
        # return (zhat * self._basis).sum(dim=-1).to(int32)
        z_8 = z[:, :, :, :3]
        z_5 = z[:, :, :, 3:]
        indice_8 = self.lqt_8.locate_intervals(z_8)
        indice_5 = self.lqt_5.locate_intervals(z_5)
        out = torch.cat([indice_8, indice_5], dim=-1)

        return (out * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        if indices.size()[-1] != 1:
            b, w, h = indices.size()
            indices = indices.view(b, -1, 1)


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        index_8 = codes_non_centered[:, :, :, :3]
        index_5 = codes_non_centered[:, :, :, 3:]

        code_8=self.lqt_8.indice_to_code(index_8)
        code_5 =self.lqt_5.indice_to_code(index_5)
        codes = torch.cat([code_8,code_5],dim=-1)


        codes=rearrange(codes, 'b n c d -> b n (c d)')

        if project_out:
            codes = self.project_out(codes)

        b, l, d = codes.size()
        codes = codes.view(b, int(l ** 0.5), int(l ** 0.5), -1)

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

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

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices


@MODELS.register_module()
class FDQNorm(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        use_norm = True,
        up = 1.0,
        low = -1.0,
        drop_out = 0.0,
        drop_out_f = 0.0,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)

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

        # implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.lqt_8 = LTQSoft(num_splits=8, input_channels=3, up=up, low=low)
        self.lqt_5 = LTQSoft(num_splits=5, input_channels=3, up=up, low=low)
        self.use_norm = use_norm
        if drop_out == 0.0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=drop_out)

        if drop_out_f == 0.0:
            self.drop_f = nn.Identity()
        else:
            self.drop_f = nn.Dropout(p=drop_out_f)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # quantized = round_ste(self.bound(z))
        # half_width = self._levels // 2  # Renormalize to [-1, 1].
        # return quantized / half_width
        if self.use_norm:
            z = F.normalize(z,p=2,dim=-1)

        z_8 = z[:,:,:,:3]
        z_5 = z[:,:,:,3:]
        code_8 = self.lqt_8(z_8)
        code_5 = self.lqt_5(z_5)
        out = torch.cat([code_8,code_5],dim=-1)
        return out

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim
        # zhat = self._scale_and_shift(zhat)
        # return (zhat * self._basis).sum(dim=-1).to(int32)
        z_8 = z[:, :, :, :3]
        z_5 = z[:, :, :, 3:]
        indice_8 = self.lqt_8.locate_intervals(z_8)
        indice_5 = self.lqt_5.locate_intervals(z_5)
        out = torch.cat([indice_8, indice_5], dim=-1)

        return (out * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        if indices.size()[-1] != 1:
            b, w, h = indices.size()
            indices = indices.view(b, -1, 1)


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        index_8 = codes_non_centered[:, :, :, :3]
        index_5 = codes_non_centered[:, :, :, 3:]

        code_8=self.lqt_8.indice_to_code(index_8)
        code_5 =self.lqt_5.indice_to_code(index_5)
        codes = torch.cat([code_8,code_5],dim=-1)


        codes=rearrange(codes, 'b n c d -> b n (c d)')

        if project_out:
            codes = self.project_out(codes)

        b, l, d = codes.size()
        codes = codes.view(b, int(l ** 0.5), int(l ** 0.5), -1)

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

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

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices


@MODELS.register_module()
class FDQNorm2(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        use_norm = True,
        up = 1.0,
        low = -1.0,
        drop_out = 0.0,
        drop_out_f = 0.0,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)

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

        # implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.lqt_8 = LTQSoft(num_splits=8, input_channels=3, up=up, low=low)
        self.lqt_5 = LTQSoft(num_splits=5, input_channels=3, up=up, low=low)
        self.use_norm = use_norm
        if drop_out == 0.0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=drop_out)

        if drop_out_f == 0.0:
            self.drop_f = nn.Identity()
        else:
            self.drop_f = nn.Dropout(p=drop_out_f)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # quantized = round_ste(self.bound(z))
        # half_width = self._levels // 2  # Renormalize to [-1, 1].
        # return quantized / half_width
        if self.use_norm:
            z = F.normalize(z,p=2,dim=-1)

        z_8 = z[:,:,:,:3]
        z_5 = z[:,:,:,3:]
        code_8 = self.lqt_8(z_8)
        code_5 = self.lqt_5(z_5)
        out = torch.cat([code_8,code_5],dim=-1)
        if self.use_norm:
            out = F.normalize(out,p=2,dim=-1)

        return out

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim
        # zhat = self._scale_and_shift(zhat)
        # return (zhat * self._basis).sum(dim=-1).to(int32)
        z_8 = z[:, :, :, :3]
        z_5 = z[:, :, :, 3:]
        indice_8 = self.lqt_8.locate_intervals(z_8)
        indice_5 = self.lqt_5.locate_intervals(z_5)
        out = torch.cat([indice_8, indice_5], dim=-1)

        return (out * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        if indices.size()[-1] != 1:
            b, w, h = indices.size()
            indices = indices.view(b, -1, 1)


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        index_8 = codes_non_centered[:, :, :, :3]
        index_5 = codes_non_centered[:, :, :, 3:]

        code_8=self.lqt_8.indice_to_code(index_8)
        code_5 =self.lqt_5.indice_to_code(index_5)
        codes = torch.cat([code_8,code_5],dim=-1)


        codes=rearrange(codes, 'b n c d -> b n (c d)')

        if project_out:
            codes = self.project_out(codes)

        b, l, d = codes.size()
        codes = codes.view(b, int(l ** 0.5), int(l ** 0.5), -1)

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

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

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices


@MODELS.register_module()
class FDQ_16384(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        use_tanh = False,
        up = 1.0,
        low = -1.0,
        drop_out = 0.0,
        drop_out_f = 0.0,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)

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

        # implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.lqt_8 = LTQSoft(num_splits=8, input_channels=3, up=up, low=low)
        self.lqt_6 = LTQSoft(num_splits=6, input_channels=1, up=up, low=low)
        self.lqt_5 = LTQSoft(num_splits=5, input_channels=1, up=up, low=low)
        self.use_tanh = use_tanh
        if drop_out == 0.0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=drop_out)

        if drop_out_f == 0.0:
            self.drop_f = nn.Identity()
        else:
            self.drop_f = nn.Dropout(p=drop_out_f)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # quantized = round_ste(self.bound(z))
        # half_width = self._levels // 2  # Renormalize to [-1, 1].
        # return quantized / half_width
        if self.use_tanh:
            z = F.tanh(z)

        z_8 = z[:,:,:,:3]
        z_6 = z[:,:,:,3:4]
        z_5 = z[:,:,:,4:]
        code_8 = self.lqt_8(z_8)
        code_6 = self.lqt_6(z_6)
        code_5 = self.lqt_5(z_5)
        out = torch.cat([code_8,code_6,code_5],dim=-1)
        return out

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim
        # zhat = self._scale_and_shift(zhat)
        # return (zhat * self._basis).sum(dim=-1).to(int32)
        z_8 = z[:, :, :, :3]
        z_6 = z[:, :, :, 3:4]
        z_5 = z[:, :, :, 4:]
        indice_8 = self.lqt_8.locate_intervals(z_8)
        indice_6 = self.lqt_6.locate_intervals(z_6)
        indice_5 = self.lqt_5.locate_intervals(z_5)
        out = torch.cat([indice_8, indice_6, indice_5], dim=-1)

        return (out * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        if indices.size()[-1] != 1:
            b, w, h = indices.size()
            indices = indices.view(b, -1, 1)


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        index_8 = codes_non_centered[:, :, :, :3]
        index_6 = codes_non_centered[:, :, :, 3:4]
        index_5 = codes_non_centered[:, :, :, 4:]

        code_8=self.lqt_8.indice_to_code(index_8)
        code_6 = self.lqt_6.indice_to_code(index_6)
        code_5 =self.lqt_5.indice_to_code(index_5)
        codes = torch.cat([code_8,code_6,code_5],dim=-1)


        codes=rearrange(codes, 'b n c d -> b n (c d)')

        if project_out:
            codes = self.project_out(codes)

        b, l, d = codes.size()
        codes = codes.view(b, int(l ** 0.5), int(l ** 0.5), -1)

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

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

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices


@MODELS.register_module()
class FDQ_15360(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        use_tanh = False,
        up = 1.0,
        low = -1.0,
        drop_out = 0.0,
        drop_out_f = 0.0,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)

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

        # implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.lqt_8 = LTQSoft(num_splits=8, input_channels=2, up=up, low=low)
        self.lqt_5 = LTQSoft(num_splits=5, input_channels=1, up=up, low=low)
        self.lqt_4 = LTQSoft(num_splits=4, input_channels=2, up=up, low=low)
        self.lqt_3 = LTQSoft(num_splits=3, input_channels=1, up=up, low=low)
        self.use_tanh = use_tanh
        if drop_out == 0.0:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(p=drop_out)

        if drop_out_f == 0.0:
            self.drop_f = nn.Identity()
        else:
            self.drop_f = nn.Dropout(p=drop_out_f)

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # quantized = round_ste(self.bound(z))
        # half_width = self._levels // 2  # Renormalize to [-1, 1].
        # return quantized / half_width
        if self.use_tanh:
            z = F.tanh(z)

        z_8 = z[:,:,:,:2]
        z_5 = z[:,:,:,2:3]
        z_4 = z[:,:,:,3:5]
        z_3 = z[:, :, :, 5:]

        code_8 = self.lqt_8(z_8)
        code_5 = self.lqt_5(z_5)
        code_4 = self.lqt_4(z_4)
        code_3 = self.lqt_5(z_3)
        out = torch.cat([code_8,code_5,code_4,code_3],dim=-1)
        return out

    def codes_to_indices(self, z: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim
        # zhat = self._scale_and_shift(zhat)
        # return (zhat * self._basis).sum(dim=-1).to(int32)
        z_8 = z[:, :, :, :2]
        z_5 = z[:, :, :, 2:3]
        z_4 = z[:, :, :, 3:5]
        z_3 = z[:, :, :, 5:]
        indice_8 = self.lqt_8.locate_intervals(z_8)
        indice_5 = self.lqt_5.locate_intervals(z_5)
        indice_4 = self.lqt_4.locate_intervals(z_4)
        indice_3 = self.lqt_3.locate_intervals(z_3)
        out = torch.cat([indice_8, indice_5, indice_4,indice_3], dim=-1)

        return (out * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        if indices.size()[-1] != 1:
            b, w, h = indices.size()
            indices = indices.view(b, -1, 1)


        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        index_8 = codes_non_centered[:, :, :, :2]
        index_5 = codes_non_centered[:, :, :, 2:3]
        index_4 = codes_non_centered[:, :, :, 3:5]
        index_3 = codes_non_centered[:, :, :, 5:]

        code_8=self.lqt_8.indice_to_code(index_8)
        code_5 = self.lqt_5.indice_to_code(index_5)
        code_4 = self.lqt_4.indice_to_code(index_4)
        code_3 =self.lqt_3.indice_to_code(index_3)

        codes = torch.cat([code_8,code_5,code_4,code_3],dim=-1)


        codes=rearrange(codes, 'b n c d -> b n (c d)')

        if project_out:
            codes = self.project_out(codes)

        b, l, d = codes.size()
        codes = codes.view(b, int(l ** 0.5), int(l ** 0.5), -1)

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

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

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices