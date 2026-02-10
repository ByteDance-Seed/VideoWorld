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
from functools import partial

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange, repeat
try:
    from flash_attn_3.flash_attn_interface import flash_attn_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=torch.bfloat16,
    ori_size=None
):
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    is_half = dtype in [torch.bfloat16, torch.float16]
    compute_cap = get_device_cc(q.device)

    if dtype not in supported_dtypes:
        raise NotImplementedError(f"{dtype=} is not supported.")

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale
    if causal:
        # import pdb;pdb.set_trace()
        assert ori_size is not None
        B, T, H, W = ori_size
        # attn_mask = torch.ones((B*H*W, T, T), dtype=torch.bool, device=q.device)
        # attn_mask = torch.tril(attn_mask)
        # attn_mask = rearrange(attn_mask, "(b h w) t d -> b (t h w) d", b=B, h=H, w=W, t=T, d=T)
        # k = k[:, :T]
        # v = v[:, :T]
        q_len = q.shape[1]
        q_indices = torch.arange(q_len, device=q.device)
        k_indices = torch.arange(T, device=q.device)
        q_frame_indices = q_indices // (H * W)
        attn_mask = k_indices.view(1, T) <= q_frame_indices.view(q_len, 1)
        padding_attn_mask = torch.ones((B*H*W*T, 512-T), dtype=torch.bool, device=q.device)
        attn_mask = torch.cat((attn_mask, padding_attn_mask), dim=1)

    else:
        attn_mask = None
    # If Flash Attention 3 is installed, and the user's running on a Hopper GPU (compute capability
    # 9.0, or SM90), use Flash Attention 3.
    if compute_cap == 90 and FLASH_ATTN_3_AVAILABLE and is_half:
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0]
    else:
        # If Blackwell or Hopper (SM100 or SM90), cuDNN has native FMHA kernels. The Hopper one is
        # not always as fast as Flash Attention 3, but when Flash Attention is unavailable, it's
        # still a far better choice than Flash Attention 2 (Ampere).
        if compute_cap in [90, 100] and is_half:
            SDPA_BACKENDS = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.CUDNN_ATTENTION
        elif is_half:
            SDPA_BACKENDS = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION
        else:
            assert dtype == torch.float32, f"Unrecognized {dtype=}."
            SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION]
            BEST_SDPA_BACKEND = SDPBackend.EFFICIENT_ATTENTION

        if deterministic:
            raise NotImplementedError(
                "Deterministic mode in attention is only supported when Flash Attention 3 is available."
            )

        # Torch 2.6 and later allows priorities for backends, but for older versions
        # we can only run with a specific backend. As long as we pick ones we're certain
        # will work on that device, it should be fine.
        try:
            sdpa_kernel(backends=SDPA_BACKENDS, set_priority_order=True)
            sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
        except TypeError:
            sdpa_kernel_ = sdpa_kernel
            SDPA_BACKENDS = [BEST_SDPA_BACKEND]
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attn_mask is not None:
            SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel_(backends=SDPA_BACKENDS):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )

        out = out.transpose(1, 2).contiguous()
        return out
