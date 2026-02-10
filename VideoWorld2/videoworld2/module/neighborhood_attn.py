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
from collections.abc import Mapping, Sized
from typing import Optional

import torch
from torch import nn

# from imaginaire.utils import log

try:
    import natten
    from natten.functional import neighborhood_attention_generic

    natten.use_kv_parallelism_in_fused_na(True)
    natten.set_memory_usage_preference("unrestricted")

    HAS_NATTEN = True

except ImportError:
    HAS_NATTEN = False

    def neighborhood_attention_generic(*args, **kwargs):
        raise RuntimeError("You attempted to run neighborhood attention, but you don't have NATTEN installed.")


from collections import namedtuple

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class NeighborhoodAttention(nn.Module):
    def __init__(self, gna_parameters, base_attn_op):
        super(NeighborhoodAttention, self).__init__()

        self.base_attn_op = base_attn_op

        self.gna_parameters = gna_parameters
        if (
            not isinstance(gna_parameters, Mapping)
            or "window_size" not in gna_parameters
            or "stride" not in gna_parameters
        ):
            raise ValueError(
                "Expected `gna_parameters` to be a dict with keys window_size " f"and stride, got {gna_parameters=}."
            )

        self.natten_window_size = gna_parameters["window_size"]
        self.natten_stride = gna_parameters["stride"]
        self.natten_base_size = None if "base_size" not in gna_parameters else gna_parameters["base_size"]

        if not isinstance(self.natten_window_size, Sized) or len(self.natten_window_size) != 3:
            raise ValueError(
                "Invalid window_size value. Expected an iterable of length 3, got " f"{self.natten_window_size}."
            )

        if (not isinstance(self.natten_stride, Sized) or len(self.natten_stride) != 3) and not isinstance(
            self.natten_stride, int
        ):
            raise ValueError(
                "Invalid stride value. Expected an iterable of length 3, or integer, got " f"{self.natten_stride}."
            )

        if self.natten_base_size is not None and (
            not isinstance(self.natten_base_size, Sized) or len(self.natten_base_size) != 3
        ):
            raise ValueError(
                "Invalid base feature map size. Expected an iterable of length 3, or None, got "
                f"{self.natten_base_size}."
            )

        # Flex can be faster with multi-dim tiling, but it's buggy...
        self.backend = "fna"  # Choices (as of natten v0.20.0.dev0): fna, fna-blackwell, flex

        # Kernel configurations

        self.q_tile_shape = None
        self.kv_tile_shape = None
        # self.q_tile_shape = (4, 4, 4)
        # self.kv_tile_shape = (4, 4, 4)

        self.backward_q_tile_shape = None  # fna only
        self.backward_kv_tile_shape = None  # fna only
        self.backward_kv_splits = None  # fna only
        self.backward_use_pt_reduction = False  # fna only

        self.run_persistent_kernel = True  # blackwell-fna only

    def forward(
        self,
        q_B_L_H_D: torch.Tensor,
        k_B_L_H_D: torch.Tensor,
        v_B_L_H_D: torch.Tensor,
        video_size: Optional[VideoSize] = None,
    ):
        if not (q_B_L_H_D.shape == k_B_L_H_D.shape == v_B_L_H_D.shape):
            raise ValueError(
                "NATTEN requires QKV shapes to match, got "
                f"{q_B_L_H_D.shape=}, {k_B_L_H_D.shape=}, {v_B_L_H_D.shape=}."
            )

        batch, seqlen, heads, head_dim = q_B_L_H_D.shape
        T, H, W = video_size

        if seqlen != T * H * W:
            raise ValueError("Mismatch between seqlen and video_size dimensions; got " f"{video_size=}, {seqlen=}.")

        if T > 1:
            # assert T in [20, 24]

            input_shape = (T, H, W)

            # 720p
            # window_size = (T, 12, 24)
            # stride = (1, 4, 8)

            # 480p
            # window_size = (T, 12, 16)
            # stride = (1, 4, 16)

            window_size = tuple(w if w > 1 else x for x, w in zip(input_shape, self.natten_window_size))
            stride = (
                tuple(self.natten_stride for _ in range(3))
                if isinstance(self.natten_stride, int)
                else tuple(x for x in self.natten_stride)
            )

            # Scale window size and stride according to some base input size
            # For example, if window size is (8, 8, 8), stride is (1, 2, 2), for a base
            # input/feature map size of (16, 16, 16); then if the input feat map in this iteration
            # has shape (8, 8, 8), we should use window size (4, 4, 4), and stride (1, 1, 1).
            if self.natten_base_size is not None:
                base_shape = tuple(b if b > 0 else x for x, b in zip(input_shape, self.natten_base_size))

                scale = tuple(x / b for x, b in zip(input_shape, base_shape))

                scaled_window_size = tuple(
                    min(max(2, round(w * s)), x) for w, s, x in zip(window_size, scale, input_shape)
                )
                scaled_stride = tuple(
                    min(max(1, round(st * s)), w) for w, s, st in zip(scaled_window_size, scale, stride)
                )

                window_size = scaled_window_size
                stride = scaled_stride

            assert all(x >= w for x, w in zip(input_shape, window_size))
            assert all(w >= s for w, s in zip(window_size, stride))

        elif T == 1:
            # Do self attention for image model
            return self.base_attn_op(q_B_L_H_D, k_B_L_H_D, v_B_L_H_D)

        else:
            raise ValueError(f"Invalid dimension {T=}.")

        dilation = 1  # = (1, 1, 1)
        causal = False  # = (False, False, False)

        q = q_B_L_H_D.view(batch, *input_shape, heads, head_dim)
        k = k_B_L_H_D.view(batch, *input_shape, heads, head_dim)
        v = v_B_L_H_D.view(batch, *input_shape, heads, head_dim)

        # log.debug(
        #     f"Running neighborhood attention on qkv.shape={q.shape} ({input_shape=}), "
        #     f"{window_size=}, {stride=}, {dilation=}, {causal=}, "
        #     f"q_tile_shape={self.q_tile_shape}, kv_tile_shape={self.kv_tile_shape}, backend={self.backend}."
        # )
        out = neighborhood_attention_generic(
            query=q,
            key=k,
            value=v,
            kernel_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=causal,
            backend=self.backend,
            q_tile_shape=self.q_tile_shape,
            kv_tile_shape=self.kv_tile_shape,
            backward_q_tile_shape=self.backward_q_tile_shape,
            backward_kv_tile_shape=self.backward_kv_tile_shape,
            backward_kv_splits=self.backward_kv_splits,
            backward_use_pt_reduction=self.backward_use_pt_reduction,
            run_persistent_kernel=self.run_persistent_kernel,
        )

        return out.view(batch, seqlen, heads, head_dim)
