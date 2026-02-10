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

from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule, build_norm_layer
from mmcv.ops.upfirdn2d import upfirdn2d
from mmcv.ops.fused_bias_leakyrelu import fused_bias_leakyrelu,FusedBiasLeakyReLU
from mmengine.model import BaseModule, normal_init
from torch.nn.init import _calculate_correct_fan
from ldm.registry import MODELS
from mmengine.dist import get_dist_info

from mmengine.runner.amp import autocast
import torch.autograd as autograd
import torch.distributed as dist



class AllGatherLayer(autograd.Function):
    """All gather layer with backward propagation path.

    Indeed, this module is to make ``dist.all_gather()`` in the backward graph.
    Such kind of operation has been widely used in Moco and other contrastive
    learning algorithms.
    """

    @staticmethod
    def forward(ctx, x):
        """Forward function."""
        ctx.save_for_backward(x)
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward function."""
        x, = ctx.saved_tensors
        grad_out = torch.zeros_like(x)
        grad_out = grad_outputs[dist.get_rank()]
        return grad_out

class EqualizedLR:
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.
    """

    def __init__(self, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.0):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        """Compute weight with equalized learning rate.

        Args:
            module (nn.Module): A module that is wrapped with equalized lr.

        Returns:
            torch.Tensor: Updated weight.
        """
        weight = getattr(module, self.name + '_orig')
        if weight.ndim == 5:
            # weight in shape of [b, out, in, k, k]
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = weight * torch.tensor(
            self.gain, device=weight.device) * torch.sqrt(
                torch.tensor(1. / fan, device=weight.device)) * self.lr_mul

        return weight

    def __call__(self, module, inputs):
        """Standard interface for forward pre hooks."""
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2**0.5, mode='fan_in', lr_mul=1.):
        """Apply function.

        This function is to register an equalized learning rate hook in an
        ``nn.Module``.

        Args:
            module (nn.Module): Module to be wrapped.
            name (str | optional): The name of weights. Defaults to 'weight'.
            mode (str, optional): The mode of computing ``fan`` which is the
                same as ``kaiming_init`` in pytorch. You can choose one from
                ['fan_in', 'fan_out']. Defaults to 'fan_in'.

        Returns:
            nn.Module: Module that is registered with equalized lr hook.
        """
        # sanity check for duplicated hooks.
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(
                    'Cannot register two equalized_lr hooks on the same '
                    f'parameter {name} in {module} module.')

        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]

        delattr(module, name)
        module.register_parameter(name + '_orig', weight)

        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # plain attribute.

        setattr(module, name, weight.data)
        module.register_forward_pre_hook(fn)

        # TODO: register load state dict hook

        return fn

def equalized_lr(module, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.):
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        module (nn.Module): Module to be wrapped.
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.

    Returns:
        nn.Module: Module that is registered with equalized lr hook.
    """
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)

    return module

@MODELS.register_module()
class EqualizedLRConvModule(ConvModule):
    r"""Equalized LR ConvModule.

    In this module, we inherit default ``mmcv.cnn.ConvModule`` and adopt
    equalized lr in convolution. The equalized learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.conv`` will be overwritten as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super().__init__(*args, **kwargs)
        self.with_equalized_lr = equalized_lr_cfg is not None
        if self.with_equalized_lr:
            self.conv = equalized_lr(self.conv, **equalized_lr_cfg)
            # initialize the conv weight with standard Gaussian noise.
            self._init_conv_weights()

    def _init_conv_weights(self):
        """Initialize conv weights as described in PGGAN."""
        normal_init(self.conv)



@MODELS.register_module()
class EqualizedLRLinearModule(nn.Linear):
    r"""Equalized LR LinearModule.

    In this module, we adopt equalized lr in ``nn.Linear``. The equalized
    learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.weight`` will be overwritten as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super().__init__(*args, **kwargs)
        self.with_equalized_lr = equalized_lr_cfg is not None
        if self.with_equalized_lr:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            # In fact, lr_mul will only be used in EqualizedLR for
            # initialization
            self.lr_mul = 1.
        if self.with_equalized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1. / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(BaseModule):
    """Blur module.

    This module is adopted rightly after upsampling operation in StyleGAN2.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        pad (list[int]): Padding for features.
        upsample_factor (int, optional): Upsampling factor. Defaults to 1.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """

        # In Tero's implementation, he uses fp32
        return upfirdn2d(x.to(torch.float32), self.kernel.to(torch.float32), padding=self.pad)


class EqualLinearActModule(BaseModule):
    """Equalized LR Linear Module with Activation Layer.

    This module is modified from ``EqualizedLRLinearModule`` defined in PGGAN.
    The major features updated in this module is adding support for activation
    layers used in StyleGAN2.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for equalized lr.
            Defaults to dict(gain=1., lr_mul=1.).
        bias (bool, optional): Whether to use bias item. Defaults to True.
        bias_init (float, optional): The value for bias initialization.
            Defaults to ``0.``.
        act_cfg (dict | None, optional): Config for activation layer.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 equalized_lr_cfg=dict(gain=1., lr_mul=1.),
                 bias=True,
                 bias_init=0.,
                 act_cfg=None,
                 **kwargs):
        super().__init__()
        self.with_activation = act_cfg is not None
        # w/o bias in linear layer
        self.linear = EqualizedLRLinearModule(
            *args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs)

        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            self.lr_mul = 1.

        # define bias outside linear layer
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.linear.out_features).fill_(bias_init))
        else:
            self.bias = None

        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = MODELS.build(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)

        return x


class EqualLinearActModule(BaseModule):
    """Equalized LR Linear Module with Activation Layer.

    This module is modified from ``EqualizedLRLinearModule`` defined in PGGAN.
    The major features updated in this module is adding support for activation
    layers used in StyleGAN2.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for equalized lr.
            Defaults to dict(gain=1., lr_mul=1.).
        bias (bool, optional): Whether to use bias item. Defaults to True.
        bias_init (float, optional): The value for bias initialization.
            Defaults to ``0.``.
        act_cfg (dict | None, optional): Config for activation layer.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 equalized_lr_cfg=dict(gain=1., lr_mul=1.),
                 bias=True,
                 bias_init=0.,
                 act_cfg=None,
                 **kwargs):
        super().__init__()
        self.with_activation = act_cfg is not None
        # w/o bias in linear layer
        self.linear = EqualizedLRLinearModule(
            *args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs)

        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            self.lr_mul = 1.

        # define bias outside linear layer
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.linear.out_features).fill_(bias_init))
        else:
            self.bias = None

        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = MODELS.build(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)

        return x



class _FusedBiasLeakyReLU(FusedBiasLeakyReLU):
    """Wrap FusedBiasLeakyReLU to support FP16 training."""

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        return fused_bias_leakyrelu(x, self.bias.to(x.dtype),
                                    self.negative_slope, self.scale)

class ConvDownLayer(nn.Sequential):
    """Convolution and Downsampling layer.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        bias (bool, optional): Whether to use bias parameter. Defaults to True.
        act_cfg (dict, optional): Activation configs.
            Defaults to dict(type='fused_bias').
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 act_cfg=dict(type='fused_bias'),
                 fp16_enabled=False,
                 conv_clamp=256.):

        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        self.with_fused_bias = act_cfg is not None and act_cfg.get(
            'type') == 'fused_bias'
        if self.with_fused_bias:
            conv_act_cfg = None
        else:
            conv_act_cfg = act_cfg
        layers.append(
            EqualizedLRConvModule(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not self.with_fused_bias,
                norm_cfg=None,
                act_cfg=conv_act_cfg,
                equalized_lr_cfg=dict(mode='fan_in', gain=1.)))
        if self.with_fused_bias:
            layers.append(_FusedBiasLeakyReLU(out_channels))

        super(ConvDownLayer, self).__init__(*layers)

    # @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        with autocast(enabled=self.fp16_enabled):
            x = super().forward(x)
            if self.fp16_enabled:
                x = torch.clamp(x, min=-self.conv_clamp, max=self.conv_clamp)
        return x


class ResBlock(BaseModule):
    """Residual block used in the discriminator of StyleGAN2.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        convert_input_fp32 (bool, optional): Whether to convert input type to
            fp32 if not `fp16_enabled`. This argument is designed to deal with
            the cases where some modules are run in FP16 and others in FP32.
            Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 blur_kernel=[1, 3, 3, 1],
                 fp16_enabled=False,
                 convert_input_fp32=True):
        super().__init__()

        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32

        self.conv1 = ConvDownLayer(
            in_channels,
            in_channels,
            3,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(
            in_channels,
            out_channels,
            3,
            downsample=True,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)

        self.skip = ConvDownLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            act_cfg=None,
            bias=False,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)

    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        # TODO: study whether this explicit datatype transfer will harm the
        # apex training speed
        if not self.fp16_enabled and self.convert_input_fp32:
            input = input.to(torch.float32)

        with autocast(enabled=self.fp16_enabled):
            out = self.conv1(input)
            out = self.conv2(out)

            skip = self.skip(input)
            out = (out + skip) / np.sqrt(2)

        return out



class ModMBStddevLayer(BaseModule):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.

    Note that to accelerate the training procedure, we also add a new feature
    of ``sync_std`` to achieve multi-nodes/machine training. This feature is
    still in beta version and we have tested it on 256 scales.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        channel_groups (int, optional): The size of groups in channel
            dimension. Defaults to 1.
        sync_std (bool, optional): Whether to use synchronized std feature.
            Defaults to False.
        sync_groups (int | None, optional): The size of groups in node
            dimension. Defaults to None.
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self,
                 group_size=4,
                 channel_groups=1,
                 sync_std=False,
                 sync_groups=None,
                 eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_std = sync_std
        self.sync_groups = group_size if sync_groups is None else sync_groups

        if self.sync_std:
            assert torch.distributed.is_initialized(
            ), 'Only in distributed training can the sync_std be activated.'
            mmengine.print_log('Adopt synced minibatch stddev layer', 'mmagic')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map with shape of (N, C+1, H, W).
        """

        if self.sync_std:
            # concatenate all features
            all_features = torch.cat(AllGatherLayer.apply(x), dim=0)
            # get the exact features we need in calculating std-dev
            rank, ws = get_dist_info()
            local_bs = all_features.shape[0] // ws
            start_idx = local_bs * rank
            # avoid the case where start idx near the tail of features
            if start_idx + self.sync_groups > all_features.shape[0]:
                start_idx = all_features.shape[0] - self.sync_groups
            end_idx = min(local_bs * rank + self.sync_groups,
                          all_features.shape[0])

            x = all_features[start_idx:end_idx]

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        assert x.shape[1] % self.channel_groups == 0, (
            '"channel_groups" must be divided by the feature channels. '
            f'channel_groups: {self.channel_groups}, '
            f'feature channels: {x.shape[1]}')

        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, Gc, C', H, W]
        y = torch.reshape(x, (group_size, -1, self.channel_groups,
                              c // self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)
