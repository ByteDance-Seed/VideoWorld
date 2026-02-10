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
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from ldm.structures import DataSample
from ..utils import  set_requires_grad
from .base_gan import BaseGan
from ldm.registry import MODELS


ModelType = Union[Dict, nn.Module]

@MODELS.register_module()
class VQGAN(BaseGan):

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         generator_steps=generator_steps,
                         data_preprocessor=data_preprocessor,
                         discriminator_steps=discriminator_steps,
                         ema_config=ema_config,
                         loss_config=loss_config)

        self.loss_config = deepcopy(loss_config)
        self.perceptual_weight = loss_config['perceptual_weight']
        self.perceptual_loss = MODELS.build(loss_config['perceptual_loss'])
        self.discriminator_weight = loss_config['disc_weight']


    def disc_loss(self, disc_pred_fake: Tensor, disc_pred_real: Tensor,
                  real_imgs: Tensor) -> Tuple:
        r"""Get disc loss. StyleGANv2 use the non-saturating loss and R1
            gradient penalty to train the discriminator.

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.
            real_imgs (Tensor): Input real images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """

        losses_dict = dict()
        # no-saturating gan loss
        # losses_dict['loss_disc_fake'] = F.softplus(disc_pred_fake).mean()
        # losses_dict['loss_disc_real'] = F.softplus(-disc_pred_real).mean()
        losses_dict['loss_disc_real'] = torch.mean(F.relu(1. - disc_pred_real)) *0.5
        losses_dict['loss_disc_fake'] = torch.mean(F.relu(1. + disc_pred_fake)) *0.5
        losses_dict['sum_disc']=losses_dict['loss_disc_real']+losses_dict['loss_disc_fake']

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            print("-----------------------last layer is None--------------------")

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def gen_loss(self, disc_pred_fake: Tensor, fake_imgs: Tensor, gt_imgs: Tensor ) -> Tuple:
        """Get gen loss. StyleGANv2 use the non-saturating loss and generator
        path regularization to train the generator.

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            batch_size (int): Batch size for generating fake images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        # no-saturating gan loss

        rec_loss = torch.abs(gt_imgs.contiguous() - fake_imgs.contiguous())
        l1_loss = rec_loss
        p_loss = self.perceptual_loss(gt_imgs.contiguous(),fake_imgs.contiguous())*self.perceptual_weight

        rec_loss = rec_loss+p_loss

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        g_loss = -torch.mean(disc_pred_fake)

        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=self.get_last_layer())


        losses_dict['loss_rec'] = nll_loss
        losses_dict['loss_gen']=d_weight*g_loss
        losses_dict['rec_value']=l1_loss

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var
    def get_last_layer(self):
        return self.generator.module.decoder.last_parameter



    def train_generator(self, inputs: Tensor, data_samples: DataSample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        images = inputs


        fake_imgs,codes,indice = self.generator(images)

        disc_pred_fake = self.discriminator(fake_imgs)
        if images.dim() == 5:
            images = images[:, :, 1]
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake, fake_imgs, images)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def train_discriminator(self, inputs: Tensor, data_samples: DataSample,
                            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = inputs

        
        with torch.no_grad():
            fake_imgs,codes,indice = self.generator(real_imgs)
        
        if real_imgs.dim() == 5:
            real_imgs = real_imgs[:, :, 1]
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real, real_imgs)
        optimizer_wrapper.update_params(parsed_losses)
        # save ada info
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('disc_pred_real', disc_pred_real)
        return log_vars


    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMagic's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # import pdb;pdb.set_trace()
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        inputs_dict, data_samples = data['inputs'], data['data_samples']

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # NOTE: Do not use context manager of optim_wrapper. Because
        # in mixed-precision training, StyleGAN2 only enable fp16 in
        # specified blocks (refers to `:attr:enable_fp16` in
        # :class:`~StyleGANv2Generator` and :class:`~StyleGAN2Discriminator`
        # for more details), but in :func:`~AmpOptimWrapper.optim_context`,
        # fp16 is applied to all modules. This may slow down gradient
        # accumulation because `no_sycn` in
        # :func:`~OptimWrapper.optim_context` will not be called any more.
        log_vars = self.train_discriminator(inputs_dict, data_samples,
                                            disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            for _ in range(self.generator_steps * gen_accu_iters):
                log_vars_gen = self.train_generator(inputs_dict, data_samples,
                                                    gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = self.gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen and (curr_iter + 1) >= (
                    self.ema_start * self.discriminator_steps *
                    disc_accu_iters):
                self.generator_ema.update_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)
                # if not update buffer, copy buffer from orig model
                if not self.generator_ema.update_buffers:
                    self.generator_ema.sync_buffers(
                        self.generator.module if is_model_wrapper(
                            self.generator) else self.generator)
            elif self.with_ema_gen:
                # before ema, copy weights from orig
                self.generator_ema.sync_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)

            log_vars.update(log_vars_gen)
        return log_vars
