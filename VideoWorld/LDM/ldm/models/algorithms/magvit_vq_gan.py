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
from ldm.utils.typing import  SampleList
from ..utils import  set_requires_grad
from .base_gan import BaseGan
from ldm.registry import MODELS
from einops import rearrange,repeat
from .magvit_utils import pick_video_frame,gradient_penalty
import cv2
import numpy as np

ModelType = Union[Dict, nn.Module]

@MODELS.register_module()
class MagVitVQGAN(BaseGan):
    """
    random sample a  frame to compute dis loss
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None,
                 only_last_frame=False,
                 test_return_feat=False,
                 only_2frame=False,
                 select_frame=[],
                 ):
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
        self.grad_penalty_loss_weight=loss_config['grad_penalty_loss_weight']
        self.rec_weight = loss_config.get('rec_weight',1.0)
        self.only_last_frame = only_last_frame
        self.select_frame = select_frame
        self.test_return_feat = test_return_feat
        self.only_2frame = only_2frame
        if self.grad_penalty_loss_weight>0:
            self.apply_gradient_penalty=True
        else:
            self.apply_gradient_penalty=False

        

    def disc_loss(self, disc_pred_fake: Tensor, disc_pred_real: Tensor,real: Tensor) -> Tuple:
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
        losses_dict['sum_disc']=losses_dict['loss_disc_real'].detach()+losses_dict['loss_disc_fake'].detach()

        if self.apply_gradient_penalty:
            gradient_penalty_loss = gradient_penalty(real, disc_pred_real)
        else:
            gradient_penalty_loss=0.0

        losses_dict['loss_disc_grad_penalty']=gradient_penalty_loss*self.grad_penalty_loss_weight

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

    def gen_loss(self, disc_pred_fake: Tensor, recon_video: Tensor, video: Tensor ,frame_indices: Tensor) -> Tuple:
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

        # rec_loss = F.mse_loss(recon_video.contiguous(),video.contiguous())
        rec_loss = torch.mean(torch.abs(video.contiguous()-recon_video.contiguous()))

        batch, channels, frames = video.shape[:3]

        # pickle the frame for perceptual loss
        input_vgg_input = pick_video_frame(video, frame_indices)
        recon_vgg_input = pick_video_frame(recon_video, frame_indices)
        if channels == 1:
            input_vgg_input = repeat(input_vgg_input, 'b 1 h w -> b c h w', c=3)
            recon_vgg_input = repeat(recon_vgg_input, 'b 1 h w -> b c h w', c=3)

        elif channels == 4:
            input_vgg_input = input_vgg_input[:, :3]
            recon_vgg_input = recon_vgg_input[:, :3]


        p_loss = self.perceptual_loss(input_vgg_input.contiguous(), recon_vgg_input.contiguous())*self.perceptual_weight

        rec_loss = rec_loss+p_loss

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        g_loss = -torch.mean(disc_pred_fake)

        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=self.get_last_layer())


        losses_dict['loss_rec'] = nll_loss*self.rec_weight
        losses_dict['loss_gen']=d_weight*g_loss
        losses_dict['rec_value'] = rec_loss.detach()
        losses_dict['d_weight']=d_weight.detach()

        
        return losses_dict 
    def get_last_layer(self):
        # import pdb;pdb.set_trace()
        return self.generator.module.decoder.last_parameter()



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
        video_or_images = inputs
        is_image = video_or_images.ndim == 4
        video_contains_first_frame=True
        video = None
        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True
        else:
            video = video_or_images
        # import pdb;pdb.set_trace()
        if self.only_2frame:
            video = video[:, :, :2]
            
        
        recon_video, codes, indice, Q, _ = self.generator(video, cond=None, video_contains_first_frame=video_contains_first_frame)
        batch, channels, frames = video.shape[:3]
        # import pdb;pdb.set_trace()
        frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices
       
            
            
        if self.only_last_frame:
            frame_indices[:] = -1
            video = pick_video_frame(video, frame_indices)[:, :, None]
            recon_video_frames = pick_video_frame(recon_video, frame_indices)
            # recon_video = recon_video[:, :, 1:]
        elif len(self.select_frame) > 0:
            video = torch.stack([video[:, :, idx] for idx in self.select_frame], dim=2)
            frames = len(self.select_frame)
            frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices
            recon_video_frames = pick_video_frame(recon_video[:, :, 1:], frame_indices)
            recon_video = recon_video[:, :, 1:]
        else:
            recon_video_frames = pick_video_frame(recon_video, frame_indices)
        
        disc_pred_fake = self.discriminator(recon_video_frames) # fake_logits

        gen_losses_dict = self.gen_loss(disc_pred_fake, recon_video, video,frame_indices)
        parsed_loss, log_vars = self.parse_losses(gen_losses_dict)
    
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
        video_or_images = inputs
        is_image = video_or_images.ndim == 4
        video_contains_first_frame=True
        video = None
        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True
        else:
            video = video_or_images
        if self.only_2frame:
            video = video[:, :, :2]
        
        
        with torch.no_grad():
            recon_video,codes,indice, Q, _ = self.generator(video, cond= None, video_contains_first_frame=video_contains_first_frame)
        # pick a random frame for image discriminator       
        
        batch, channels, frames = video.shape[:3]
        
        frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices
        if self.only_last_frame:
            frame_indices[:] = -1
        elif len(self.select_frame) > 0:
            video = torch.stack([video[:, :, idx] for idx in self.select_frame], dim=2)
            frames = len(self.select_frame)
            recon_video = recon_video[:, :, 1:]
            frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices
        real = pick_video_frame(video, frame_indices)

        if self.apply_gradient_penalty:
            real = real.requires_grad_()
        

        fake = pick_video_frame(recon_video, frame_indices)
        # only use one scale
        disc_pred_fake = self.discriminator(fake.detach())
        disc_pred_real = self.discriminator(real)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real,real)
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

    def forward(self,
                data: Tensor,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> SampleList:
        """
        Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.
        :param inputs: Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
        :param data_samples: Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
        :param mode:`mode` is not used in :class:`BaseGAN`.
                Defaults to None.
        :return: SampleList: A list of ``DataSample`` contain generated results.
        """

        #     noise = inputs.get('noise', None)
        #     num_batches = get_valid_num_batches(inputs, data_samples)
        #     noise = self.noise_fn(noise, num_batches=num_batches)
        #     sample_kwargs = inputs.get('sample_kwargs', dict())
        # num_batches = noise.shape[0]
        data = self.data_preprocessor(data, True)
        inputs_dict, data_samples = data['inputs'], data['data_samples']
        data_src = (inputs_dict+1.0)*127.5
        # data_src is RGB
        data_src=data_src.clamp(0, 255).to(torch.uint8)

        sample_model = self._get_valid_model(inputs_dict)
        batch_sample_list = []
        assert sample_model in ['ema','orig'] ," sample model must in ['ema','orig' ]"

        generator=None
        if sample_model == 'ema':
            generator = self.generator_ema
        else:
            generator = self.generator

        video_or_images = inputs_dict
        is_image = video_or_images.ndim == 4
        video_contains_first_frame = True
        video = None
        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True
        else:
            video = video_or_images
        
        # import pdb;pdb.set_trace()
        return_encode_feat = True if (data_samples.get('states', None) is not None or self.test_return_feat) else False
        if return_encode_feat:
            outputs,codes, indice, encode_feat, _ = generator(video, cond= None, video_contains_first_frame=video_contains_first_frame, return_feat=return_encode_feat)
        else:
            outputs,codes,indice = generator(video, cond= None, video_contains_first_frame=video_contains_first_frame)
        
        if is_image:
            outputs = outputs.squeeze(dim=2)
        # outputs is BGR
        outputs = self.data_preprocessor.destruct(outputs, data_samples)
       
        gen_sample = DataSample()
        gen_sample.indice = indice
        gen_sample.fake_img = outputs
        gen_sample.gt_img = data_src[:,[2,1,0]] # keep the gen_sample with the BGR channel order
        gen_sample.sample_model = sample_model
        if return_encode_feat:
            gen_sample.clip_start_end_id = data_samples.get('clip_start_end_id', None)
            gen_sample.type = data_samples.get('type', None)
            gen_sample.capture = data_samples.get('capture', None)
            gen_sample.idx = data_samples.get('idx', None)
            gen_sample.action = data_samples.get('action', None)
            gen_sample.prompt = data_samples.get('prompt', None)
            gen_sample.encode_feat = encode_feat.detach().to(torch.float16).cpu()
        batch_sample_list = gen_sample.split(allow_nonseq_value=True, )
        
        return batch_sample_list



