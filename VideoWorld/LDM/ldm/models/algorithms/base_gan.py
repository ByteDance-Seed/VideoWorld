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
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from ldm.registry import MODELS
from ldm.structures import DataSample
from ldm.utils.typing import ForwardInputs, SampleList
from ..utils import  set_requires_grad

ModelType = Union[Dict, nn.Module]

class BaseGan(BaseModel,metaclass=ABCMeta):
    """Base class for GAN models.

        Args:
            generator (ModelType): The config or model of the generator.
            discriminator (Optional[ModelType]): The config or model of the
                discriminator. Defaults to None.
            data_preprocessor (Optional[Union[dict, Config]]): The pre-process
                config or :class:`~mmagic.models.DataPreprocessor`.
            generator_steps (int): The number of times the generator is completely
                updated before the discriminator is updated. Defaults to 1.
            discriminator_steps (int): The number of times the discriminator is
                completely updated before the generator is updated. Defaults to 1.
            ema_config (Optional[Dict]): The config for generator's exponential
                moving average setting. Defaults to None.
        """


    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None):
        super().__init__(data_preprocessor=data_preprocessor)


        # build generator
        if isinstance(generator,dict):
            self._gen_cfg = deepcopy(generator)
            # build generator with default `noise_size` and `num_classes`
            gen_args = dict()
            generator = MODELS.build(generator, default_args=gen_args)

        self.generator = generator


        # build discriminator
        if discriminator:
            if isinstance(discriminator,dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                discriminator = MODELS.build(
                    discriminator, default_args=disc_args)

        self.discriminator = discriminator

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

        self._init_loss(loss_config)

    @staticmethod
    def gather_log_vars(log_vars_list: List[Dict[str, Tensor]]
                        ) -> Dict[str, Tensor]:
        """Gather a list of log_vars.
        Args:
            log_vars_list: List[Dict[str, Tensor]]

        Returns:
            Dict[str, Tensor]
        """
        if len(log_vars_list) == 1:
            return log_vars_list[0]

        log_keys = log_vars_list[0].keys()

        log_vars = dict()
        for k in log_keys:
            assert all([k in log_vars for log_vars in log_vars_list
                        ]), (f'\'{k}\' not in some of the \'log_vars\'.')
            log_vars[k] = torch.mean(
                torch.stack([log_vars[k] for log_vars in log_vars_list],
                            dim=0))

        return log_vars

    def _init_loss(self, loss_config: Optional[Dict] = None) -> None:
        """Initialize customized loss modules.

        If loss_config is a dict, we allow kinds of value for each field.

        1. `loss_config` is None: Users will implement all loss calculations
            in their own function. Weights for each loss terms are hard coded.
        2. `loss_config` is dict of scalar or string: Users will implement all
            loss calculations and use passed `loss_config` to control the
            weight or behavior of the loss calculation. Users will unpack and
            use each field in this dict by themselves.

            loss_config = dict(gp_norm_mode='HWC', gp_loss_weight=10)

        3. `loss_config` is dict of dict: Each field in `loss_config` will
            used to build a corresponding loss module. And use loss calculation
            function predefined by :class:`BaseGAN` to calculate the loss.

            loss_config = dict()

        Example:
            loss_config = dict(
                # `BaseGAN` pre-defined fields
                gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
                disc_auxiliary_loss=dict(
                    type='R1GradientPenalty',
                    loss_weight=10. / 2.,
                    interval=2,
                    norm_mode='HWC',
                    data_info=dict(
                        real_data='real_imgs',
                        discriminator='disc')),
                gen_auxiliary_loss=dict(
                    type='GeneratorPathRegularizer',
                    loss_weight=2,
                    pl_batch_shrink=2,
                    interval=g_reg_interval,
                    data_info=dict(
                        generator='gen',
                        num_batches='batch_size')),
                # user-defined field for loss weights or loss calculation
                my_loss_2=dict(weight=2, norm_mode='L1'),
                my_loss_3=2,
                my_loss_4_norm_type='L2')


        Args:
            loss_config (Optional[Dict], optional): Loss config used to build
                loss modules or define the loss weights. Defaults to None.
        """
        if loss_config is None:
            self.gan_loss = None
            self.gen_auxiliary_losses = None
            self.disc_auxiliary_losses = None
            self.loss_config = dict()
            return

        self.loss_config = deepcopy(loss_config)

        # build pre-defined losses
        gan_loss = loss_config.get('gan_loss', None)
        if gan_loss is not None:
            self.gan_loss = MODELS.build(gan_loss)
        else:
            self.gan_loss = None

        disc_auxiliary_loss = loss_config.get('disc_auxiliary_loss', None)
        if disc_auxiliary_loss:
            if not isinstance(disc_auxiliary_loss, list):
                disc_auxiliary_loss = [disc_auxiliary_loss]
            self.disc_auxiliary_losses = nn.ModuleList(
                [MODELS.build(loss) for loss in disc_auxiliary_loss])
        else:
            self.disc_auxiliary_losses = None

        gen_auxiliary_loss = loss_config.get('gen_auxiliary_loss', None)
        if gen_auxiliary_loss:
            if not isinstance(gen_auxiliary_loss, list):
                gen_auxiliary_loss = [gen_auxiliary_loss]
            self.gen_auxiliary_losses = nn.ModuleList(
                [MODELS.build(loss) for loss in gen_auxiliary_loss])
        else:
            self.gen_auxiliary_losses = None

    @property
    def generator_steps(self) -> int:
        """int: The number of times the generator is completely updated before
        the discriminator is updated."""
        return self._gen_steps

    @property
    def discriminator_steps(self) -> int:
        """int: The number of times the discriminator is completely updated
        before the generator is updated."""
        return self._disc_steps

    @property
    def device(self) -> torch.device:
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    @property
    def with_ema_gen(self) -> bool:
        """Whether the GAN adopts exponential moving average.

        Returns:
            bool: If `True`, means this GAN model is adopted to exponential
                moving average and vice versa.
        """
        return self._with_ema_gen

    def _init_ema_model(self, ema_config: dict):
        """Initialize a EMA model corresponding to the given `ema_config`. If
        `ema_config` is an empty dict or `None`, EMA model will not be
        initialized.

        Args:
            ema_config (dict): Config to initialize the EMA model.
        """
        ema_config.setdefault('type', 'ExponentialMovingAverage')
        self.ema_start = ema_config.pop('start_iter', 0)
        src_model = self.generator.module if is_model_wrapper(
            self.generator) else self.generator
        self.generator_ema = MODELS.build(
            ema_config, default_args=dict(model=src_model))


    def _get_valid_model(self,batch_inputs: ForwardInputs) -> str:
        """Try to get the valid forward model from inputs.

        - If forward model is defined in `batch_inputs`, it will be used as
          forward model.
        - If forward model is not defined in `batch_inputs`, 'ema' will
          returned if :property:`with_ema_gen` is true. Otherwise, 'orig' will
          be returned.

        Args:
            batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

        Returns:
            str: Forward model to generate image. ('orig', 'ema' or
                'ema/orig').
        """
        if isinstance(batch_inputs, dict):
            sample_model = batch_inputs.get('sample_model', None)
        else:  # batch_inputs is a Tensor
            sample_model = None

            # set default value
        if sample_model is None:
            if self.with_ema_gen:
                sample_model = 'ema'
            else:
                sample_model = 'orig'

            # security checking for mode
        assert sample_model in [
            'ema', 'ema/orig', 'orig'
        ], ('Only support \'ema\', \'ema/orig\', \'orig\' '
            f'in {self.__class__.__name__}\'s image sampling.')
        if sample_model in ['ema', 'ema/orig']:
            assert self.with_ema_gen, (
                f'\'{self.__class__.__name__}\' do not have EMA model.')
        return sample_model


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
        # import pdb;pdb.set_trace()
        outputs,codes,indice = generator(inputs_dict)
        # outputs is BGR
        outputs = self.data_preprocessor.destruct(outputs, data_samples)

        gen_sample = DataSample()
        gen_sample.fake_img = outputs
        gen_sample.gt_img = data_src[:,[2,1,0]] # keep the gen_sample with the BGR channel order
        gen_sample.sample_model = sample_model
        batch_sample_list = gen_sample.split(allow_nonseq_value=True, )

        return batch_sample_list


    def val_step(self,data:dict)->SampleList:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        # inputs_dict, data_samples = data['inputs'], data['data_samples']
        # data = self.data_preprocessor(data)
        outputs = self(data)
        return outputs

    def test_step(self,data:dict)->SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            List[DataSample]: Generated image or image dict.
        """
        # inputs_dict, data_samples = data['inputs'], data['data_samples']
        # data = self.data_preprocessor(data)
        # import pdb;pdb.set_trace()
        outputs = self(data)
        
        # import pdb;pdb.set_trace()
        import cv2
        import numpy as np

        # # NOTE: Image
        # for i in range(len(data['inputs'])):
        #     img = outputs[i].gt_img.permute(1, 2, 0).cpu().numpy()
        #     fake = outputs[i].fake_img.permute(1, 2, 0).cpu().numpy()
        #     # img = img[:, :, ::-1]
        #     # fake = fake[:, :, ::-1]
        #     # fake = outputs[i].fake_img.permute(1, 2, 0).cpu().numpy()
        #     show = np.concatenate((img, fake), axis=0)
        #     # show = np.concatenate((show, fake), axis=0)
        #     cv2.imwrite('/opt/tiger/mmagicinit/visualize/test_{}.jpg'.format(i), show)


        # NOTE: Video 
        # print([outputs[i].indice for i in range(len(data['inputs']))])
        # for i in range(len(data['inputs'])):
        #     img0 = outputs[i].gt_img[:, 0].permute(1, 2, 0).cpu().numpy()
        #     img1 = outputs[i].gt_img[:, 1].permute(1, 2, 0).cpu().numpy()
        #     fake0 = outputs[i].fake_img[:, 0].permute(1, 2, 0).cpu().numpy()
        #     fake1 = outputs[i].fake_img[:, -1].permute(1, 2, 0).cpu().numpy()
        #     # gt = outputs[i].gt_img.permute(1, 2, 0).cpu().numpy()
        #     # fake = outputs[i].fake_img.permute(1, 2, 0).cpu().numpy()
        #     show0 = np.concatenate((img0, img1), axis=0)
        #     show1 = np.concatenate((fake0, fake1), axis=0)
        #     show =  np.concatenate((show0, show1), axis=1)
        #     cv2.imwrite('/opt/tiger/mmagicinit/visualize/test_{}.jpg'.format(i), show)
        
        # NOTE: Video MultiFrame
        # print([outputs[i].indice for i in range(len(data['inputs']))])
        # for i in range(len(data['inputs'])):
        #     real_imgs = outputs[i].gt_img.permute(1, 2, 3, 0)
        #     fake_imgs = outputs[i].fake_img.permute(1, 2, 3, 0)
        #     if len(real_imgs) > len(fake_imgs):
        #         real_imgs = real_imgs[1:]
        #     for fi, real in enumerate(real_imgs):
        #         _real = real.detach().cpu().numpy().astype(np.uint8)
        #         show = _real if fi == 0 else np.concatenate((show, _real))
        #     # for fi in range(len(real_img)):
        #     #     if fi % 2 == 0:
        #     #         _fake = torch.zeros_like(real).detach().cpu().numpy().astype(np.uint8)
        #     #     else:
        #     #         _fake = torch.clamp((fake_imgs[fi//2] + 1)*127, min=0, max=255).detach().cpu().numpy().astype(np.uint8)
        #     #     fake_show = _fake if fi == 0 else np.concatenate((fake_show, _fake))
        #     # CALVIN
        #     for fi, fake in enumerate(fake_imgs):
        #         _fake = fake.cpu().numpy().astype(np.uint8)
        #         fake_show = _fake if fi == 0 else np.concatenate((fake_show, _fake))

        #     show = np.concatenate((show, fake_show), axis=1)
        #     cv2.imwrite(f'/opt/tiger/mmagicinit/visualize/test_{i}.jpg', show[:, :, ::-1])ite(f'/opt/tiger/mmagicinit/visualize/test_{idx}.jpg', show[:, :, ::-1])
        return outputs

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

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(
                **data, optimizer_wrapper=disc_optimizer_wrapper)

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
            #init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)

            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        **data, optimizer_wrapper=gen_optimizer_wrapper)

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

    def _get_gen_loss(self, out_dict):
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            out_dict['disc_pred_fake_g'], target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.gen_auxiliary_losses is not None:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

    def _get_disc_loss(self, out_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake'] = self.gan_loss(
            out_dict['disc_pred_fake'], target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(
            out_dict['disc_pred_real'], target_is_real=True, is_disc=True)

        # disc auxiliary loss
        if self.disc_auxiliary_losses is not None:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

    def train_generator(self, inputs: dict, optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        img = inputs['img']
        num_batches = img.size(0)


        fake_imgs = self.generator(img)
        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_gen_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars

    def train_discriminator(self, inputs: dict, optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs, num_batches = inputs['img'], inputs['img'].shape[0]

        fake_imgs = self.generator(real_imgs)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_disc_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars





