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
from functools import cache
import collections
import math
from typing import Any, Dict, Mapping, Optional, Tuple
import copy
import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
import numpy as np
from videoworld2.conditioner import DataType, T2VCondition
from videoworld2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B, PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B, Video2WorldPipelineConfig, Video2WorldWithCtrlPipelineConfig
from videoworld2.configs.base.config_ldm import LDM_DEFAULT, LatentDynamicModelConfig
import imageio
from videoworld2.networks.model_weights_stats import WeightTrainingStat
from videoworld2.pipelines.video2world import Video2WorldPipeline
from videoworld2.pipelines.video2world_ctrl import Video2WorldPipelineWithCtrl
from videoworld2.utils.checkpointer import non_strict_load_model
from videoworld2.utils.optim_instantiate import get_base_scheduler
from videoworld2.utils.torch_future import clip_grad_norm_
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video
import os
from videoworld2.models.utils import GeometricAugmentor_K, compute_lpips, compute_psnr, compute_ssim
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from lpips import LPIPS
import torchvision.transforms as T
import glob
import random
IMAGE_KEY = "images"
VIDEO_KEY = "video"
RECON_KEY = "reconstructions"
LATENT_KEY = "latent"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"
LDM_H, LDM_W = 240, 416

from decord import VideoReader, cpu
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms.v2 import UniformTemporalSubsample
from videoworld2.data.dataset_utils import Resize_Preprocess, ToTensorVideo

def _load_video(video_path):
    # import pdb;pdb.set_trace()
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    # frame_ids = np.linspace(0, len(vr) - 1).astype(np.int32)
    frame_ids = np.arange(0, len(vr) - 1).tolist()
    vr.seek(0)
    frame_data = vr.get_batch(frame_ids).asnumpy()
    try:
        fps = vr.get_avg_fps()
    except Exception:  # failed to read FPS
        fps = 24
    return frame_data, fps

def _get_frames(video_path):
    preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess((480, 832))])
    frames, fps = _load_video(video_path)
    frames = frames.astype(np.uint8)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
    frames = UniformTemporalSubsample(93)(frames)
    frames = preprocess(frames)
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
    return frames, fps



@attrs.define(slots=False)
class Predict2ModelManagerConfig:
    # Local path, use it in fast debug run
    dit_path: str = "checkpoints/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt"
    # For inference
    text_encoder_path: str = ""  # not used in training.


@attrs.define(slots=False)
class Predict2Video2WorldModelConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True

    precision: str = "bfloat16"
    input_data_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()
    # This is a new way to load models
    pipe_config: Video2WorldPipelineConfig = PREDICT2_VIDEO2WORLD_PIPELINE_2B
    pipe_type: str = 'Video2WorldPipeline'
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0
    ldm_config: LatentDynamicModelConfig = LDM_DEFAULT
    dldm_warmup: bool = False
    only_dit: bool = False
    geometric_aug_on: bool = False
    ldm_act_embedding: bool = False
    use_cross_embedding: bool = False

@attrs.define(slots=False)
class Predict2Video2WorldModelWithCtrlConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True

    precision: str = "bfloat16"
    input_data_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()
    # This is a new way to load models
    pipe_config: Video2WorldWithCtrlPipelineConfig = PREDICT2_VIDEO2WORLD_PIPELINE_CTRL_2B
    pipe_type: str = 'Video2WorldPipelineWithCtrl'
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0
    ldm_config: LatentDynamicModelConfig = LDM_DEFAULT
    dldm_warmup: bool = False
    only_dit: bool = False
    geometric_aug_on: bool = False
    ldm_act_embedding: bool = False
    use_cross_embedding: bool = True
    finetune_base_model: bool = False
    control_requires_grad: bool = False

class TokenizerModel(ImaginaireModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_latent_dynamic_model()

    def init_latent_dynamic_model(self):
        self.network = instantiate(self.config.network).to(dtype=torch.bfloat16).cuda()
        self.loss = instantiate(self.config.loss).to(dtype=torch.bfloat16).cuda()
        self.metric = instantiate(self.config.metric).to(dtype=torch.bfloat16).cuda()
        # import pdb;pdb.set_trace()
        if self.config.ldm_path != "":
            log.info(f"- Loading the LDM Model, load_path: {self.config.ldm_path}...")
            ldm_state_dict = torch.load(self.config.ldm_path, map_location=lambda storage, loc: storage, weights_only=False)
            if ldm_state_dict.get('model', None):
                log.info(self.load_state_dict(ldm_state_dict['model'], strict=False))
            else:
                log.info(self.load_state_dict(ldm_state_dict, strict=False))



class Predict2Video2WorldModel(ImaginaireModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1
        # New way to init pipe
        use_ctrl = (config.pipe_type == 'Video2WorldPipelineWithCtrl')
        if not use_ctrl:
            self.pipe = Video2WorldPipeline.from_config(
                config.pipe_config,
                dit_path=config.model_manager_config.dit_path,
            )
        else:
            self.pipe = Video2WorldPipelineWithCtrl.from_config(
                config.pipe_config,
                dit_path=config.model_manager_config.dit_path,
            )
        self.use_ctrl = use_ctrl
        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        if use_ctrl and not self.config.finetune_base_model:
            self.pipe.dit.requires_grad_(False)
            self.pipe.dit_ctrl.train()
            self.pipe.dit_ctrl.requires_grad_(True)
            dit_ctrl_total_params = sum(p.numel() for p in self.pipe.dit_ctrl.parameters() if p.requires_grad)
            log.info(f"DiT CTRL total parameters: {dit_ctrl_total_params / 1e9:.2f}B")
        elif use_ctrl and self.config.finetune_base_model:
            self.pipe.dit.requires_grad_(True)
            self.pipe.dit_ctrl.train()
            self.pipe.dit_ctrl.requires_grad_(True)
            dit_ctrl_total_params = sum(p.numel() for p in self.pipe.dit_ctrl.parameters() if p.requires_grad)
            dit_total_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
            log.info(f"DiT total parameters: {dit_total_params / 1e9:.2f}B")
            log.info(f"DiT CTRL total parameters: {dit_ctrl_total_params / 1e9:.2f}B")
        
        dit_total_params = sum(p.numel() for p in self.pipe.dit.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.ldm_net = TokenizerModel(self.config.ldm_config)
  

        ldm_trainable_params = sum(p.numel() for p in self.ldm_net.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(f"DIT total parameters: {dit_total_params / 1e9:.2f}B, Frozen parameters: {frozen_params / 1e9:.2f}B, DiT Trainable parameters: {trainable_params / 1e9:.2f}B, LDM Trainable parameters: {ldm_trainable_params / 1e9:.2f}B")
        
        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
            self.ldm_net.network = FSDP(self.ldm_net.network, sharding_strategy=ShardingStrategy.NO_SHARD)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")
        self.validation_results = []
        self.dldm_warmup = config.dldm_warmup
        self.only_dit = config.only_dit
        self.ldm_act_embedding = False
        self.geometric_aug_on = config.geometric_aug_on
        self.geometric_aug = GeometricAugmentor_K()
        self.color_aug = GeometricAugmentor_K(geo_on=False)
        self.grayscale_transform = T.Grayscale(num_output_channels=3)
        self.control_requires_grad = False if not use_ctrl else config.control_requires_grad
        self.use_cross_embedding = config.use_cross_embedding
    @property
    def net(self) -> torch.nn.Module:
        return self.pipe.dit

    # New function, added for i4 adaption
    @property
    def net_ema(self) -> torch.nn.Module:
        return self.pipe.dit_ema

    # New function, added for i4 adaption
    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.pipe_config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.pipe.dit_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # New function, added for i4 adaption
    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.pipe_config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        for module in [self.net, self.pipe.tokenizer, self.ldm_net]:
            if module is not None:
                module.to(memory_format=memory_format, **self.tensor_kwargs)


    def freeze_parameters(self) -> None:
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
    ):
        from peft import LoraConfig, inject_adapter_in_model

        # Add LoRA to UNet
        self.lora_alpha = lora_alpha

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        # for param in model.parameters():
        #     # Upcast LoRA parameters into fp32
        #     if param.requires_grad:
        #         param.data = param.to(torch.float32)
        # import pdb;pdb.set_trace()
        # pth = torch.load("checkpoints/posttraining/video2world/2b_openx_lora_onlydit_f93_fps1to4_all_1014/checkpoints/model/iter_000033000.pt")
        # new_pth = {}
        # for key, val in pth.items():
        #     if "net." in key:
        #         new_pth[key.replace('net.', '')] = val
        # peft_model.model.load_state_dict(new_pth, strict=False)
        # peft_model = peft_model.merge_and_unload()
        # save_pth = {}
        # for key, val in peft_model.state_dict().items():
        #     save_pth['net.'+key] = val
        # torch.save(save_pth, "checkpoints/posttraining/video2world/2b_openx_lora_onlydit_f93_fps1to4_all_1014/checkpoints/model/iter_000033000_merged.pt")
    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert (
            is_image != is_video
        ), "Only one of the input_image_key or input_data_key should be present in the data_batch."
        return is_image

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_data_key
        if isinstance(self.pipe.dit, WeightTrainingStat):
            if is_image:
                self.pipe.dit.accum_image_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size
            else:
                self.pipe.dit.accum_video_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size

    def draw_training_sigma_and_epsilon(self, x0_size: torch.Size, condition: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        if is_video_batch and self.config.high_sigma_ratio > 0:
            # Implement the high sigma strategy LOGUNIFORM200_100000
            LOG_200 = math.log(200)
            LOG_100000 = math.log(100000)
            mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
            log_new_sigma = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                + LOG_200
            )
            sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: T2VCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        data_batch_idx: int,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # import pdb;pdb.set_trace()
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2

        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        output_batch["loss"] = kendall_loss.mean()  # check if this is what we want

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def gray_transform(self, video):
        ndim = video.ndim
        video = video[0].permute(1, 0, 2, 3) if ndim == 5 else video.permute(1, 0, 2, 3)
        video_gray = self.grayscale_transform(video)
        return video_gray.permute(1, 0, 2, 3)[None]

    def geo_transform(self, video, gray_on=False):
        ndim = video.ndim
        video = video[0].permute(1, 0, 2, 3) if ndim == 5 else video.permute(1, 0, 2, 3)
        video = video.float() / 255 #data_batch['video']: B, 3, T, H, W
        video_aug = (self.geometric_aug(video) * 255).to(torch.uint8)
        video_aug = video_aug.permute(1, 0, 2, 3)[None]
        if gray_on and random.random() > 0.5:
            video_aug = self.gray_transform(video_aug)
        return video_aug
    
    def color_transform(self, video):
        ndim = video.ndim
        video = video[0].permute(1, 0, 2, 3) if ndim == 5 else video.permute(1, 0, 2, 3)
        video = video.float() / 255 #data_batch['video']: B, 3, T, H, W
        video_aug = (self.color_aug(video) * 255).to(torch.uint8)
        video_aug = video_aug.permute(1, 0, 2, 3)[None]
        return video_aug
    
    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        rank = torch.distributed.get_rank()
        use_ctrl = self.use_ctrl
        self._update_train_stats(data_batch)
        b, _, t, h, w = data_batch['video'].shape
        hint_key = self.config.pipe_config.hint_key if use_ctrl else "video"
       
        # NOTE: Aug Video Color For DiT Ctrl
        # show = data_batch["recon_image"][0].permute(1, 2, 3, 0).cpu().numpy()
        # imageio.mimsave('/opt/tiger/test.mp4', show, fps=30)
        data_batch['t5_text_embeddings'] = torch.zeros_like(data_batch['t5_text_embeddings'])
        ldm_loss_value = 0
        if self.dldm_warmup:
            self.pipe._normalize_video_databatch_inplace(data_batch)
            dldm_data_batch = copy.deepcopy(data_batch)
            dldm_data_batch['video'] = torch.nn.functional.interpolate(dldm_data_batch['video'].permute(0, 2, 1, 3, 4).flatten(0, 1), size=(LDM_H, LDM_W), mode='bilinear', align_corners=False).view(b, t, 3, LDM_H, LDM_W).permute(0, 2, 1, 3, 4)            
            ldm_output_batch, ldm_recon_loss, crossattn_embed = self.ldm_training_step(dldm_data_batch, data_batch_idx)
            return ldm_output_batch, ldm_recon_loss
        elif self.only_dit:
            output_batch, kendall_loss = self.pipe_training_step(data_batch, data_batch_idx)
            return output_batch, kendall_loss
        else:
            dldm_tensor_batch = torch.nn.functional.interpolate(((data_batch["video"] / 127.5) - 1).permute(0, 2, 1, 3, 4).flatten(0, 1), size=(LDM_H, LDM_W), mode='bilinear', align_corners=False).view(b, t, 3, LDM_H, LDM_W).permute(0, 2, 1, 3, 4).to(torch.bfloat16)
            encoded_video, _, (quant_codes, _), _ = self.ldm_net.network.encode(dldm_tensor_batch)
            recon_image = self.ldm_net.network.decode(encoded_video, quant_codes.detach()) 
            ori_recon_image = recon_image
            recon_image = torch.nn.functional.interpolate(recon_image[0].permute(1, 0, 2, 3), size=(data_batch['video'].shape[-2], data_batch['video'].shape[-1]), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)[None]
            recon_image = (((recon_image[0] * 0.5) + 0.5).clamp(0, 1)) * 255
            recon_image = recon_image[None] if recon_image.ndim == 4 else recon_image
            gray_video = self.gray_transform(recon_image)

            assert self.use_cross_embedding
            quant = self.ldm_net.network.post_quant_conv(quant_codes)
            quant = self.ldm_net.network.quant_to_dit_dim(quant).flatten(2)
            quant = quant.permute(0, 2, 1)
            crossattn_embed = torch.cat([quant, torch.zeros((1, 512 - quant.shape[1], 1024)).to(quant)], dim=1)
            data_batch['t5_text_embeddings'] = crossattn_embed

            inputs = {INPUT_KEY: dldm_tensor_batch, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(dldm_tensor_batch))}
    
            _, ldm_loss_value = self.ldm_net.loss(inputs, {"reconstructions": ori_recon_image}, iteration=0)
            
            data_batch['video'] = self.color_transform(data_batch["video"])
            gray_video = (gray_video.to(torch.bfloat16) / 127.5) - 1
            gray_video = gray_video.detach()
            self.pipe._normalize_video_databatch_inplace(data_batch)

            data_batch['recon_image'] = gray_video
            output_batch, kendall_loss = self.pipe_training_step(data_batch, data_batch_idx)
            kendall_loss = kendall_loss + ldm_loss_value
            return output_batch, kendall_loss      
        

    def ldm_training_step(self, data_batch, data_batch_idx):
        # import pdb;pdb.set_trace()
        if data_batch.get('video_aug_0', None) is not None:
            tensor_batch_aug_0 = data_batch['video_aug_0'][:, :, :1]
            tensor_batch_aug_1 = data_batch['video_aug_1']

            output_dict = self.ldm_net.network(tensor_batch_aug_0, tensor_batch_aug_1)

        else:
            tensor_batch = data_batch["video"]
            output_dict = self.ldm_net.network(tensor_batch)

        input_images, recon_image, mid_h = data_batch["video"], output_dict["reconstructions"], output_dict["mid_h"]
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}
        # import pdb;pdb.set_trace()
        loss_dict, loss_value = self.ldm_net.loss(inputs, output_dict, iteration=0)
        quant_codes = output_dict['quant_codes']
        quant = self.ldm_net.network.post_quant_conv(quant_codes)
        if self.ldm_net.network.connector_type == 'llama':
            quant = self.ldm_net.network.quant_to_dit_dim(quant)
        else:
            quant = self.ldm_net.network.quant_to_dit_dim(quant).flatten(2)
            quant = quant.permute(0, 2, 1)

        return dict({"prediction": recon_image, "mid_h": mid_h, "loss": loss_value}), loss_value, quant

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # import pdb;pdb.set_trace()
        rank = torch.distributed.get_rank()
        local_dir = data_batch['local_dir']
        val_iter = data_batch['val_iter']
        del data_batch['local_dir']
        del data_batch['val_iter']
        for name in data_batch.keys():
            data_batch[name] = data_batch[name][0] if isinstance(data_batch[name], torch.Tensor) and data_batch[name].ndim==6 else data_batch[name]

        self.pipe._normalize_video_databatch_inplace(data_batch)
        batch_size = data_batch['video'].shape[0]
        video_name = '_'.join(data_batch['video_name']['video_path'][0].split('/')[-1].split('_')[:-1])
        video = None
        output_video = []
        quant_infos = []

        for bz in range(batch_size):
            input_video = data_batch['video'][bz][None]

            ldm_tensor_batch = input_video
            ldm_tensor_batch = torch.nn.functional.interpolate(ldm_tensor_batch[0].permute(1, 0, 2, 3), size=(LDM_H, LDM_W), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)[None]
            if bz == 0:
                encoded_video, quant_info, (quant_codes, _), _ = self.ldm_net.network.encode(ldm_tensor_batch)
                recon_image = self.ldm_net.network.decode(encoded_video, quant_codes) 

            else:
                encoded_video_ori, quant_info, (quant_codes, _), _ = self.ldm_net.network.encode(ldm_tensor_batch)
                encoded_video, _ = self.ldm_net.network.encoder(last_frame)
                recon_image = self.ldm_net.network.decode(encoded_video, quant_codes) 
                
                # _recon_image = self.ldm_net.network.decode(encoded_video_ori, quant_codes) 
                # _recon_image = torch.nn.functional.interpolate(_recon_image[0].permute(1, 0, 2, 3), size=(data_batch['video'].shape[-2], data_batch['video'].shape[-1]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                # _recon_image = (((_recon_image * 0.5) + 0.5).clamp(0, 1))
                # _recon_image = (_recon_image * 255).to(torch.uint8).cpu().numpy()
                # imageio.mimsave(os.path.join(local_dir, f"{video_name}_{bz}.mp4"), _recon_image, fps=30)

            last_frame = recon_image[:, :, -1:]
            output_video.append(recon_image[:, :, :-1])
            quant_infos.append(quant_info.flatten().cpu())

       
        # all_input_video = data_batch['video'].permute(1, 0, 2, 3, 4).flatten(1, 2)
        # all_input_video = (((all_input_video * 0.5) + 0.5).clamp(0, 1)).permute(1, 2, 3, 0)
        # all_input_video = (all_input_video * 255).to(torch.uint8).cpu().numpy()
        # imageio.mimsave(os.path.join(local_dir, f"{video_name}_ori.mp4"), all_input_video, fps=30)

        output_video = torch.cat(output_video, dim=2)
        output_video = torch.nn.functional.interpolate(output_video[0].permute(1, 0, 2, 3), size=(data_batch['video'].shape[-2], data_batch['video'].shape[-1]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        output_video = (((output_video * 0.5) + 0.5).clamp(0, 1))
        output_video = (output_video * 255).to(torch.uint8).cpu().numpy()
        imageio.mimsave(os.path.join(local_dir, f"{video_name}.mp4"), output_video, fps=30)
        quant_infos = torch.stack(quant_infos, dim=0)
        out = {
            'global_step': iteration,
            'latent_codes': quant_infos,
            'video_path': video_name,
            'local_dir': local_dir,
        } 
        self.validation_results.append(out)
        return {"result": recon_image}, torch.tensor([0]).to('cuda').to(torch.bfloat16)


        


    def pipe_training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        # import pdb;pdb.set_trace()
        self.pipe.device = self.device

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        

        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T, data_batch_idx
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> Dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        # convert DTensor to Tensor
        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()

        ldm_state_dict = self.ldm_net.state_dict(prefix="ldm_net.")
        # convert DTensor to Tensor
        for key, val in ldm_state_dict.items():
            if 'loss.' in key:
                continue
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()
        if self.use_ctrl:
            net_ctrl_state_dict = self.pipe.dit_ctrl.state_dict(prefix="net_ctrl.")
            # convert DTensor to Tensor
            for key, val in net_ctrl_state_dict.items():
                if isinstance(val, DTensor):
                    # Convert to full tensor
                    net_state_dict[key] = val.full_tensor().detach().cpu()
                else:
                    net_state_dict[key] = val.detach().cpu()

        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        # NOTE: DEBUG
        # dp_mesh = self.pipe.dit.blocks[0]._checkpoint_wrapped_module.self_attn.k_norm.weight.device_mesh
        # for n, p in self.pipe.dit.named_parameters():
        #     if isinstance(p, torch.distributed.tensor.DTensor):
        #         _reg_state_dict[n] = dist_tensor.DTensor.from_local(v, dp_mesh)
        
        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.pipe.dit.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )

            if self.config.pipe_config.ema.enabled:
                ema_results: _IncompatibleKeys = self.pipe.dit_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.pipe_config.ema.enabled else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.pipe_config.ema.enabled else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.pipe.dit, _reg_state_dict), rank0_only=False)
            if self.config.pipe_config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.pipe.dit_ema, _ema_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.pipe_config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.pipe.ema_exp_coefficient + 1)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> torch.Tensor:
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
    
class Predict2Video2WorldModelWithCtrl(Predict2Video2WorldModel):
    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        data_batch['use_ctrl'] = True
        return super().training_step(data_batch, data_batch_idx)

    @torch.no_grad()
    def inference(self, frame, latent_dynamic_codes):
        # import pdb;pdb.set_trace()
        #latent_dynamic_codes: [N, L]
        # frames: [1, 3, 1, H, W]
        clip_num = latent_dynamic_codes.shape[0]
        dldm_query_length = 4
        temporal_downsampling = 4
        clip_length = (latent_dynamic_codes.shape[1] // dldm_query_length) * temporal_downsampling  + 1
        
        frame = ((frame / 127.5) - 1).to(torch.bfloat16)  # [1, 3, 1, H, W]
        last_frame = torch.nn.functional.interpolate(frame[0].permute(1, 0, 2, 3), size=(LDM_H, LDM_W), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)[None]
        dit_input = frame.repeat(1, 1, clip_length, 1, 1)
        dldm_input = last_frame
        decoder_outputs = []
        dit_outputs = []
        for ci in range(clip_num):
            # dLDM Decoder
            quant_codes = self.ldm_net.network.quantizer.indices_to_codes(latent_dynamic_codes[ci].view(1, len(latent_dynamic_codes[ci]), 1, 1).to('cuda'))
            encoded_video, _ = self.ldm_net.network.encoder(dldm_input)
            decoder_output = self.ldm_net.network.decode(encoded_video, quant_codes) 

            dldm_input = decoder_output[:, :, -1:]
            decoder_outputs.append(decoder_output[:, :, :-1])
            
            recon_image = torch.nn.functional.interpolate(decoder_output[0].permute(1, 0, 2, 3), size=(frame.shape[-2], frame.shape[-1]), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)[None]
            recon_image = (((recon_image[0] * 0.5) + 0.5).clamp(0, 1))
            recon_image = recon_image * 255

            #For Visualization
            # visualization =  recon_image.permute(1, 2, 3, 0).detach().cpu().float().numpy().astype(np.uint8)
            # imageio.mimsave('recon_image.mp4', visualization, fps=30)

            gray_video = self.gray_transform(recon_image)
            gray_video = (gray_video.to(torch.bfloat16) / 127.5) - 1


            quant = self.ldm_net.network.post_quant_conv(quant_codes)
            quant = self.ldm_net.network.quant_to_dit_dim(quant).flatten(2)
            quant = quant.permute(0, 2, 1)
            crossattn_embed = torch.cat([quant, torch.zeros((1, 512 - quant.shape[1], 1024)).to(quant)], dim=1)
        
            video = self.pipe.validation(
                video=dit_input,
                prompt_embedding=crossattn_embed,
                num_conditional_frames=1,
                guidance=7.0,
                seed=0,
                use_cuda_graphs=False,
                actions=None,
                hint=gray_video,
                hint_key='recon_image',
                initial_noise=None,
            )
    
            dit_input = video[:, :, -1:].repeat(1, 1, video.shape[2], 1, 1)
            dit_outputs.append(video[:, :, :-1])
            save_image_or_video(video, f"infer_output/dit_output_clip_{ci}.mp4", fps=16)

        output_video = torch.cat(dit_outputs, dim=2)
        save_image_or_video(output_video, "infer_output/dit_output_clip.mp4", fps=16)

        