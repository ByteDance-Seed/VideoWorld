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
from videoworld2.pipelines.video2world import *


class Video2WorldPipelineWithCtrl(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.text_encoder: CosmosT5TextEncoder = None
        self.dit: torch.nn.Module = None
        self.dit_ema: torch.nn.Module = None
        self.tokenizer: TokenizerInterface = None
        self.conditioner = None
        self.prompt_refiner = None
        self.text_guardrail_runner = None
        self.video_guardrail_runner = None
        self.model_names = ["text_encoder", "dit", "tokenizer"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False

    @staticmethod
    def from_config(
        config: Video2WorldWithCtrlPipelineConfig,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # import pdbc;pdb.set_trace()
        # Create a pipe
        pipe = Video2WorldPipelineWithCtrl(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        if text_encoder_path:
            # inference
            pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
            pipe.text_encoder.to(device)
        else:
            # training
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"


        pipe.text_guardrail_runner = None
        pipe.video_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        with init_weights_on_device():
            dit_config = config.net
            dit_ctrl_config = config.net_ctrl
            pipe.dit = instantiate(dit_config).eval()  # inference
            pipe.dit_ctrl = instantiate(dit_ctrl_config).eval()  # inference

        # import pdb;pdb.set_trace()
        if dit_path:
            state_dict = load_state_dict(dit_path)
        # drop net. prefix
        state_dict_dit_compatible = dict()
        state_dict_dit_ctrl_compatible = dict()
        has_pretrained_ctrl = False
        for k, v in state_dict.items():
            if 'ldm_net' in k:
                continue
            if k.startswith("net."):
                state_dict_dit_compatible[k[4:]] = v
            elif k.startswith("net_ctrl."):
                has_pretrained_ctrl = True
                state_dict_dit_ctrl_compatible[k[9:]] = v
            else:
                state_dict_dit_compatible[k] = v
                state_dict_dit_ctrl_compatible[k] = v

        pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
        if has_pretrained_ctrl:
            pipe.dit_ctrl.load_state_dict(state_dict_dit_ctrl_compatible, strict=False, assign=True)
        else:
            pipe.dit_ctrl.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)

        del state_dict, state_dict_dit_compatible
        log.success(f"Successfully loaded DiT from {dit_path}")

        # 6-2. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        # for n, p in pipe.dit.named_parameters():
        #     print(n, p.device)
        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        pipe.dit_ctrl = pipe.dit_ctrl.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe
    
    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: torch.Tensor,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.dit.enable_context_parallel(cp_group)
            self.dit_ctrl.enable_context_parallel(cp_group)
        else:
            self.dit.disable_context_parallel()
            self.dit_ctrl.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
    
    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: T2VCondition, use_cuda_graphs: bool = False
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (T2VCondition): conditional information, generated from self.conditioner
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        # import pdb;pdb.set_trace()
        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            if self.config.conditioning_strategy == str(ConditioningStrategy.FRAME_REPLACE):
                # In case of frame replacement strategy, replace the first few frames of the video with the conditional frames
                # Update the c_noise as the conditional frames are clean and have very low noise

                # Make the first few frames of x_t be the ground truth frames
                net_state_in_B_C_T_H_W = (
                    condition_state_in_B_C_T_H_W * condition_video_mask
                    + net_state_in_B_C_T_H_W * (1 - condition_video_mask)
                )

                # Adjust c_noise for the conditional frames
                sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
                _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )
            elif self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                # In case of channel concatenation strategy, concatenate the conditional frames in the channel dimension
                condition_state_in_masked_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask
                net_state_in_B_C_T_H_W = torch.cat([net_state_in_B_C_T_H_W, condition_state_in_masked_B_C_T_H_W], dim=1)

        else:
            # In case of image batch, simply concatenate the 0 frames when channel concat strategy is used
            if self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                net_state_in_B_C_T_H_W = torch.cat(
                    [net_state_in_B_C_T_H_W, torch.zeros_like(net_state_in_B_C_T_H_W)], dim=1
                )

        x_ctrl = self.dit_ctrl(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
            use_cuda_graphs=use_cuda_graphs,
            hint_key=self.config.hint_key
        )
        # forward pass through the network
        net_output_B_C_T_H_W = self.dit(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
            use_cuda_graphs=use_cuda_graphs,
            x_ctrl=x_ctrl
        ).float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.dit.fully_shard(mesh=dp_mesh)
        self.dit = fully_shard(self.dit, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.dit, dp_mesh)
        if self.dit_ema:
            self.dit_ema.fully_shard(mesh=dp_mesh)
            self.dit_ema = fully_shard(self.dit_ema, mesh=dp_mesh, reshard_after_forward=True)
            broadcast_dtensor_model_states(self.dit_ema, dp_mesh)
            self.dit_ema_worker = DTensorFastEmaModelUpdater()

        self.dit_ctrl.fully_shard(mesh=dp_mesh)
        self.dit_ctrl = fully_shard(self.dit_ctrl, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.dit_ctrl, dp_mesh)