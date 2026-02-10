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
import math
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from einops import rearrange
from megatron.core import parallel_state
from tqdm import tqdm
from PIL import Image
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard


from videoworld2.auxiliary.text_encoder import CosmosT5TextEncoder
from videoworld2.conditioner import DataType, T2VCondition
from videoworld2.configs.base.config_video2world import ConditioningStrategy, Video2WorldPipelineConfig, Video2WorldWithCtrlPipelineConfig
from videoworld2.datasets.utils import VIDEO_RES_SIZE_INFO
from videoworld2.models.utils import init_weights_on_device, load_state_dict
from videoworld2.module.denoise_prediction import DenoisePrediction
from videoworld2.module.denoiser_scaling import RectifiedFlowScaling
from videoworld2.pipelines.base import BasePipeline
from videoworld2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from videoworld2.tokenizers.tokenizer import TokenizerInterface
from videoworld2.utils.context_parallel import broadcast, broadcast_split_tensor, cat_outputs_cp, split_inputs_cp
from videoworld2.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
import pickle
IS_PREPROCESSED_KEY = "is_preprocessed"
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


def resize_input(video: torch.Tensor, resolution: list[int]) -> torch.Tensor:
    r"""
    Resizes and crops the input video tensor while preserving aspect ratio.

    The video is first resized so that the smaller dimension matches the target resolution,
    preserving the aspect ratio. Then, it's center-cropped to the target resolution.

    Args:
        video (torch.Tensor): Input video tensor of shape (T, C, H, W).
        resolution (list[int]): Target resolution [H, W].

    Returns:
        torch.Tensor: Resized and cropped video tensor of shape (T, C, target_H, target_W).
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, resolution)
    return video_cropped


def read_and_process_image(
    img_path: str, resolution: list[int], num_video_frames: int, resize: bool = True
) -> torch.Tensor:
    """
    Reads an image, converts it to a video tensor, and processes it for model input.

    The image is loaded, converted to a tensor, and replicated to match the
    `num_video_frames`. It's then optionally resized and permuted to the
    standard video format (B, C, T, H, W).

    Args:
        img_path (str): Path to the input image file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): The number of frames the output video tensor should have.
        resize (bool, optional): Whether to resize the image to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W).

    Raises:
        ValueError: If the image extension is not one of the supported types.
    """
    ext = os.path.splitext(img_path)[1]
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {ext}")

    # Read the image and convert to tensor
    img = Image.open(img_path)
    img = torchvision.transforms.functional.to_tensor(img)

    # Resize the single image first if needed (more efficient than resizing full tensor later)
    if resize:
        orig_h, orig_w = img.shape[1], img.shape[2]
        target_h, target_w = resolution

        # Calculate scaling based on aspect ratio
        scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
        resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))

        # Resize and crop the single image
        img_resized = torchvision.transforms.functional.resize(img.unsqueeze(0), resizing_shape).squeeze(0)
        img = torchvision.transforms.functional.center_crop(img_resized, resolution)

    # Get final dimensions
    C, H, W = img.shape

    # Pre-allocate tensor with target dimensions and directly in uint8 format
    # This avoids allocating a large float tensor that would be converted later
    vid_input = torch.zeros((num_video_frames, C, H, W), dtype=torch.uint8, device=img.device)

    # Set first frame and convert to uint8 (only the first frame needs conversion)
    vid_input[0] = (img * 255.0).to(torch.uint8)

    # Add batch dimension and permute in one step
    # [T, C, H, W] -> [1, C, T, H, W]
    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return vid_input


def read_and_process_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    # Calculate how many frames to extract from input video
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    log.info(f"Will extract the last {frames_to_extract} frames from input video and pad to {num_video_frames}")

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}")

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least {frames_to_extract} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Extract the last frames_to_extract from input video
    start_idx = available_frames - frames_to_extract
    # import pdb;pdb.set_trace()
    # extracted_frames = video_tensor[:, start_idx:, :, :]  # (C, frames_to_extract, H, W)

    extracted_frames = video_tensor[:, :1, :, :]  # (C, frames_to_extract, H, W)
    

    # Convert to (frames_to_extract, C, H, W) for resize
    extracted_frames = extracted_frames.permute(1, 0, 2, 3)  # (frames_to_extract, C, H, W)

    # Get last frame for padding
    last_frame = extracted_frames[-1]  # (C, H, W)

    # Resize the extracted frames if needed (more efficient than resizing full tensor later)
    if resize:
        C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]
        target_h, target_w = resolution

        # Calculate scaling based on aspect ratio
        scaling_ratio = max((target_w / W), (target_h / H))
        resizing_shape = (int(math.ceil(scaling_ratio * H)), int(math.ceil(scaling_ratio * W)))

        # Resize and crop the extracted frames
        extracted_frames = torchvision.transforms.functional.resize(extracted_frames, resizing_shape)
        extracted_frames = torchvision.transforms.functional.center_crop(extracted_frames, resolution)

        # Resize and crop the last frame separately (for padding)
        last_frame = last_frame.unsqueeze(0)  # Add batch dim for resize
        last_frame = torchvision.transforms.functional.resize(last_frame, resizing_shape)
        last_frame = torchvision.transforms.functional.center_crop(last_frame, resolution)
        last_frame = last_frame.squeeze(0)  # Remove batch dim

    # Get final dimensions
    C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]

    # Pre-allocate tensor with target dimensions and directly in uint8 format
    # This avoids allocating a large float tensor that would be converted later
    full_video = torch.zeros((num_video_frames, C, H, W), dtype=torch.uint8, device=extracted_frames.device)

    # Set extracted frames and convert to uint8
    full_video[:frames_to_extract] = (extracted_frames * 255.0).to(torch.uint8)

    # Pad remaining frames with the last frame (already resized if needed)
    if frames_to_extract < num_video_frames:
        last_frame_uint8 = (last_frame * 255.0).to(torch.uint8)
        for i in range(num_video_frames - frames_to_extract):
            full_video[frames_to_extract + i] = last_frame_uint8

    # Add batch dimension and permute in one operation to final format
    # [T, C, H, W] -> [1, C, T, H, W]
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return full_video


class Video2WorldPipeline(BasePipeline):
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
        config: Video2WorldPipelineConfig,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = Video2WorldPipeline(device=device, torch_dtype=torch_dtype)
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

        # if load_prompt_refiner:
        #     pipe.prompt_refiner = CosmosReason1(
        #         checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
        #         offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
        #         enabled=config.prompt_refiner_config.enabled,
        #     )

        # if config.guardrail_config.enabled:
        #     from videoworld2.auxiliary.guardrail.common import presets as guardrail_presets

        #     pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
        #         config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
        #     )
        #     pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
        #         config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
        #     )
        # else:
        pipe.text_guardrail_runner = None
        pipe.video_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        with init_weights_on_device():
            dit_config = config.net
            pipe.dit = instantiate(dit_config).eval()  # inference

   
        if dit_path:
            state_dict = load_state_dict(dit_path)
        # drop net. prefix
        state_dict_dit_compatible = dict()
        for k, v in state_dict.items():
            if 'ldm_net' in k:
                continue
            if k.startswith("net."):
                state_dict_dit_compatible[k[4:]] = v
            else:
                state_dict_dit_compatible[k] = v

        pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
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
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.dit.fully_shard(mesh=dp_mesh)
        self.dit = fully_shard(self.dit, mesh=dp_mesh, reshard_after_forward=True)
        broadcast_dtensor_model_states(self.dit, dp_mesh)
        if self.dit_ema:
            self.dit_ema.fully_shard(mesh=dp_mesh)
            self.dit_ema = fully_shard(self.dit_ema, mesh=dp_mesh, reshard_after_forward=True)
            broadcast_dtensor_model_states(self.dit_ema, dp_mesh)
            self.dit_ema_worker = DTensorFastEmaModelUpdater()
            # No need to copy weights to EMA when applying FSDP, it is already copied before applying FSDP.

    def apply_cp(self) -> None:
        pass

    def _get_data_batch_input(
        self, video: torch.Tensor, prompt: str, negative_prompt: str = "", num_latent_conditional_frames: int = 1
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        self.batch_size = 1
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": self.encode_prompt(prompt).to(dtype=self.torch_dtype),
            "fps": torch.randint(16, 32, (self.batch_size,)),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_latent_conditional_frames,  # Specify number of conditional frames
        }

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    def denoising_model(self) -> torch.nn.Module:
        return self.dit

    def encode_prompt(
        self, prompts: Union[str, List[str]], max_length: int = 512, return_mask: bool = False
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.text_encoder.encode_prompts(prompts, max_length=max_length, return_mask=return_mask)  # type: ignore

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        # import pdb;pdb.set_trace()
        # input_key = self.input_data_key if input_key is None else input_key
        input_keys = ['video', 'video_aug_0', 'video_aug_1', 'control_input_depth', 'extra_video'] 
        if data_batch.get('use_ctrl', False):
            input_keys.append(self.config.hint_key)
        input_keys = list(set(input_keys))
        # only handle video batch
        for input_key in input_keys:
            if input_key in data_batch:
                # Check if the data has already been normalized and avoid re-normalizing
                if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                    assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                    assert torch.all(
                        (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
                    ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                else:
                    assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                    data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0

                if self.config.resize_online:
                    from torchvision.transforms.v2 import UniformTemporalSubsample

                    expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                    original_length = data_batch[input_key].shape[2]
                    if original_length != expected_length:
                        video = rearrange(data_batch[input_key], "b c t h w -> b t c h w")
                        video = UniformTemporalSubsample(expected_length)(video)
                        data_batch[input_key] = rearrange(video, "b t c h w -> b c t h w")
        data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        return self.tokenizer.encode(state) * self.sigma_data

    @staticmethod
    def get_context_parallel_group() -> Any | None:
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

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
        else:
            self.dit.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, T2VCondition]:
        # self._normalize_video_databatch_inplace(data_batch) # Remove to training_step 
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        extra_condition = {}
        if data_batch.get('use_ctrl', False):
            data_batch["hint_key"] = self.config.hint_key
            latent_hint = self.encode(data_batch[self.config.hint_key]) if self.config.hint_key != 'decoder_feature' else data_batch[self.config.hint_key]
            extra_condition[self.config.hint_key] = latent_hint
            
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            **extra_condition,
        )
        
        # print(condition.condition_video_input_mask_B_C_T_H_W[0][0][0].sum(), condition.condition_video_input_mask_B_C_T_H_W[0][0][1].sum(), condition.condition_video_input_mask_B_C_T_H_W[0][0][2].sum())
        return raw_state, latent_state, condition

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

    def blend_noise(self, noise_background, noise_foreground, alpha):
        # import pdb;pdb.set_trace()
        return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5
    
    def visualize_noise(self, noise):
        import imageio
        noise = noise[0, :3]/5 + 0.5
        noise = noise.clamp(0, 1)
        noise = noise.permute(1, 2, 3, 0)
        noise = (noise * 255).to(torch.uint8).cpu().numpy() 
        imageio.mimsave("/opt/tiger/noise.mp4", noise, fps=30 )

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
        # if step_idx == 0 and initial_noise is not None:
        #     net_state_in_B_C_T_H_W = self.blend_noise(initial_noise, net_state_in_B_C_T_H_W, 0.3)
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

        # forward pass through the network
        net_output_B_C_T_H_W = self.dit(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
            use_cuda_graphs=use_cuda_graphs,
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

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        use_cuda_graphs: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        # import pdb;pdb.set_trace()
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        extra_condition = {}
        if data_batch.get('hint_key', None) is not None:
            hint_key = data_batch['hint_key']
            latent_hint = self.encode(data_batch[hint_key]) if hint_key!= 'decoder_feature' else data_batch[hint_key]
            extra_condition[hint_key] = latent_hint

        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            **extra_condition
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            **extra_condition
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert (
                not self.dit.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition, use_cuda_graphs=use_cuda_graphs).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition, use_cuda_graphs=use_cuda_graphs).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    @torch.no_grad()
    def __call__(
        self,
        input_path: str,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor | None:
        # Parameter check
        width, height = VIDEO_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames in [1, 5], "num_conditional_frames must be 1 or 5"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        # Run text guardrail on the prompt
        if self.text_guardrail_runner is not None:
            from videoworld2.auxiliary.guardrail.common import presets as guardrail_presets

            log.info("Running guardrail check on prompt...")
            if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                return None
            else:
                log.success("Passed guardrail on prompt")
        elif self.text_guardrail_runner is None:
            log.warning("Guardrail checks on prompt are disabled")

        # refine prompt only if prompt refiner is enabled
        if (
            hasattr(self, "prompt_refiner")
            and self.prompt_refiner is not None
            and getattr(self.config, "prompt_refiner_config", None)
            and getattr(self.config.prompt_refiner_config, "enabled", False)
        ):
            log.info("Starting prompt refinement...")
            prompt = self.prompt_refiner.refine_prompt(input_path, prompt)
            log.info("Finished prompt refinement")

            # Run text guardrail on the refined prompt
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on refined prompt...")
                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    return None
                else:
                    log.success("Passed guardrail on refined prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on refined prompt are disabled")
        elif (
            hasattr(self, "config")
            and hasattr(self.config, "prompt_refiner_config")
            and not self.config.prompt_refiner_config.enabled
        ):
            log.warning("Prompt refinement is disabled")

        num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t)

        # Detect file extension to determine appropriate reading function
        ext = os.path.splitext(input_path)[1].lower()
        if ext in _VIDEO_EXTENSIONS:
            # Always use video reading for video files, regardless of num_latent_conditional_frames
            vid_input = read_and_process_video(
                input_path, [height, width], num_video_frames, num_latent_conditional_frames, resize=True
            )
        elif ext in _IMAGE_EXTENSIONS:
            if num_latent_conditional_frames == 1:
                # Use image reading for single frame conditioning with image files
                vid_input = read_and_process_image(input_path, [height, width], num_video_frames, resize=True)
            else:
                raise ValueError(
                    f"Cannot use multi-frame conditioning (num_conditional_frames={num_conditional_frames}) with image input. Please provide a video file."
                )
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported extensions are {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
            )

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input, prompt, negative_prompt, num_latent_conditional_frames=num_latent_conditional_frames
        )

        # preprocess
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=True, use_cuda_graphs=use_cuda_graphs
        )

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating world", leave=False)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        # Run video guardrail on the generated video and apply postprocessing
        if self.video_guardrail_runner is not None:
            # Clamp to safe range before normalization
            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2  # [0, 1]

            # Convert tensor to NumPy frames for guardrail processing
            video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            # Run guardrail
            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        return video

    @torch.no_grad()
    def validation(
        self,
        video,
        prompt_embedding,
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        use_cuda_graphs: bool = False,
        actions=None,
        hint=None,
        hint_key=None,
        initial_noise=None,
        initial_noise_level=0.3
    ) -> torch.Tensor | None:
        # Parameter check
        # width, height = VIDEO_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        # height, width = self.check_resize_height_width(height, width)
        # _DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        # import pdb;pdb.set_trace()
        with open("checkpoints/VideoWorld2_dLDM_2B/default_neg.pickle", "rb") as fp:
            neg_t5_text_embeddings = pickle.load(fp)
        neg_t5_text_embeddings = torch.from_numpy(neg_t5_text_embeddings[0])[None].to(dtype=self.torch_dtype)
        if neg_t5_text_embeddings.shape[1] < prompt_embedding.shape[1]:
            neg_t5_text_embeddings = torch.cat([neg_t5_text_embeddings.to(prompt_embedding), torch.zeros(1, prompt_embedding.shape[1]-neg_t5_text_embeddings.shape[1], prompt_embedding.shape[2]).to(prompt_embedding)], dim=1)

        # width, height = VIDEO_RES_SIZE_INFO[self.config.resolution]["16:9"]
        # height, width = self.check_resize_height_width(height, width)
        # video = torch.nn.functional.interpolate(video[0].permute(1, 0, 2, 3), (height, width)).permute(1, 0, 2, 3)[None]
        assert num_conditional_frames in [1, 5], "num_conditional_frames must be 1 or 5"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)
        
        H, W = video.shape[-2:]
        data_batch = {
                "dataset_name": "video_data",
                "video": video,
                "t5_text_embeddings": prompt_embedding,
                "neg_t5_text_embeddings": neg_t5_text_embeddings,
                "fps": torch.ones((1,)) * 16,  # Random FPS (might be used by model)
                "padding_mask": torch.zeros(1, 1, H, W),  # Padding mask (assumed no padding here)
                "num_conditional_frames": num_latent_conditional_frames,  # Specify number of conditional frames
            }
        if hint is not None:
            data_batch[hint_key] = hint
            data_batch['hint_key'] = hint_key

        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        # Prepare the data batch with text embeddings
        # data_batch = self._get_data_batch_input(
        #     vid_input, prompt, negative_prompt, num_latent_conditional_frames=num_latent_conditional_frames
        # )
        # preprocess
        # self._normalize_video_databatch_inplace(data_batch) # Move to video2world_model.py
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=True, use_cuda_graphs=use_cuda_graphs
        )

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        if initial_noise is not None:
            B, C, T, H, W = initial_noise.shape
            N = (T - 1) // (x_sigma_max.shape[2] - 1)
            if N > 1:
                first_noise_bg, later_noise_bg = initial_noise[:, :, :1], initial_noise[:, :, 1:]
                later_noise_bg = later_noise_bg.reshape(B, C, (T - 1)//N, N, H, W)
                later_noise_bg = later_noise_bg[:, :, :, 0, :, :]
                initial_noise = torch.cat([first_noise_bg, later_noise_bg], dim=2)
            initial_noise = initial_noise * self.scheduler.config.sigma_max
            x_sigma_max = self.blend_noise(initial_noise, x_sigma_max, initial_noise_level)
            # initial_noise = split_inputs_cp(x=initial_noise, seq_dim=2, cp_group=self.get_context_parallel_group())
    
        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())
        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating world", leave=False)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            # x0_pred = x0_fn(sample, sigma_in, initial_noise=initial_noise, step_idx=i)
            x0_pred = x0_fn(sample, sigma_in)
            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        # Run video guardrail on the generated video and apply postprocessing

        log.success("Video generation completed successfully")
        return video
