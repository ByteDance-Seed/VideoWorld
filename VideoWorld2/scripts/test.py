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
import argparse
import importlib
import os
from torchvision.transforms.v2 import UniformTemporalSubsample
from loguru import logger as logging
from PIL import Image
import torchvision.transforms.functional as F
from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override
import torch
from decord import VideoReader, cpu
import numpy as np
from torchvision import transforms as T
from videoworld2.data.dataset_utils import Resize_Preprocess, ToTensorVideo


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    # frame_ids = np.linspace(0, len(vr) - 1).astype(np.int32)
    frame_ids = np.arange(0, len(vr)).tolist()
    vr.seek(0)
    frame_data = vr.get_batch(frame_ids).asnumpy()
    try:
        fps = vr.get_avg_fps()
    except Exception:  # failed to read FPS
        fps = 24
    return frame_data, fps

def get_frames(video_path):
    preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess((480, 832))])
    frames, fps = load_video(video_path)
    frames = frames.astype(np.uint8)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
    frames = UniformTemporalSubsample(93)(frames)
    frames = preprocess(frames)
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
    return frames, fps

@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Create the model
    model = instantiate(config.model)
    # Create the dataloaders.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )

def load_frame(input_frame_path):
    input_frame = Image.open(input_frame_path).convert('RGB')
    input_frame = F.resize(input_frame, (480, 832))
    input_frame = F.to_tensor(input_frame)
    input_frame = input_frame.view(1, 3, 1, 480, 832)
    input_frame = torch.clamp(input_frame * 255.0, 0, 255).to(torch.uint8)
    input_frame = input_frame.to('cuda')
    return input_frame

@logging.catch(reraise=True)
def infer(config: Config, args: argparse.Namespace) -> None:
    
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    
    model = model.to("cuda", memory_format=trainer.config.trainer.memory_format)  # type: ignore
    model.on_train_start(trainer.config.trainer.memory_format)
    trainer.callbacks.on_optimizer_init_start()
    optimizer, scheduler = model.init_optimizer_scheduler(trainer.config.optimizer, trainer.config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **trainer.config.trainer.grad_scaler_args)
    trainer.callbacks.on_optimizer_init_end()
    # Load the model checkpoint and get the starting iteration number.

    quant_infos = torch.load('checkpoints/VideoWorld2_dLDM_2B/VideoCraft-dLDM-codes.pt')
    if args.input_path != 'default':
        input_frame = load_frame(args.input_path)
        vid_name = args.input_path.split('/')[-1].replace('.jpg', '')
        latent_dynamic_codes = torch.tensor(quant_infos[vid_name]).to('cuda')
        model.inference(input_frame, latent_dynamic_codes)

    else:
        print('Inference using all assets in videocraft_example folder')
        input_frame_root = 'assets/videocraft_example'
        input_frame_list = os.listdir(input_frame_root)
        for input_frame_name in input_frame_list:
            input_frame_path = os.path.join(input_frame_root, input_frame_name)
            input_frame = load_frame(input_frame_path)

            latent_dynamic_codes = torch.tensor(quant_infos[input_frame_name.replace('.jpg', '')]).to('cuda')
            model.inference(input_frame, latent_dynamic_codes)

    
    # for quant_info in quant_infos[10:]:
    #     vid_name = quant_info[1]
    #     vid_path = f'/opt/tiger/mix_allstep_infer_f177_1110/{vid_name}_0.mp4'
    #     video, fps = get_frames(vid_path)
    #     # frames = video.permute(1, 0, 2, 3)[:, :1][None].to('cuda')
    #     frames = video.permute(1, 0, 2, 3)[:, :1][None].to('cuda')
    #     # frames = video.permute(1, 0, 2, 3)[None].to('cuda')
    #     # frames = frames.astype(np.uint8)
    #     # frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
    #     # frames = F.resize(frames, (480, 832))[:1].permute(1, 0, 2, 3)[None]
    #     # frames = torch.clamp(frames, 0, 255).to(torch.uint8).to('cuda')
    #     latent_dynamic_codes = torch.tensor(quant_info[0]).to('cuda')
    #     model.inference(frames, latent_dynamic_codes)

if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=videoworld2/configs/base/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets
    
    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument("--mode", help="Path to the config file", required=True)
    parser.add_argument("--input_path", default='', required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
   
    if args.mode == 'train':
        # Launch the training job.
        launch(config, args)
    else:
        infer(config, args)