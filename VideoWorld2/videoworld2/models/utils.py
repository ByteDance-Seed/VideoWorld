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
import kornia as K
import kornia.augmentation as K_aug
from einops import reduce
from skimage.metrics import structural_similarity
import hashlib
import os
from contextlib import contextmanager

import torch
from safetensors.torch import load as safetensors_torch_load

from imaginaire.utils.easy_io import easy_io


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    backend_args = None
    state_dict = {}
    byte_stream = easy_io.load(file_path, backend_args=backend_args, file_format="byte")
    state_dict = safetensors_torch_load(byte_stream)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    backend_args = None
    state_dict = easy_io.load(
        file_path, backend_args=backend_args, file_format="pt", map_location="cpu", weights_only=False
    )
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(
                        search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    )
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()



@torch.no_grad()
def compute_psnr(
    ground_truth,
    predicted,
) :
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@torch.no_grad()
def compute_lpips(
    lpips_model,
    ground_truth,
    predicted,
):
    lpips_model = lpips_model.to(predicted.device)
    value = lpips_model.forward(ground_truth, predicted, normalize=True)
    lpips_model = lpips_model.to('cpu')
    return value[:, 0, 0, 0]

@torch.no_grad()
def compute_ssim(
    ground_truth,
    predicted,
):
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

class GeometricAugmentor_K():
    def __init__(self, geo_on=True) -> None:
        if geo_on:
            self.transform = K_aug.AugmentationSequential(
              
                K_aug.RandomAffine(
                    degrees=30.0,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2), 
                    p=0.7,
                    same_on_batch=True,
                ),
               
                K_aug.RandomPerspective(
                    distortion_scale=0.2, 
                    p=0.7,
                    same_on_batch=True,
                ),
             
                K_aug.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2, 
                    hue=0.055, 
                    p=1.0,
                    same_on_batch=True,
                ),
                data_keys=["input"],
            )
        else:
            self.transform = K_aug.AugmentationSequential(
                K_aug.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2, 
                    hue=0.055,  
                    p=1.0,
                    same_on_batch=True,
                ),
                data_keys=["input"],
            )
        # self.turn_on = turn_on

    def __call__(self, video_frames: dict) -> dict:
        # import pdb;pdb.set_trace()
        if video_frames.dtype == torch.uint8:
            video_frames = video_frames.float()
        if video_frames.max() > 1:
            video_frames = video_frames / 255.0

        augmented_frames = self.transform(video_frames) 
        # augmented_frames = torch.clip(augmented_frames, 0, 255)
        return augmented_frames