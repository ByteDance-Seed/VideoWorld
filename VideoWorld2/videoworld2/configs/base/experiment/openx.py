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
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler
from videoworld2.latent_dynamics.modules import DiscreteQuantizer, Encoder3DType, Decoder3DType
from videoworld2.data.dataset_video import Dataset, InferDataset
from imaginaire.lazy_config import LazyCall as L
from videoworld2.data.dataset_openx_pickle import  EnhancedOpenXPickleDataset
# from videoworld2.data.dataset_openx_pickle_latent_gen import EnhancedOpenXPickleLatentGenDataset
# from videoworld2.data.dataset_calvin import CALVINDataset

def get_sampler(dataset, shuffle=True) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=shuffle,
        seed=0,
    )


cs = ConfigStore.instance()


# training with only videocraft 
example_video_dataset_videocraft = L(Dataset)(
    dataset_dir="datasets/Video-CraftBench/Paper_and_Block_clips",
    num_frames=93,
    video_size=(480, 832),
)

example_video_dataset_videocraft_val = L(InferDataset)(
    dataset_dir="datasets/Video-CraftBench/Paper_and_Block_clips",
    num_frames=93,
    video_size=(480, 832),
)

# training with only videocraft and openx
example_video_dataset_openx_videocraft = L(EnhancedOpenXPickleDataset)(
    dataset_dir="datasets/openx_untar",
    num_frames=93,
    video_size=(480, 832),
    subfolders=None,
    use_cache=True,
    cache_path="datasets/openx_videocraft_cache.json",
    meta_path="datasets/openx_videocraft_meta.json",
    fps_variation=False
)



# dataloader: openx train
dataloader_train_videocraft = L(DataLoader)(
    dataset=example_video_dataset_videocraft,
    sampler=L(get_sampler)(dataset=example_video_dataset_videocraft),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

dataloader_val_videocraft = L(DataLoader)(
    dataset=example_video_dataset_videocraft_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_videocraft_val),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

dataloader_train_openx_videocraft = L(DataLoader)(
    dataset=example_video_dataset_openx_videocraft,
    sampler=L(get_sampler)(dataset=example_video_dataset_openx_videocraft),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)


videoworld2_dldm_openx_and_videocraft_wodit_warmup = dict(
    defaults=[
        {"override /model": "predict2_video2world_ldm_fsdp_2b_480p_16fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="videoworld2_dldm_openx_and_videocraft_wodit_warmup",
    ),
    model=dict(
        config=dict(
            dldm_warmup=True,
            pipe_config=dict(
                resize_online=False,
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
            ),
            train_architecture='lora',
            ldm_config=dict(
                network=dict(
                    decoder=Decoder3DType.LDM_V2.name, 
                    levels=[8,5,5,5],
                    embedding_dim=4,
                    persistent_quantizer=False,
                    z_channels=256,
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=16,
                    act_embedding_num=4,
                    qformer_type='QFormerAdjacentFSingleQ',
                ),
                ldm_path="checkpoints/VideoWorld2_dLDM_2B/tokenizer/ldm_tokenizer_training_init_weights.pt"
            ),
            
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_openx_videocraft,
    dataloader_val=dataloader_val_videocraft,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=100000),
        ),
        max_iter=100000,
        run_validation=False
    ),
    checkpoint=dict(
        save_iter=5000,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[400_000],
        f_max=[0.6],
        f_min=[0.3],
    ),
)


videoworld2_dldm_openx_and_videocraft_only_dit = dict(
    defaults=[
        {"override /model": "predict2_video2world_ldm_fsdp_2b_480p_16fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="videoworld2_dldm_openx_and_videocraft_only_dit",
    ),
    model=dict(
        config=dict(
            only_dit=True,
            use_cross_embedding=False,
            pipe_config=dict(
                resize_online=False,
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,                
            ),
            train_architecture='lora',
            ldm_config=dict(
                network=dict(
                    decoder=Decoder3DType.LDM_V2.name, 
                    levels=[8,5,5,5],
                    embedding_dim=4,
                    persistent_quantizer=False,
                    z_channels=256,
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=16,
                    act_embedding_num=4,
                    qformer_type='QFormerAdjacentFSingleQ',
                ),
                ldm_path="checkpoints/VideoWorld2_dLDM_2B/tokenizer/ldm_tokenizer_training_init_weights.pt"
            ),
            
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_openx_videocraft,
    dataloader_val=dataloader_val_videocraft,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=100000),
        ),
        max_iter=300000,
    ),
    checkpoint=dict(
        save_iter=20000,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[400_000],
        f_max=[0.6],
        f_min=[0.3],
    ),
)


videoworld2_dldm_openx_and_videocraft = dict(
    defaults=[
        {"override /model": "predict2_video2world_ctrl_ldm_fsdp_2b_480p_16fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        # {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="videoworld2_dldm_openx_and_videocraft",
    ),
    model=dict(
        config=dict(
            use_cross_embedding=True,
            finetune_base_model=True,
            pipe_config=dict(
                resize_online=False,
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net_ctrl=dict(
                    control_weight=0.5,
                    layer_mask=[True if (i >= 3) else False for i in range(28)],
                ),
                hint_key="recon_image"
                
            ),
            train_architecture='base',
            ldm_config=dict(
                network=dict(
                    decoder=Decoder3DType.LDM_V2.name, 
                    levels=[8,5,5,5],
                    embedding_dim=4,
                    persistent_quantizer=False,
                    z_channels=256,
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=16,
                    act_embedding_num=4,
                    qformer_type='QFormerAdjacentFSingleQ',
                ),
                ldm_path="checkpoints/VideoWorld2_dLDM_2B/VideoWorld2_dLDM_VAE.pt"
                # ldm_path="checkpoints/posttraining/video2world/VideoWorld2_dLDM_2B/videoworld2_dldm_openx_and_videocraft_wodit_warmup/checkpoints/model/iter_000100000_ldm.pt"
            ),
            
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_openx_videocraft,
    dataloader_val=dataloader_val_videocraft,

    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=100000),
        ),
        max_iter=30000,
        run_validation=False
    ),
    checkpoint=dict(
        save_iter=5000,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[400_000],
        f_max=[0.6],
        f_min=[0.3],
    ),
)



videoworld2_dldm_openx_and_videocraft_infer = dict(
    defaults=[
        {"override /model": "predict2_video2world_ctrl_ldm_fsdp_2b_480p_16fps_infer"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        # {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="videoworld2_dldm_openx_and_videocraft_infer",
    ),
    model=dict(
        config=dict(
            use_cross_embedding=True,
            finetune_base_model=True,
            model_manager_config=dict(
                dit_path="checkpoints/VideoWorld2_dLDM_2B/VideoWorld2_dLDM_DiT.pth",
            ),
            pipe_config=dict(
                resize_online=False,
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                net_ctrl=dict(
                    control_weight=0.5,
                    layer_mask=[True if (i >= 3) else False for i in range(28)],
                ),
                hint_key="recon_image"
                
            ),
            train_architecture='base',
            ldm_config=dict(
                network=dict(
                    decoder=Decoder3DType.LDM_V2.name, 
                    levels=[8,5,5,5],
                    embedding_dim=4,
                    persistent_quantizer=False,
                    z_channels=256,
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=16,
                    act_embedding_num=4,
                    qformer_type='QFormerAdjacentFSingleQ',
                ),
                ldm_path="checkpoints/VideoWorld2_dLDM_2B/VideoWorld2_dLDM_VAE.pt"
            ),
            
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_videocraft,
    dataloader_val=dataloader_train_videocraft,

    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=100000),
        ),
        max_iter=300000,
        run_validation=True
    ),
    checkpoint=dict(
        save_iter=5000,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[400_000],
        f_max=[0.6],
        f_min=[0.3],
    ),
)




for _item in [
    # 2b, paper_airplane
    videoworld2_dldm_openx_and_videocraft,
    videoworld2_dldm_openx_and_videocraft_only_dit,
    videoworld2_dldm_openx_and_videocraft_wodit_warmup,
    videoworld2_dldm_openx_and_videocraft_infer #only for inference
  
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )



