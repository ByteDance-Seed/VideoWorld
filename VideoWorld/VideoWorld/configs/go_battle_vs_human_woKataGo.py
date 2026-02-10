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
default_scope = 'falcon'
level='9d'
model = dict(
    work_dir="/opt/tiger/",
    pred_image=True,
    pred_action=True,
    type='VideoWorldGoBattleVSHumanwoKataGo',
    battle_with_katago={'level':level},
    mode='go_battle',
    kata_ana=True,
    max_generate_length=1+16+1 + 1+1+1 + 1+16+1 +1,
    vq_decoder_cfg=dict(width_mults=(4,4,2,2,1,1,1)),
    vbackbone=dict(
        type='VQGANEncoder',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./work_dirs/go_fsq.pth'),
        width_mults=(1,1,1,2,2,4,4),
        ),
        
    neck=dict(
        type='InternLMGenModel',
        pretrain_path='./work_dirs/Intern_300m',
        vq_num=64000,
        sepcial_token_num=3+64000+2,
        use_text=True
    ),
    quantizer=dict(
        type='FSQ',
        levels=[8,8,8,5,5,5],
        dim=256,
    ),
    head=dict(
        type='ITGHead',
        text_weight=1.0,
        ignore_index=-100,
        
    ))

# dataset settings
aux_info = ['input_ids', 'attention_mask', 'pred_label', 'invalid']
test_aux_info = ['input_ids', 'attention_mask', 'level', 'data_mode', 'katrain_level']
test_to_tensor = ['input_ids', 'attention_mask']
dataset_type = '_WebGoImageDataset'
img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)

train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='TokenizerforGoImage',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        game_type='short',
        board_size=9,
        sub_board_size=9,
        offset=(0, 0, 0, 80),
        pred_image=True,
        is_llama=True,
        use_action=True
    ),
    dict(type='Collect', keys=['img', *aux_info]),
    dict(type='ToTensor', keys=['img', *aux_info]),
]

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='TokenizerforGoImage',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        game_type='short',
        board_size=9,
        sub_board_size=9,
        offset=(0, 0, 0, 80),
        test_mode=True,
        pred_image=True,
        is_llama=True,
        use_action=True
    ),
    dict(type='Collect', keys=['img', *test_aux_info]),
    dict(type='ToTensor', keys=['img', *test_to_tensor]),
]

# dataset summary
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    use_web=True,
    dataset=dict(
        type=dataset_type,
        shuffle=True,
        dataset_length=10000000,
        shards='data/go_dataset_size9/kataselfpaly_filterdup_ignore_wcap_10M_image_tar/{000000000..000001259}.tar',
        pipeline=train_pipeline,
        pred_image=True,
        level_list = ['9d'],
        
        
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='MyConcatDataset',
        datasets=[
            dict(type='GoImageValDataset',
                pipeline=test_pipeline,
                level_list=['all'],
                length=30,
                level=level),
       
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False)


test_dataloader = val_dataloader

# base_lr = 1.5e-4
# lr = base_lr * data['samples_per_gpu'] * update_interval * gpu_num / 256
paramwise_options = dict(
    custom_keys={
        'neck.layers': dict(lr_mult=1.0),
        'neck.embed_tokens': dict(lr_mult=0.1),
    },
    bypass_duplicate=True)

optimizer = dict(
    type='AdamW', lr=1.0e-4, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-6)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=optimizer,
    dtype="float16",
    accumulative_counts=3,
    clip_grad=dict(max_norm=0.2),
    paramwise_cfg=paramwise_options,
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.000000001,
        by_epoch=False,
        begin=0,
        end=30000,
        ),
    dict(
        type='CosineAnnealingLR',
        T_max=970000,
        by_epoch=False,
        begin=30000,
        end=1000001,
        )
]

# runtime settings
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000001,
    val_begin=200000,
    val_interval=100000
)
val_cfg = dict()
val_evaluator = dict(type='GoEva', collect_device='cpu', collect_dir='./work_dirs/multinode_test')

test_cfg = dict()
test_evaluator = dict(type='GoEva', collect_device='cpu', collect_dir='./work_dirs/multinode_test')

# runtime settings
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=30000, by_epoch=False, max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = []

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    by_epoch=False,
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])


log_level = 'INFO'
load_from = None
compile_options = dict(backend='inductor', mode='max-autotune')
cfg = dict(compile=compile_options)
resume = True
randomness = dict(seed=60, diff_rank_seed=True)
