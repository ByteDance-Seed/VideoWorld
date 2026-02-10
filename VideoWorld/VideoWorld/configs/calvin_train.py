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
clip_length = 10
action_scope=5
test_wo_generate=True
model = dict(
    work_dir="/opt/tiger/",
    pred_image=True,
    pred_action=True,
    use_img_start=True,
    use_la_action=True,
    la_act_scope=action_scope,
    use_time_embedding=False,
    max_new_tokens=6,
    type='VideoWorldRobotics',
    vq_decoder_cfg=dict(width_mults=(4,4,2,2,1,1,1)),
    pre_encode_lang=True,
    use_clip_lang=True,
    test_wo_generate=test_wo_generate,
    vbackbone=dict(
        type='VQGANEncoder',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./work_dirs/calvin_fsq.pth'),
        width_mults=(1,1,1,2,2,4,4),
    ),
    neck=dict(
        type='InternLMGenModel',
        pretrain_path='./work_dirs/Intern_300m',
        vq_num=64000,
        sepcial_token_num=2+64000,
        use_text=False
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
    )
)

# dataset settings
dataset_type = 'CALVINDataset'
aux_info = ['input_ids', 'attention_mask', 'state', 'action', 'action_idx', 'lang_emb', 'prompt', 'hand', 'la_action']
to_tensor = ['input_ids', 'attention_mask', 'state', 'action', 'action_idx', 'lang_emb', 'hand', 'la_action']
test_aux_info = ['input_ids', 'attention_mask', 'scene', 'state', 'eval_sequence', 'initial_state', 'action_idx']
test_to_tensor = ['input_ids', 'attention_mask', 'scene', 'state']

img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    rescale=True,
    norm_pred_label=True)

train_pipeline = [
    dict(type='VideoNormalize', **img_norm_cfg),
    dict(
        type='TokenizerforCALVIN',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        pred_image=True,
        pred_image_num=0,
        use_lang_embed=True,
    ),
    dict(type='Collect', keys=['img', *aux_info]),
    dict(type='ToTensor', keys=['img', *to_tensor]),
]

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='TokenizerforCALVIN',
        input_text='prompt',
        padding_side='right',
        max_length = 1024,
        pred_image=True,
        pred_image_num=0,
        use_lang_embed=True,
    ),

    dict(type='Collect', keys=['img', *test_aux_info]),
    dict(type='ToTensor', keys=['img', *test_to_tensor]),
]

data_root = "./data/calvin/task_ABCD_D/training"
la_data_path = "./work_dirs/calvin_ldm_results.pth"
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    pin_memory=False,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CALVINDataset',
        data_root=data_root,
        clip_length=clip_length,
        interval_range=[1, 1],
        pipeline=train_pipeline,
        use_hand=True,
        la_path=la_data_path
        ),
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    use_web=False,
    pin_memory=False,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CALVINEnvValDataset',
        data_root = "./data/calvin/task_ABCD_D/",
        pipeline=test_pipeline,
        num_sequences=20),
    )

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
        end=15000,
        ),
    dict(
        type='CosineAnnealingLR',
        T_max=485000,
        by_epoch=False,
        begin=15000,
        end=500001,
        )
]

# runtime settings
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=500001,
    val_begin=30001,
    val_interval=30000000,
)
val_cfg = dict()
val_evaluator = dict(type='GoEva', collect_device='cpu', collect_dir='./work_dirs/multinode_test')

test_cfg = dict()
test_evaluator = dict(type='GoEva', collect_device='cpu', collect_dir='./work_dirs/multinode_test')

# runtime settings
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook', save_hdfs=False),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=30000, by_epoch=False, max_keep_ckpts=10, save_hdfs=False),
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
