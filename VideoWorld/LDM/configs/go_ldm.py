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
default_scope = 'ldm'
import os

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

ema_config = dict(
    type='ExponentialMovingAverage',
    interval=1,
    momentum=0.0002,
    update_buffers=True,
    start_iter=200000)

model = dict(
    type='MagVitVQGAN',
    only_last_frame=True,
    test_return_feat=True,
    data_preprocessor=dict(type='DataPreprocessor',output_channel_order='RGB',),
    ema_config=ema_config,
    only_2frame=True,
    generator=dict(
        type='MagVitGenerator',
        encoder=dict(type='MagvitV2LAencoder',
                image_size=128,
                channels=3,
                init_dim=128,
                pre_out_layer=9,
                use_la_norm=True,
                use_qformer_for_code=True,
                sep_qformer=True,
                frame_num=2,
                act_embedding_num=1,
                input_conv_kernel_size=(7,7,7),
                time_padding=3,
                layers=(
                 ('consecutive_residual', 4),
                 ('spatial_down', 1),
                 ('channel_residual', 1),
                 ('consecutive_residual', 3),
                 ('spatial_down', 1),
                 ('consecutive_residual', 4),
                 ('spatial_down', 1),
                 ('channel_residual', 1),
                 ('consecutive_residual', 3),
                 ('consecutive_residual', 4),
                
                 )
                ),
        quantizer=dict(type='FSQ',
                levels=[8,8,8,5,5,5],
                dim=512),
        decoder=dict(type='MagvitV2LAAdadecoder',
                     image_size=128,
                     channels=3,
                     init_dim=128,
                     use_pre_video=False,
                     use_pre_encode=True,
                     time_padding=3,
                     layers=(
                         ('consecutive_residual', 3),
                         ('channel_residual', 1),
                         ('condation',1),
                         ('spatial_up', 1),
                         ('consecutive_residual', 4),
                         ('condation',1),
                         ('spatial_up', 1),
                         ('consecutive_residual', 3),
                         ('channel_residual', 1),
                         ('condation',1),
                         ('spatial_up', 1),
                         ('consecutive_residual', 4),
                         ('condation',1),
                         ('consecutive_residual', 4)
                     ),
                     
    ),),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=128
    ),
    loss_config=dict(
    perceptual_weight=1.0,
    disc_weight=0.75,
    grad_penalty_loss_weight=1.,
    perceptual_loss=dict(type='LPIPS')
    )
)

# dataset settings

# different from mmcls, we adopt the setting used in BigGAN.
# Remove `RandomFlip` augmentation and change `RandomCropLongEdge` to
# `CenterCropLongEdge` to eliminate randomness.
# dataset settings
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='PackVideoInputs')
]

test_pipeline = [
    dict(type='ResizeVideo', scale=(128, 128), keep_ratio=False),
    dict(type='PackVideoInputs', meta_keys=['action', 'capture'])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='_WebGoImageDataset',
        shuffle=True,
        dataset_length=10000000,
        shards='data/go_dataset_size9/kataselfpaly_filterdup_ignore_wcap_AI_image_tar/{000000000..000001259}.tar',
        pipeline=train_pipeline,
        clip_length=5,
        interval=2,
        size=128,
        ),

    )

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type='GoImageMF',
        interval=3,
        sample_num=50000,
        data_root="/opt/tiger/PointVIS/la_train_ai/opt/tiger/kataselfpaly_filterdup_ignore_wcap_10M_image/",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False)

test_dataloader = val_dataloader

# config for model wrapper
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True)

# config for optim_wrapper_constructor lr=5.4e-5,
#             betas=(0.5, 0.9),
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=5.4e-5, betas=(0.5, 0.9))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=4.32e-4, betas=(0.5, 0.9))))


# config for training
train_cfg = dict(by_epoch=False, max_iters=500000,val_begin=1, val_interval=500000)
# train_cfg = dict(by_epoch=True, max_epochs=100,val_begin=10, val_interval=1)


metrics = [
    dict(
        type='FVD',
        prefix='FVD',
        fake_nums=19772,
        inception_style='StyleGAN',
        sample_model='ema'),
]

# config for val
val_cfg = dict(type='MultiValLoop')
val_evaluator = dict(type='Evaluator',metrics=metrics)

# config for test
test_cfg = dict()
test_evaluator = dict(type='LAGoFeatMetric',collect_device='cpu', la_num = 729, gt_act_num = 81, gt_select_frame=[1])

# load from which checkpoint
# load_from = '/opt/tiger/mmagicinit/ldm/work_dirs/go_2frame_sepqformer_likebefore_interval5/iter_64000_new.pth' # load_from=None
load_from = './work_dirs/magvit_init.pth'
# whether to resume training from the loaded checkpoint
resume = False

# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True,
        ))

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='nccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type='LogProcessor', by_epoch=False)

# env settings
dist_params = dict(backend='nccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# set visualizer
vis_backends = [dict(type='VisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)






