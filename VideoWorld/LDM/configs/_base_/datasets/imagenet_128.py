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
# dataset settings
dataset_type = 'ImageNet'

# different from mmcls, we adopt the setting used in BigGAN.
# We use `RandomCropLongEdge` in training and `CenterCropLongEdge` in testing.
train_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='RandomCropLongEdge', keys='gt'),
    dict(type='Resize', scale=(128, 128), keys='gt', backend='pillow'),
    dict(type='Flip', keys='gt', flip_ratio=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='CenterCropLongEdge', keys='gt'),
    dict(type='Resize', scale=(128, 128), keys='gt', backend='pillow'),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=None,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='./data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='./data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader
