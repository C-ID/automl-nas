#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
topo_nums = [34, 16, 8]                         # Three topo graph, construct whole picture
state_space = [int(34*33/2+16*15/2+8*7/2), 5]     #  i.e. dims = [[topo[0]*(topo[0]-1))/2 , 5],
                                              #               [topo[1]*(topo[1]-1))/2 , 5],
                                              #               [topo[2]*(topo[2]-1))/2 , 5]
cls_num=1000
Space = dict(
    observation_space=state_space,
    action_space=topo_nums,
)

Critic=dict(
    type='ReductCritic',
    backbone=dict(
        type='StageTopo',
        NodeNum=topo_nums[0],
        C_in=64,
        C_out=128,
        stride=1,
        affine=False,
        track_running_stats=True,
        search_space="nas-bench-201",
        reduction_factor=[17],
        fpn=dict(
            type='wrapperFPN',
            stride=1
        )
    ),
    reduction_bn=dict(
        type="ResNetBasicblock",
        C_in=128,
        C_out=128,
        stride=1,
        affine=True,
        track_running_stats=True
    ),
    neck=dict(
        type='StageTopo',
        NodeNum=topo_nums[1],
        C_in=128,
        C_out=128,
        stride=1,
        affine=True,
        track_running_stats=True,
        search_space="nas-bench-201",
        reduction_factor=[],
        fpn=dict(
            type='wrapperFPN',
            stride=1
        )
    ),
    reduction_nh=dict(
        type="ResNetBasicblock",
        C_in=128,
        C_out=128,
        stride=1,     #Carefully tunning
        affine=True,
        track_running_stats=True
        ),
    cls_head=dict(
        type='StageTopo',
        NodeNum=topo_nums[2],
        C_in=128,
        C_out=32,
        stride=1,
        affine=True,
        track_running_stats=True,
        search_space="nas-bench-201",
        reduction_factor=[],
        fpn=dict(
            type='wrapperFPN',
            stride=1
        )
    ),
    num_classes=cls_num,
    loss_func=dict(
        train=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            use_soft=False,
            reduction='mean',
            loss_weight=1.0,
            class_weight=None,
            pos_weight=None),
        eval=dict(type='Accurary', topk=(1, 5))
    )
)

Actor=dict(
     # Configure action space dim here. Must be the same as the number of critic nodes num.
    type='ActorGCN',
    features_extractor="GCN",
    features_extractor_kwargs=dict(
        n_layers=2,
        in_features=5,
        hidden=10,
        out_features=2
    )
)


# dataset tools
dataset_name = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(248, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(248, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    type = "cls",
    imgs_per_gpu=200,
    workers_per_gpu=4,
    train=dict(
        type=dataset_name,
        data_prefix='/home/Data/imagenet/ILSVRC2012_img_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_name,
        data_prefix='/home/Data/imagenet/ILSVRC2012_img_val',
        ann_file='/home/Data/imagenet/ILSVRC2012_val.txt',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_name,
        data_prefix='/home/Data/imagenet/ILSVRC2012_img_val',
        ann_file='/home/Data/imagenet/ILSVRC2012_val.txt',
        pipeline=test_pipeline)
)


# training and testing settings, w.r.t critic and actor net
c_optim = dict(
    optimizer=dict(type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0003),
    optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)),
)
a_optim = dict(
    optimizer=dict(type='Adam', lr=0.3, betas=(0.5,0.999), weight_decay=0.0003),
)

# learning policy
lr_config = dict(
        c_cfg = dict(
                policy='CosineAnnealing',
                by_epoch=False,
                warmup='linear',
                warmup_iters=10,
                warmup_ratio=1.0 / 3,
                warmup_by_epoch=True,
                min_lr=0
                ),
        a_cfg = dict(
            policy='step',
            by_epoch=True,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=1.0 / 3,
            warmup_by_epoch=True,
            step=[100, 150, 300])
)
# RL policy
Policy = dict(
    type='EMAPolicyHook',
    learning_step=30,
    distribution=dict(type='CategoricalDist', action_dim=state_space[0]),
    action_space=topo_nums,
    grad_clip=dict(max_norm=10, norm_type=1),
    broadcast=dict(type='one2many'),
    momentum=0.9
)
evaluation = dict(interval=1, metric='accuracy')

# logger
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
total_epochs = 500
workflow = [('train', 1), ('val', 1)]
log_level = 'INFO'
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
load_from = None
resume_from = None

work_dir = 'ImageNet/reduct_noAug_b34_n16'
