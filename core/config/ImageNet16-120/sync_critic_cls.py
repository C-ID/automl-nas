#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

topo_nums = [8, 8, 8]                     # Three topo graph, construct whole picture
state_space = [int(8*7/2+8*7/2+8*7/2), 5] # i.e. dims = [[topo[0]*(topo[0]-1))/2 , 5],
                                             #              [topo[1]*(topo[1]-1))/2 , 5],
                                             #              [topo[2]*(topo[2]-1))/2 , 5]
cls_num=120
Space = dict(
    observation_space=state_space,
    action_space=topo_nums,
)

Critic=dict(
    type='CriticTopo',
    backbone=dict(
        type='ScalableTopo',
        NodeNum=topo_nums[0],
        C_in=64,
        C_out=64,
        stride=1,
        affine=False,
        track_running_stats=True,
        search_space="nas-bench-201",
        downsample=dict(
            type='normal',
            down_sample_step=8,
            down_sample_ratio=2,
        ),
    ),
    reduction_bn=dict(
        type="ResNetBasicblock",
        C_in=64,
        C_out=64,
        stride=1,
        affine=True,
        track_running_stats=True
    ),
    neck=dict(
        type='ScalableTopo',
        NodeNum=topo_nums[1],
        C_in=64,
        C_out=64,
        stride=1,
        affine=True,
        track_running_stats=True,
        search_space="nas-bench-201",
        downsample=dict(
            type='normal',
            down_sample_step=32,
            down_sample_ratio=2,
        )
    ),
    reduction_nh=dict(
        type="ResNetBasicblock",
        C_in=64,
        C_out=64,
        stride=1,     #Carefully tunning
        affine=True,
        track_running_stats=True
        ),
    cls_head=dict(
        type='ScalableTopo',
        NodeNum=topo_nums[2],
        C_in=64,
        C_out=32,
        stride=1,
        affine=True,
        track_running_stats=True,
        search_space="nas-bench-201",
        downsample=dict(
            type='normal',
            down_sample_step=16,
            down_sample_ratio=2,
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

# dataset tools
dataset_name = 'ImageNet16'
image_path = "/app/member/tongyaobai/XAutoDL/TORCH_HOME/cifar.python/ImageNet16"
split_txt="/app/member/tongyaobai/XAutoDL/configs/nas-benchmark"
img_norm_cfg = dict(
    mean=[x / 255 for x in [122.68, 116.66, 104.01]], std=[x / 255 for x in [63.22, 61.26, 65.09]])
train_pipeline = [
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomCrop', size=16, padding=2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

data = dict(
    type = "cls",
    imgs_per_gpu=250,
    workers_per_gpu=1,
    train=dict(
        type=dataset_name,
        root=image_path,
        is_train=True,
        transform=train_pipeline,
        use_num_of_class_only=cls_num),
    val=dict(
        type=dataset_name,
        root=image_path,
        is_train=False,
        transform=test_pipeline,
        use_num_of_class_only=cls_num),
    test=dict(
        type=dataset_name,
        root=image_path,
        is_train=False,
        transform=test_pipeline,
        use_num_of_class_only=cls_num)
)

# training and testing settings, w.r.t critic and actor net
c_optim = dict(
    optimizer=dict(type='SGD', lr=0.025, momentum=0.95, weight_decay=0.0003),
    optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)),
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
                min_lr=1e-5
                )
)

# logger
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
total_epochs = 1000
workflow = [('train', 1), ('val', 1)]
log_level = 'INFO'
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
load_from = '/app/member/tongyaobai/Automl-NAS/res/ImageNet16-120/test/last_checkpoint.pth'
# load_from = None
resume_from=None

work_dir = 'ImageNet16-120/test'

