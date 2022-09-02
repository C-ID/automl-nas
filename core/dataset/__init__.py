#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from .dataset_construct import ImageNet16, ImageNet
# from .loader import DistributedGroupSampler, GroupSampler, build_dataloader
from .loader import DistributedSampler, build_dataloader
from .piplines import (AutoAugment, AutoContrast, Brightness,
                       ColorTransform, Contrast, Cutout, Equalize, Invert,
                       Posterize, RandAugment, Rotate, Sharpness, Shear,
                       Solarize, SolarizeAdd, Translate, Compose, Collect,
                       ImageToTensor, ToNumpy, ToPIL, ToTensor, Transpose,
                       to_tensor, LoadImageFromFile, CenterCrop, ColorJitter,
                       Lighting, Normalize, Pad, RandomCrop, RandomErasing,
                       RandomFlip, RandomGrayscale, RandomResizedCrop, Resize)
from .registry import DATASETS_SEG, DATASETS_PCD, DATASETS_DET, DATASETS_CLS, PIPELINES
from .builder import build_dataset, build_piplines

__all__ = [
    'build_dataset', 'ImageNet16', 'DistributedSampler',
    #'DistributedGroupSampler', 'GroupSampler',
    'build_dataloader', 'DATASETS_SEG', 'DATASETS_PCD', 'DATASETS_DET',
    'DATASETS_CLS', 'AutoAugment', 'AutoContrast', 'Brightness',
    'ColorTransform', 'Contrast', 'Cutout', 'Equalize', 'Invert',
    'Posterize', 'RandAugment', 'Rotate', 'Sharpness', 'Shear',
    'Solarize', 'SolarizeAdd', 'Translate', 'Compose', 'Collect',
    'ImageToTensor', 'ToNumpy', 'ToPIL', 'ToTensor', 'Transpose',
     'to_tensor', 'LoadImageFromFile', 'CenterCrop', 'ColorJitter',
     'Lighting', 'Normalize', 'Pad', 'RandomCrop', 'RandomErasing',
     'RandomFlip', 'RandomGrayscale', 'RandomResizedCrop', 'Resize',
     'PIPELINES', 'build_piplines'
]