#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from .registry import PIPELINES, DATASETS_CLS, DATASETS_DET, DATASETS_SEG, DATASETS_PCD
from core.utils.registry import build_from_cfg


def build_piplines(cfg):
    return build_from_cfg(cfg, PIPELINES)


def build_dataset(cfg, basis_type, default_args=None):
    if basis_type=='cls':
        dataset = build_from_cfg(cfg, DATASETS_CLS, default_args)
    elif basis_type=='det':
        dataset = build_from_cfg(cfg, DATASETS_DET, default_args)
    elif basis_type=='seg':
        dataset = build_from_cfg(cfg, DATASETS_SEG, default_args)
    elif basis_type=='pcd':
        dataset = build_from_cfg(cfg, DATASETS_PCD, default_args)
    else:
        raise NotImplementedError('Dataset type has exceeded the scope '
                                  'of the code framework. Currently we '
                                  'only support four types, i.e. cls(Classifiaction), '
                                  'det(2D Detection), seg(Segmentation), '
                                  'pcd(3D Detection) ')
    return dataset
