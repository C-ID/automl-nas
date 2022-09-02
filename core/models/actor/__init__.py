#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from .actor_gcn import ActorGCN
from .actor_mlp import ActorMLP

__all__ = [
    'ActorGCN', 'ActorMLP'
]