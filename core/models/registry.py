#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
from core.utils.registry import Registry

# basic module
ACTOR = Registry('actor')  # actor network
CRITIC = Registry('critic')  # critic network
TOPO = Registry('topo')   # topological
CELL = Registry('cell')
FPN = Registry('fpn')
#losses
LOSSES = Registry('losses')

#training tools
RUNNERHOOKS = Registry('async_runner')
POLICIES = Registry('policies')  # policy. i.e. AC framework
#distributions
DISTRIBUTIONS = Registry('distributions')
BOARDCASTS = Registry('boardcasts')

