#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import torch.distributed as dist, torch
from tools.runner import master_only, get_dist_info
from ...registry import BOARDCASTS
from tools.parallel import scatter_kwargs
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

class BroadCast(object):

    def __init__(self):
        self.broadcast_bucket_size = 5 * 1024

    def receive(self, runner):
        raise NotImplementedError


    def send(self, runner, actions):
        raise NotImplementedError

@BOARDCASTS.register('one2many')
class One2many(BroadCast):

    def receive(self, runner):
        state, adj = runner.model.module.get_observation_state()
        return state, adj

    def send(self, runner, actions, **kwargs):
        dist.broadcast(actions, 0)
        runner.model.module.set_action(actions)
        dist.barrier()

@BOARDCASTS.register('many2many')
class Many2many(BroadCast):

    def receive(self, runner):
        (state, adj) = runner.model.module.get_observation_state()
        return state, adj

    def send(self, runner, actions, **kwargs):
        runner.model.module.set_action(actions)
        dist.barrier()



