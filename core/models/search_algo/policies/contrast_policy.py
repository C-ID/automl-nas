#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
# some bugs here, not fixed

from ...registry import POLICIES
from .policy import PolicyHook


@POLICIES.register('ContrastPolicyHook')
class ContrastPolicyHook(PolicyHook):
    def __init__(self, distribution, action_space, clip_grads, broadcast):
        super(ContrastPolicyHook, self).__init__(
            distribution,
            action_space,
            clip_grads,
            broadcast
        )

    def _learn(self, runner):
        # actor learning
        if hasattr(runner.actor, 'module'):
            runner.actor = runner.actor.module
        advantage = runner.outputs['critic_diff_loss']
        runner.a_optim.zero_grad()
        (self.last_log_dist * advantage).mean().backward()
        runner.a_optim.step()
        if self.grad_clip is not None:
            self.clip_grads(runner.actor.parameters())
        runner.optimizer.step()





