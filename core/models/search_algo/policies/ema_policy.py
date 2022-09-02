#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from ...registry import POLICIES
from .policy import PolicyHook

class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator

# Reward
def acc_to_adv(logs):
    top_1, top_5 = logs.get('top_1', False), logs.get('top_5', False)
    rewards = top_1 + top_5
    return rewards

def loss_to_adv(logs):
    loss = logs.get('loss', False)
    return loss
#

@POLICIES.register('EMAPolicyHook')
class EMAPolicyHook(PolicyHook):
    def __init__(self, learning_step, distribution, action_space, grad_clip, broadcast, momentum):
        super(EMAPolicyHook, self).__init__(
            learning_step,
            distribution,
            action_space,
            grad_clip,
            broadcast
        )
        self.baslines = ExponentialMovingAverage(momentum)
        self.advantages = []
        self._parse_logs = acc_to_adv

    def _call_advantages(self, runner):
        log_vars = runner.outputs.get('log_vars', None)
        rewards = log_vars.get('loss', False)
        self.baslines.update((-rewards))
        self.advantages.append((-rewards) - self.baslines.value())

    def after_train_iter(self, runner):
        self._call_advantages(runner)
        runner.rewards = self.baslines.value()
        super().after_train_iter(runner)

    def _learn(self, runner):
        # actor learning
        if hasattr(runner.actor, 'module'):
            runner.actor = runner.actor.module
        runner.a_optim.zero_grad()
        a_loss = (-self.last_log_dist * sum(self.advantages)).mean()
        a_loss.backward()
        runner.a_optim.step()
        if self.grad_clip is not None:
            self.clip_grads(runner.actor.parameters())
        runner.optimizer.step()
        self.advantages.clear()
        runner.a_loss = a_loss
