#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
import torch
from ...registry import POLICIES
from tools.runner import Hook, master_only, get_dist_info
from ...builder import build_dist, build_boardcast
from ...common import get_action_dim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad


@POLICIES.register('offPolicy')
class offPolicy(Object):
    def __init__(self, learning_step, distribution, action_space, grad_clip, broadcast):
        self.interval = learning_step
        self.distribution = build_dist(distribution)
        self.last_log_dist = None
        self.grad_clip = grad_clip
        self.broadcast = build_boardcast(broadcast)
        self.action_dims = get_action_dim(action_space)
        self.actions = torch.zeros(torch.Size([sum(self.action_dims)])).cuda()

    def _make_action(self, prob):
        actions, log_p = self.distribution.log_prob_from_params(prob)
        return actions, log_p

    def _acts_info(self, acts):
        act0 = acts[0: self.action_dims[0]]
        act1 = acts[self.action_dims[0]: self.action_dims[0] + self.action_dims[1]]
        act2 = acts[self.action_dims[0] + self.action_dims[1]:]
        return [str(sum(act0).item()) + "/" + str(self.action_dims[0]),
                str(sum(act1).item()) + "/" + str(self.action_dims[1]),
                str(sum(act2).item()) + "/" + str(self.action_dims[2])]

    def _get_action_prob(self, runner, state, adj):
        if hasattr(runner.actor, 'module'):
            runner.actor = runner.actor.module
        act_prob = runner.actor(state, adj)
        return act_prob

    def _get_observation(self, runner):
        state, adj = self.broadcast.receive(runner)
        return (state, adj)

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def _reset_critic(self, runner):
        (state, adj) = runner._get_observation(runner)
        print("before rank: {}, adj: {}, adj_shape: {}".format(runner.rank, adj, adj.shape))
        act_prob = self._get_action_prob(runner, state, adj)
        print("before rank: {}, act_prob: {}".format(runner.rank, act_prob))
        actions, self.last_log_dist = self._make_action(act_prob)
        runner.action_edges_num = self._acts_info(actions.detach())
        return actions

    # def before_train_iter(self, runner):
    #     if runner.iter == 0 or runner.is_resume:
    #         actions = self._reset_critic(runner)
    #         if actions is not None: self.actions = actions
    #         self.broadcast.send(runner, self.actions)
    #
    # def after_train_iter(self, runner):
    #     if not self.every_n_iters(runner, self.interval): return
    #     self._learn(runner)
    #     actions = self._reset_critic(runner)
    #     print("rank: {}, action: {}".format(runner.rank, actions))
    #     if actions is not None: self.actions = actions
    #     self.broadcast.send(runner, self.actions)

    def _learn(self, runner):
        raise NotImplementedError

