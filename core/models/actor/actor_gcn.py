#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import torch.nn as nn, torch as th, torch.nn.functional as F, math
from core.models.common.gcn import GCN
from core.models.common.util import get_action_dim
from core.models.registry import ACTOR

@ACTOR.register("ActorGCN")
class ActorGCN(nn.Module):
    """
    Actor network (policy) for LSTO.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor,
        features_extractor_kwargs,
        optimizer_kwargs=None,
    ):
        super(ActorGCN, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.features_arch = self.build_net_arch(features_extractor, features_extractor_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        gcn_out_dim = features_extractor_kwargs["out_features"]
        action_dim = get_action_dim(action_space)
        self.out = self.builder_net_head(gcn_out_dim, action_dim)



    def build_net_arch(self, features_extractor, features_extractor_kwargs):
        assert features_extractor in ["GCN"], "Actor Net Arch Not Implemented."
        return GCN(observation_space=self.observation_space, total_nodes=sum(self.action_space), **features_extractor_kwargs)

    def builder_net_head(self, out_gcn_dim, action_dim):
        # return nn.Linear(sum(self.action_space) * out_gcn_dim, (action_dim[0] + action_dim[1] + action_dim[2])*2)
        return nn.Linear(sum(self.action_space) * out_gcn_dim, action_dim[0] + action_dim[1] + action_dim[2])

    def forward(self, obs, adj):
        b = self.features_arch(obs, adj)
        b = b.view(b.size(0), -1)
        # return self.activation_fn(self.out(b))
        # return F.softmax(self.out(b).view(-1, 2), dim=-1)
        return F.sigmoid(self.out(b))[0]