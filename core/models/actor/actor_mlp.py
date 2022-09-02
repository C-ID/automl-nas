#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import torch.nn as nn, torch as th, torch.nn.functional as F, math
from core.models.common.gcn import GCN
from core.models.common.util import get_action_dim
from core.models.registry import ACTOR


@ACTOR.register("ActorMLP")
class ActorMLP(nn.Module):
    """
    Actor network (policy) for LSTO.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
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
        super(ActorMLP, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.stem = nn.Sequential(
            nn.Linear(features_extractor_kwargs.in_features, features_extractor_kwargs.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(features_extractor_kwargs.hidden, features_extractor_kwargs.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(features_extractor_kwargs.hidden, features_extractor_kwargs.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(features_extractor_kwargs.hidden, features_extractor_kwargs.out_features),
            nn.ReLU(inplace=True),
        )

        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer, self.scheduler = None, None

    def forward(self, obs, adj):
        x = self.stem(obs)
        x = x.squeeze(0)
        return F.softmax(x, dim=-1)