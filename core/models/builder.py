#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from core.models import ACTOR, CRITIC, TOPO, CELL, FPN, RUNNERHOOKS, POLICIES, LOSSES, DISTRIBUTIONS, BOARDCASTS
from core.utils.registry import build_from_cfg

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_actor(cfg, observation_space, action_space):
    return build(cfg, ACTOR, dict(observation_space=observation_space, action_space=action_space))

def build_critic(cfg, observation_space, action_space):
    return build(cfg, CRITIC, dict(observation_space=observation_space, action_space=action_space))

def build_policy(cfg):
    return build(cfg, POLICIES)

def build_norm_cell(cfg):
    return build(cfg, CELL)

def build_topo(cfg):
    return build(cfg, TOPO)

def build_fpn(cfg):
    return build(cfg, FPN)

def build_loss(cfg):
    return build(cfg, LOSSES)

def build_dist(cfg):
    return build(cfg, DISTRIBUTIONS)

def build_boardcast(cfg):
    return build(cfg, BOARDCASTS)

def build_runner_hook(cfg, model, batch_processor, optimizer, work_dir, logger):
    return build(cfg, RUNNERHOOKS, dict(model=model,
                                        batch_processor=batch_processor,
                                        optimizer=optimizer,
                                        work_dir=work_dir,
                                        logger=logger))

