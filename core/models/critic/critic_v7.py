#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import torch.nn as nn, copy, torch
import numpy as np, networkx as nx
from core.models.registry import CRITIC
from .. import builder
from ..common import get_action_dim
from tools.cnn import kaiming_init, constant_init, normal_init
from core.models.common.cell_operations import Identity

def WattsStrogatz(nodeNum):
    g = nx.random_graphs.watts_strogatz_graph(nodeNum, 4, 0.25)
    g = g.to_directed()
    dag_route = np.zeros((nodeNum, nodeNum), dtype=np.int32)
    route = np.array([[idx, n] for idx, adj_ in g.adjacency() for n in adj_.keys()])
    dag_route[tuple(route.transpose().tolist())] = 1
    return dag_route.tolist()

@CRITIC.register("FlexibleCritic")
class FlexibleCritic(nn.Module):
    """
        Critic network.

        :param observation_space: Obervation space
        :param action_space: Action space
        :param config: configure critic network.
        """

    def __init__(
            self,
            backbone,
            reduction_bn,
            neck,
            reduction_nh,
            cls_head,
            num_classes,
            observation_space,
            action_space,
            loss_func,
    ):
        super(FlexibleCritic, self).__init__()
        assert isinstance(loss_func, dict)
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dims = get_action_dim(action_space)

        self.stem = nn.Sequential(
            nn.Conv2d(3, backbone.C_in, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(backbone.C_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone.C_in, backbone.C_in, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(backbone.C_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone.C_in, backbone.C_in, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(backbone.C_in),
            nn.ReLU(inplace=True)
        )

        self.arch = self._build_arch(
            backbone,
            reduction_bn,
            neck,
            reduction_nh,
            cls_head
        )

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(cls_head.C_out),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(cls_head.C_out, num_classes)
        self.loss = builder.build_loss(loss_func.get('train', None))
        self.acc = builder.build_loss(loss_func.get('eval'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def init_random_topo(self):
        for (name, module) in self.arch.items():
            if name == "backbone":
                adj = WattsStrogatz(self.action_dims[0])
                module.set_routing(adj)
            elif name == 'neck':
                adj = WattsStrogatz(self.action_dims[1])
                module.set_routing(adj)
            elif name == 'head':
                adj = WattsStrogatz(self.action_dims[2])
                module.set_routing(adj)
            else:
                continue

    def _build_arch(self, backbone, reduction_bn, neck, reduction_nh, cls_head):
        # init total network
        assert isinstance(backbone, dict) and isinstance(reduction_bn, dict) and isinstance(neck, dict) and \
               isinstance(reduction_nh, dict) and isinstance(cls_head, dict)

        return nn.ModuleDict({
            "backbone": self._build_upper_module(backbone) if backbone.NodeNum else self._build_none_block(backbone),
            "bridge1": self._build_bridge_block(reduction_bn),
            "neck": self._build_upper_module(neck) if neck.NodeNum else self._build_none_block(neck),
            "bridge2": self._build_bridge_block(reduction_nh),
            "head": self._build_upper_module(cls_head) if cls_head.NodeNum else self._build_none_block(cls_head)
        })

    def _build_upper_module(self, _config):
        # Note that this func aim at building basis-block, every block has constructed a
        # topological graphs which composed of several nodes according to config.
        return builder.build_topo(_config)

    def _build_bridge_block(self, _config):
        # build a bridging layer between each topology graph.
        # down-sampling.
        return builder.build_norm_cell(_config)

    def _build_none_block(self, _config):
        # build a none block, with inference input directly
        assert _config.NodeNum is 0, "NodeNum is not 0, Can not build None-Block"
        return Identity()

    def get_observation_state(self):
        # Tracking and Evaluating the Critic parameters (i.e. node params and op params),
        # then transformed into the observation space which will help Actor net to getting
        # next action.
        return (self._compose_observation_state(
                    self._get_tp_state('backbone'),
                    self._get_tp_state('neck'),
                    self._get_tp_state('head')),
                self._compose_adj_state(
                    self._get_tp_adj('backbone'),
                    self._get_tp_adj('neck'),
                    self._get_tp_adj('head'))
        )

    def _compose_observation_state(self, state_backbone, state_neck, state_head):
        #compose each tp graph observation to a batch tensor.
        return torch.cat([state_backbone, state_neck, state_head]).unsqueeze(0).cuda()

    def _compose_adj_state(self, adj_backbone, adj_neck, adj_head):
        #compose each tp graph observation to a batch tensor.
        adj = torch.zeros(len(adj_backbone) + len(adj_neck) + len(adj_head),
                          len(adj_backbone) + len(adj_neck) + len(adj_head))
        adj[0:len(adj_backbone), 0:len(adj_backbone)] = torch.tensor(adj_backbone)
        adj[len(adj_backbone):len(adj_backbone) + len(adj_neck),
        len(adj_backbone):len(adj_backbone) + len(adj_neck)] = torch.tensor(adj_neck)
        adj[len(adj_backbone) + len(adj_neck):, len(adj_backbone) + len(adj_neck):] = torch.tensor(adj_head)
        return adj.unsqueeze(0).cuda()

    def _get_tp_state(self, which_tp):
        assert which_tp in ["backbone", "neck", "head"], \
            "Out of observation range, just obtain the state of topology graph."
        if self.arch[which_tp]._get_name() is not "Identity":
            return None
        else:
            return self.arch[which_tp].get_observation()

    def _get_tp_adj(self, which_tp):
        assert which_tp in ["backbone", "neck", "head"], \
            "Out of observation range, just obtain the adj of topology graph."
        if self.arch[which_tp]._get_name() is not "Identity":
            return None
        else:
            return self.arch[which_tp].get_routing()

    def set_action(self, action_tensor):
        # Decompose the action value into the circulation path of the
        # topological graph(each block. i.e. backbone tp, neck tp, head tp).
        self._decompose_action(action_tensor)

    def _split_acts(self, acts):
        act0 = acts[0: self.action_dims[0]]
        act1 = acts[self.action_dims[0]: self.action_dims[0] + self.action_dims[1]]
        act2 = acts[self.action_dims[0] + self.action_dims[1]:]
        return (act0, act1, act2)

    def _decompose_action(self, action_tensor):
        # Decompose the action and send it to the corresponding topology.
        def _set_tp_action(which_tp, adj):
            assert which_tp in ["backbone", "neck", "head"], \
                "Out of observation range, just set the routing path of topology graph."
            self.arch[which_tp].reset_routing(adj)
        (backbone, neck, head) = self._split_acts(action_tensor)
        _set_tp_action("backbone", backbone.detach().view(-1, 1).tolist())
        _set_tp_action("neck", neck.detach().view(-1, 1).tolist())
        _set_tp_action("head", head.detach().view(-1, 1).tolist())

    def _decompose_actionV2(self, action_tensor):
        # Decompose the action and send it to the corresponding topology.
        def _set_tp_action(which_tp, adj):
            assert which_tp in ["backbone", "neck", "head"], \
                "Out of observation range, just set the routing path of topology graph."
            if self.arch[which_tp]._has_name():pass
            self.arch[which_tp].reset_routing(adj)
        (backbone, neck, head) = self._split_acts(action_tensor)
        _set_tp_action("backbone", backbone.detach().view(-1, 1).tolist())
        _set_tp_action("neck", neck.detach().view(-1, 1).tolist())
        _set_tp_action("head", head.detach().view(-1, 1).tolist())

    def get_last_routing(self):
        # Action value is not generated at every step,
        # so we need to save the last action value so far.
        return {"backbone_topo": self.arch["backbone"].get_routing(),
                "neck_topo": self.arch["neck"].get_routing(),
                "head_topo": self.arch["head"].get_routing()}

    def set_last_routing(self, save_params):
        def _set_tp_action(which_tp, adj):
            assert which_tp in ["backbone", "neck", "head"], \
                "Out of observation range, just set the routing path of topology graph."
            self.arch[which_tp].set_routing(adj)
        assert len(save_params) == 3
        for which_tp, params in save_params.items():
            _set_tp_action(which_tp.split("_")[0], params)

    def _predict(self, image):
        feat = self.stem(image)
        feat = self.arch["backbone"](feat)
        feat = self.arch["bridge1"](feat)
        feat = self.arch["neck"](feat)
        feat = self.arch["bridge2"](feat)
        feat = self.arch["head"](feat)
        out = self.lastact(feat)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits

    def _forward_test(self, img, **kwargs):
        return {"pred": self._predict(img)}

    def _forward_train(self, img, label, **kwargs):
        pred = self._predict(img)
        loss = self.loss(pred, label)
        top_1, top_5 = self.acc(pred, label)
        return {'loss': loss, 'top_1': top_1, 'top_5': top_5}

    def forward(self, img, gt_label, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        data have label. Note this setting will change the expected inputs.
        """
        if gt_label is not None:
            return self._forward_train(img, gt_label)
        else:
            return self._forward_test(img)

