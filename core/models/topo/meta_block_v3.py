#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
'''
@Target: 1: Produce basic blocks, generate neural network structure
            connection routing through searching theory algorithm.
         2: Reformulated the network structure representation form
'''

import numpy as np, networkx as nx
import torch, torch.nn as nn, torch.nn.functional as F
from core.models.common.cell_operations import SearchSpaceFactory, OPS
from ..registry import TOPO
from ..builder import build_fpn


@TOPO.register("StageTopo")
class StageTopology(nn.Module):
    '''
    Note: 1. block object aims at giving a general description
             of the generated graph structureluding node
             information and its wiring form. The search procedure
             will focus on two aspects, with respect to node operation
             selection and wiring method.

          2. self.op_params indicate the weight of the operators inside
             each node, similars as Darts.

          3. self.topo_routing indicate the routing path of whole topology
             graph, giving by the Actor.

          4. The above mentioned two parameters can completely characterize
             the structure of the a topology graph.
    :param search_space: op names
    :param NodeNum: total node numbers
    :param C_in, C_out, stride, affine, track_running_stats: config each operator layer
    '''

    def __init__(self,
                 search_space,
                 NodeNum,
                 C_in,
                 C_out,
                 stride,
                 affine,
                 track_running_stats,
                 reduction_factor,
                 fpn,
                 **kwargs
                 ):
        super(StageTopology, self).__init__()

        self.op_names = SearchSpaceFactory.get(search_space, None)
        self.nodeNum = NodeNum

        # build Topology Graph Arch.
        self.NodeInfos = nn.ModuleDict()
        self.edge_keys, self.edge2index, self.topo_routing  = [], {}, self._init_topo()
        self.idx_in, self.idx_out, self.edges= self.check_in_out(self.topo_routing)

        # Enum Nodes module.
        self._stages_range = self._cal_downsample_stages(NodeNum, reduction_factor)
        self._stages_pooling, self._stages_size = dict(), dict()

        # Build wrapper-fpn module, To prevent the problem of dim misalignment, each stage could downsampling.
        self.wrap_fpn = self._build_arch_fpn(fpn, self._stages_range, C_in, C_out, affine, track_running_stats)
        # build arch topo-block
        self._build_whole_picture(self.nodeNum, C_out, C_out, stride, affine, track_running_stats)

        # Build Node Params.
        self.op_params = nn.Parameter(
            1e-3 * torch.randn(len(self.edge_keys), len(self.op_names))
        )

    def _build_arch_fpn(self, cfg, num_stages, C_in, C_out, affine, track_running_stats):
        assert isinstance(cfg, dict)
        cfg.update({
            'num_stages': num_stages, 'C_in': C_in, 'C_out': C_out,
            'affine': affine, 'track_running_stats': track_running_stats
        })
        return build_fpn(cfg)

    def _cal_downsample_stages(self, NodeNum, reduction_factor):
        assert reduction_factor is not None
        reduction_factor.insert(0, 0)
        reduction_factor.append(NodeNum)
        _stages_range = [(reduction_factor[i], reduction_factor[i+1]) for i in range(len(reduction_factor) - 1)]
        return _stages_range

    def _build_whole_picture(self, NodesNum, c_in, c_out, stride, affine, track_running_stats):
        dag_route = np.zeros((NodesNum, NodesNum), dtype=np.int32)
        for idx, i2j in enumerate(dag_route):
            for idy, j in enumerate(i2j):
                if idy <= idx: continue
                node_str = "{:}->{:}".format(idx, idy)
                xlists = [
                    OPS[op_name](c_in, c_out, stride, affine, track_running_stats)
                    for op_name in self.op_names
                ]
                self.NodeInfos[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted((list(self.NodeInfos.keys())), key=lambda x: int(x.split('->')[1]))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.index2adjcoord = {i: [int(key.split('->')[0]), int(key.split('->')[1])] for i, key in enumerate(self.edge_keys)}


    def _init_topo(self):
        g = nx.random_graphs.watts_strogatz_graph(self.nodeNum, 4, 0.25)
        g = g.to_directed()
        dag_route = np.zeros((self.nodeNum, self.nodeNum), dtype=np.int32)
        route = np.array([[idx, n] for idx, adj_ in g.adjacency() for n in adj_.keys()])
        dag_route[tuple(route.transpose().tolist())] = 1
        return dag_route.tolist()

    def _init_topoV2(self):
        dag_route = np.ones((self.nodeNum, self.nodeNum), dtype=np.int32)
        return dag_route.tolist()

    def reset_routing(self, action):
        # Set the routing path of the topology graph
        def act2coor(_action):
            (res, _) = np.where(np.asarray(_action) > 0.0)
            coor = [self.index2adjcoord[res_] for res_ in res]
            return tuple(np.transpose(coor).tolist())
        def rot180(mat):
            return np.transpose(mat)
        assert len(action) == len(self.edge_keys)
        adj = np.zeros((self.nodeNum, self.nodeNum), dtype=np.int32)
        activate = act2coor(action)
        if activate:
            adj[act2coor(action)] = 1
            adj_ = adj + rot180(adj)
            self.topo_routing = adj_.tolist()
            self.idx_in, self.idx_out, self.edges = self.check_in_out(adj)
        else:
            print("Not Activate Any Path!")

    def get_routing(self):
        # get current routing path
        return self.topo_routing

    def set_routing(self, adj):
        print("topo: ", adj[0])
        self.topo_routing = adj
        self.idx_in, self.idx_out, self.edges = self.check_in_out(adj)

    def _in_out_parse(self, degrees):
        return [node for (node, degree) in degrees if not degree]

    def check_in_out(self, TOPO):
        # using networkx check the in and out degree of nodes in the topology graph.
        g = nx.DiGraph()
        edges = []
        for idx, i2j in enumerate(TOPO):
            for idy, j in enumerate(i2j):
                if idy <= idx or not j: continue
                edges.append((idx, idy))
        g.add_nodes_from(list(range(self.nodeNum)))
        g.add_edges_from(edges)
        idx_in = self._in_out_parse(g.in_degree())
        idx_out = self._in_out_parse(g.out_degree())
        return idx_in, idx_out, edges

    def get_observation(self):
        """
        # meta block compose observations according to l1 norm.
        :return: observations
        """
        # observations = torch.mm(self._get_topo_state(), self.op_params.detach().t()) if self.topo_routing else self.op_params
        return self.op_params.detach()

    def _check_stage(self, index):
        for i, stage in enumerate(self._stages_range):
            stage_name = str(stage[0]) + '->' + str(stage[1])
            if index in list(range(*stage)):
                return stage_name
            elif i==len(self._stages_range) - 1 and index >= stage[-1]:
                return stage_name

    def _forward_arch_fpn(self, feat):
        stages_out = self.wrap_fpn(feat)
        for idx, (keys, feat) in enumerate(stages_out.items()):
            if not self._stages_size.get(keys, False):
                self._stages_size.update({keys: feat.shape[-2:]})
                self._stages_pooling.update({keys: nn.AdaptiveAvgPool2d(feat.shape[-2:])})
        nodes = [0] * self.nodeNum
        for idx in self.idx_in:
            nodes[idx] += stages_out[self._check_stage(idx)]
        return nodes

    def _foward_node_params(self, op_params):
        return F.softmax(op_params, dim=-1)

    def _forward_arch_topo(self, nodes, alphas):
        for (in_, out_) in self.edges:
            inter_nodes = []
            node_str = "{:}->{:}".format(in_, out_)
            inter_nodes.append(
                sum(
                    layer(nodes[in_]) * w
                    for layer, w in zip(self.NodeInfos[node_str], alphas[self.edge_keys.index(node_str)])
                )
            )
            nodes[out_] += self._stages_pooling[self._check_stage(out_)](sum(inter_nodes))
        return nodes

    def _forward_node_outlet(self, nodes):
        if len(self.idx_out) == 1:
            out = nodes[self.idx_out[0]]
        else:
            out = sum(self._stages_pooling[self._check_stage(self.idx_out[-1])](nodes[idx]) for idx in self.idx_out)
        return out

    def forward(self, inputs):
        nodes = self._forward_arch_fpn(inputs)
        alphas = self._foward_node_params(self.op_params)
        nodes = self._forward_arch_topo(nodes, alphas)
        out = self._forward_node_outlet(nodes)
        return out


