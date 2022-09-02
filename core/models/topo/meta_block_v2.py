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
from ..builder import build_topo




@TOPO.register("ScalableTopo")
class ScalableTopology(nn.Module):
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
                 affine,
                 track_running_stats,
                 downsample,
                 **kwargs
                 ):
        super(ScalableTopology, self).__init__()

        self.op_names = SearchSpaceFactory.get(search_space, None)
        self.nodeNum = NodeNum

        # build Topology Graph Arch.
        self.NodeInfos = nn.ModuleDict()
        self.edge_keys, self.edge2index, self.topo_routing  = [], {}, self._init_topo()
        self.idx_in, self.idx_out, self.edges= self.check_in_out(self.topo_routing)
        # Enum Nodes module.
        assert isinstance(downsample, dict)
        self.down_sample_step, self.down_sample_ratio = downsample['down_sample_step'], downsample['down_sample_ratio']
        if hasattr(downsample, "type") and downsample["type"] is 'normal':
            self._downsample_steps, self._range_steps = self._cal_downsample_step(NodeNum, self.down_sample_step)
        # Build stem module, To prevent the problem of dim misalignment, each stem could downsampling.
        self.stem = self._stem_downsample_interval(self._downsample_steps, C_in, C_out)
        self._build_whole_picture(self.nodeNum, C_out, C_out, affine, track_running_stats, self.down_sample_step, self.down_sample_ratio)

        # Build Node Params.
        self.op_params = nn.Parameter(
            1e-3 * torch.randn(len(self.edge_keys), len(self.op_names))
        )

    def _cal_downsample_step(self, NodeNum, down_sample_step):
        _downsample_steps, _range_steps = [], []
        if NodeNum // down_sample_step > 1:
            _downsample_steps += [i*down_sample_step for i in range(NodeNum // down_sample_step)]
            _range_steps += [(i*down_sample_step, (i+1)*down_sample_step) for i in range(NodeNum // down_sample_step - 1)]
        return _downsample_steps, _range_steps

    def _stem_downsample_interval(self, _downsample_steps, C_in, C_out):
        stem = nn.ModuleList()
        stem.append(
            nn.Sequential(
                nn.Conv2d(C_in, C_out//2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(C_out//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out//2, C_out//2, kernel_size=3, stride=1, bias=False),
                nn.BatchNorm2d(C_out//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out//2, C_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=True),
            )
        )
        if _downsample_steps:
            for _ in range(len(_downsample_steps)):
                stem.append(
                    nn.Sequential(
                        nn.ReLU(inplace=False),
                        nn.Conv2d(
                            C_out,
                            C_out,
                            kernel_size=1,
                            stride=2,
                            bias=True,
                        ),
                        nn.BatchNorm2d(C_out, affine=True, track_running_stats=True)
                    )
                )
        return stem


    def _build_whole_picture(self, NodesNum, c_in, c_out, affine, track_running_stats, down_sample_step, down_sample_ratio):
        dag_route = np.zeros((NodesNum, NodesNum), dtype=np.int32)
        for idx, i2j in enumerate(dag_route):
            for idy, j in enumerate(i2j):
                if idy <= idx: continue
                node_str = "{:}->{:}".format(idx, idy)
                if self._downsample_steps and (idy//down_sample_step - idx//down_sample_step) >= 1:
                    if idy > self._downsample_steps[-1]:
                        stride = down_sample_ratio ** (self._downsample_steps[-1]//down_sample_step - idx//down_sample_step)
                    else:
                        stride = down_sample_ratio ** (idy//down_sample_step - idx//down_sample_step)
                    xlists = [
                        OPS[op_name](c_in, c_out, stride, affine, track_running_stats)
                            for op_name in self.op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](c_in, c_out, 1, affine, track_running_stats)
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

    def _return_act_edge(self, router):
        self.activate_edge = [self.edge2index["->".join([str(nodes[0]), str(nodes[1])])] for nodes in router if
                              nodes[1] > nodes[0]]

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

    def _forward_arch_stems(self, feat):
        if self._downsample_steps:
            stem = [0] * len(self._downsample_steps)
            for i in range(len(self._downsample_steps)):
                feat = self.stem[i](feat)
                stem[i] += feat
        else:
            stem = [self.stem[0](feat)]
        return stem

    def _forward_arch_fpn(self, stems):
        nodes = [0] * self.nodeNum
        for idx in self.idx_in:
            # Check whether the previous node is the entrance of the topology graph
            if self._downsample_steps and self._range_steps:
                for i, range_ in enumerate(self._range_steps):
                    if idx in list(range(*range_)):
                        nodes[idx] += stems[i]
                    elif i==len(self._range_steps) - 1 and idx >= range_[-1]:
                        nodes[idx] += stems[-1]
            else:
                nodes[idx] = stems[0]
        return nodes

    def _foward_node_params(self):
        return F.softmax(self.op_params, dim=-1)

    # def _forward_node_outlet(self, nodes):
    #     pooling = nn.AdaptiveAvgPool2d(nodes[self.idx_out[-1]].shape[-2:])
    #     if len(self.idx_out) == 1:
    #         out = nodes[self.idx_out[-1]]
    #     else:
    #         out = sum(pooling(nodes[i]) for i in self.idx_out)
    #     return out

    def _forward_node_outlet(self, nodes):
        if len(self.idx_out) == 1:
            out = nodes[self.idx_out[-1]]
        else:
            minum_size = nodes[self.idx_out[-1]].shape[-2:]
            for i in self.idx_out:
                if nodes[i] is not 0:
                    minum_size = nodes[i].shape[-2:]
            pooling = nn.AdaptiveAvgPool2d(minum_size)
            out = sum(pooling(nodes[i]) for i in self.idx_out if nodes[i] is not 0)
        return out

    def forward(self, inputs):
        stems = self._forward_arch_stems(inputs)
        nodes = self._forward_arch_fpn(stems)
        alphas = self._foward_node_params()
        for (in_, out_) in self.edges:
            inter_nodes = []
            node_str = "{:}->{:}".format(in_, out_)
            inter_nodes.append(
                sum(
                    layer(nodes[in_]) * w
                    for layer, w in zip(self.NodeInfos[node_str], alphas[self.edge_keys.index(node_str)])
                )
            )
            nodes[out_] += sum(inter_nodes)
        return self._forward_node_outlet(nodes)


