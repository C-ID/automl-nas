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

@TOPO.register("AtomicTopo")
class Topology(nn.Module):
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
    in_out_parse = lambda degrees: [node for (node, degree) in degrees if not degree]

    def __init__(self,
                 search_space,
                 NodeNum,
                 C_in,
                 C_out,
                 stride,
                 affine,
                 track_running_stats,
                 ):
        super(Topology, self).__init__()

        self.op_names = SearchSpaceFactory.get(search_space, None)
        self.nodeNum = NodeNum

        # To prevent the problem of dim misalignment, each stem could upsampling or downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )

        # build Topology Graph Arch.
        self.NodeInfos = nn.ModuleDict()
        self.edge_keys, self.edge2index, self.topo_routing  = [], {}, self._init_topoV2()
        self.idx_in, self.idx_out, self.edges= self.check_in_out(self.topo_routing)
        # Enum Nodes module.
        self._build_whole_picture(self.nodeNum, C_out, C_out, stride, affine, track_running_stats)

        self.op_params = nn.Parameter(
            1e-3 * torch.randn(len(self.edge_keys), len(self.op_names))
        )

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
        idx_in = Topology.in_out_parse(g.in_degree())
        idx_out = Topology.in_out_parse(g.out_degree())
        return idx_in, idx_out, edges

    def _get_topo_state(self):
        # perform l1 norm to encoding the state of each node,
        # since l1 norm has better sparsity which could express
        # the siginificance of each node

        topo_state = torch.zeros(len(self.edge_keys), len(self.op_names))
        for idx, (str, layer) in enumerate(self.NodeInfos.items()):
            for idy, (name, params) in enumerate(layer.named_parameters()):
                l1_norm = torch.sum(torch.abs(params.detach()))
                topo_state[idx, int(name.split('.')[0])] += l1_norm
        return topo_state

    def get_observation(self):
        """
        # meta block compose observations according to l1 norm.
        :return: observations
        """
        # observations = torch.mm(self._get_topo_state(), self.op_params.detach().t()) if self.topo_routing else self.op_params
        return self.op_params.detach()

    def forward(self, inputs):
        nodes_0 = self.stem(inputs)
        nodes = [0] * self.nodeNum
        for i in self.idx_in:
            # Check whether the previous node is the entrance of the topology graph
            nodes[i] = nodes_0
        alphas = F.softmax(self.op_params, dim=-1)
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
        return sum(nodes[i] for i in self.idx_out)


