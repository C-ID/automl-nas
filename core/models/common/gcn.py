'''
modified from https://github.com/tkipf/pygcn/blob/master/pygcn/models.py

The MIT License

Copyright (c) 2017 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restrictionluding without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIEDLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import torch.nn as nn, torch as th, math
import torch.nn.functional as F

from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden, out_features, total_nodes, observation_space, **kwargs):
        super(GCN, self).__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.weight1 = nn.Parameter(th.Tensor(observation_space[1], total_nodes))
        self.bias1 = nn.Parameter(th.Tensor(total_nodes))
        self.weight2 = nn.Parameter(th.Tensor(observation_space[0], observation_space[1]))
        self.bias2 = nn.Parameter(th.Tensor(observation_space[1]))
        for i in range(n_layers-1):
            in_channels = in_features if i == 0 else hidden
            self.layers.append(GraphConvolution(in_channels, hidden))
        self.last_layer = GraphConvolution(hidden, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
        if self.bias2 is not None:
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        x = th.matmul(x, self.weight1)
        x += self.bias1
        x = x.permute(0, 2, 1)
        x = th.matmul(x, self.weight2)
        x += self.bias2

        for l in self.layers:
            x = F.relu(l(x, adj))
        x = self.last_layer(x, adj)
        x = F.relu(x)
        return x

