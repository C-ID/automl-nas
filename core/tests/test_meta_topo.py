#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import unittest
from core.models import builder
import numpy as np, os, torch

cls_head=dict(
    type='AtomicTopo',
    NodeNum=8,
    C_in=32,
    C_out=32,
    stride=2,
    affine=True,
    track_running_stats=True,
    search_space="nas-bench-201"
)
action =      [1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
action_list = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
action_list2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
input = torch.randn(1, 32, 300, 300)

class TestTopoArch(unittest.TestCase):
    def test_meta_block(self):
        topo = builder.build_topo(cls_head)
        topo.reset_routing(action_list)
        print(topo.topo_routing)
        print(topo.idx_in, topo.idx_out)
        res = topo(input)
        # print(res)




if __name__ == "__main__":
    unittest.main()
