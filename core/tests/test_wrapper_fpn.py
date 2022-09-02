#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

from core.models.common import WrapperFPN
import unittest, numpy as np, os, torch

cfg = {
    'num_stages': [(0,4), (4,8),(8, 16), (16,34)],
    'C_in': 3,
    'C_out': 128,
    'stride': 1,
    'affine': False,
    'track_running_stats': True
}
input = torch.randn(1, 3, 224, 224)

class TestCritic(unittest.TestCase):
    def test_critic(self):
        wfpn = WrapperFPN(**cfg)
        with torch.no_grad():
            res = wfpn(input)
        print([res[str(stage[0]) + '->' + str(stage[1])].shape for stage in cfg['num_stages']])




if __name__ == "__main__":
    unittest.main()