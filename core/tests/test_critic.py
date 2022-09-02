#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
import unittest, numpy as np, os, torch
from core.models import builder
from core.utils.file_utils import OUTPUT_DIR, CONFIG_DIR
from tools import Config
import time

'''
cfg_file='ImageNet/sync_imagenet_stage_b8_n8_h8_c256.py'
# cfg_file='ImageNet/sync_imagenet_b8_n8_h8_c64.py'
data = {
    'img': torch.randn(400, 3, 224, 224),
    # 'gt_label': torch.randn(400, 1)
    'gt_label':None
}

cfg = Config.fromfile(os.path.join(CONFIG_DIR, cfg_file))

class TestCritic(unittest.TestCase):
    def test_critic(self):
        critic = builder.build_critic(cfg.Critic, **cfg.Space)
        critic.eval()
        for i in range(1000):
            actions = self.mute_action(cfg.Space.observation_space)
            critic.set_action(actions)
            t = time.time()
            with torch.no_grad():
                res = critic(**data)
            print(res, "activate edges: ", sum(actions), " time: ", time.time()-t)

    def mute_action(self, observation_space):
        return torch.from_numpy(np.random.randint(0, 2, (observation_space[0], 1)))
'''

cfg_file='ImageNet/sync_critic_kaimingInit_stem.py'
cfg = Config.fromfile(os.path.join(CONFIG_DIR, cfg_file))
class TestKaimingCritic(unittest.TestCase):
    def test_init(self):
        critic = builder.build_critic(cfg.Critic, **cfg.Space)


if __name__ == "__main__":
    unittest.main()