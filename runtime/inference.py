#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import numpy as np
import logging
import os
import argparse
import torch, torch.nn as nn
from core.utils.file_utils import OUTPUT_DIR, CONFIG_DIR
from core.models import builder, Accuracy
from core.dataset.builder import build_dataset
from core.dataset import build_dataloader
from tools.parallel import MMDataParallel
from core.utils.logger import AverageMeter
from tools import Config, ProgressBar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="cls_type.py",
        help="Multiple models for benchmarking (separate with comma)",
    )
    parser.add_argument("--ckpt", type=str, default="ImageNet16-120/sync_lsto/last_checkpoint.pth", help="Folder to save checkpoints and log.")
    args = parser.parse_args()
    return args

def load_ckpt(path):
    assert os.path.isfile(path)
    infos = torch.load(path)
    return infos.get("critic", False), infos.get("last_routing", False)

def single_gpu_test(
        model,
        data_loader
):
    acc = Accuracy(topk=(1, 5))
    arch_top1, arch_top5 = AverageMeter(), AverageMeter()
    prog_bar = ProgressBar(len(data_loader.dataset))
    model.eval()
    for i, data in enumerate(data_loader):
        img = data['img']
        label = data['gt_label']
        with torch.no_grad():
            res = model(img=img, gt_label=None)
        top_1, top_5 = acc(res["pred"], label.cuda())
        arch_top1.update(top_1.item(), img.size(0))
        arch_top5.update(top_5.item(), img.size(0))
        batch_size = img.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    print("\ntop_1: {}, top_5: {}".format(arch_top1.avg, arch_top5.avg))

def main():
    args = parse_args()
    cfg = Config.fromfile(os.path.join(CONFIG_DIR, args.config_file))

    # build dataset
    dataset = build_dataset(cfg.data.test, cfg.data.type)
    # build loader
    data_loader = build_dataloader(
                    dataset,
                    cfg.data.imgs_per_gpu,
                    cfg.data.workers_per_gpu,
                    dist=False,
                    round_up=True)
    # build net
    model = builder.build_critic(cfg.Critic, cfg.Space.observation_space, cfg.Space.action_space)
    path = os.path.join(OUTPUT_DIR, args.ckpt)
    state_info, last_routing = load_ckpt(path)

    model.set_last_routing(last_routing)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_info)

    single_gpu_test(model.cuda(), data_loader)

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 python3 inference.py --config_file normal_cls.py --ckpt "ImageNet16-120/sync_lsto/best_arch_checkpoint.pth"
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --config_file async_actor.py --ckpt "ImageNet16-120/async_actor/best_arch_checkpoint.pth"