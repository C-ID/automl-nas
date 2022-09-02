#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
from __future__ import division

import os, os.path as osp, time, torch, argparse
from core.api.trainner import train_algorithm
from core.utils.file_utils import OUTPUT_DIR, CONFIG_DIR
from core.utils.logger import get_root_logger
from core.models.common.util import set_random_seed
from core.dataset import build_dataset
from tools import Config
from tools import mkdir_or_exist
from tools.runner import init_dist
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(os.path.join(CONFIG_DIR, args.config))
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mkdir_or_exist(osp.join(OUTPUT_DIR, cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(OUTPUT_DIR, cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, using_cuda=args.deterministic)

    meta = dict()
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    datasets = [build_dataset(cfg.data.train, cfg.data.type)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val, cfg.data.type))

    # add an attribute for visualization convenience
    train_algorithm(
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()


# bash dist_train.sh normal_cls.py 2
# bash dist_train.sh ImageNet16-120/sync_s4_b34_n16_h8.py
# bash dist_train.sh ImageNet/sync_imagnet_b8_n32_h16.py
# bash dist_train.sh ImageNet/sync_imagenet_b8_n8_h8_c64.py
# bash dist_train.sh ImageNet/sync_imagenet_stage_b8_n8_h8_c256.py