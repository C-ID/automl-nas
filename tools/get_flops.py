#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai


import argparse, os

from tools import Config
from core.utils.file_utils import CONFIG_DIR
from core.models.builder import build_critic
from tools.cnn.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(os.path.join(CONFIG_DIR, args.config))
    model = build_critic(
        cfg.Critic, cfg.Space.observation_space, cfg.Space.action_space).cuda()
    model.eval()

    if hasattr(model, '_predict'):
        model.forward = model._predict
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))


    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=1 python3 get_flops.py ImageNet16-120/sync_s4_b34_n16_h8.py --shape 224 224
# CUDA_VISIBLE_DEVICES=1 python3 get_flops.py ImageNet/sync_imagnet_b8_n32_h16.py --shape 224 224
# CUDA_VISIBLE_DEVICES=1 python3 get_flops.py ImageNet/sync_imagenet_stage_b8_n8_h8_c256.py --shape 224 224
# CUDA_VISIBLE_DEVICES=1 python3 get_flops.py ImageNet16-120/scable_topo_test.py --shape 16 16