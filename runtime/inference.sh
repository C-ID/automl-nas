#!/usr/bin/env bash
export PYTHONPATH=/app/member/tongyaobai/Automl-NAS
CUDA_VISIBLE_DEVICES=1 python3 inference.py --config_file normal_cls.py --ckpt "ImageNet16-120/sync_lsto/best_arch_checkpoint.pth"
unset