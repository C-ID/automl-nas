#!/usr/bin/env bash
export PYTHONPATH=/app/member/tongyaobai/Automl-NAS
PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPU_ID="1,2,3"
GPUS=3
PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=2 python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/dist_train.py $CONFIG --launcher pytorch ${@:3} --deterministic

