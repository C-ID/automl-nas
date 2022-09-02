#!/usr/bin/env python3
# Copyright (c) Tongyao Bai and its affiliates. All Rights Reserved
# @Author: Tongyao Bai
# Copy from mmdet, made some

import random
import re
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist

from core.utils.file_utils import OUTPUT_DIR
from tools.parallel import MMDataParallel, MMDistributedDataParallel
from tools.runner import DistSamplerSeedHook, obj_from_dict, DistOptimizerHook
from core.models import builder, async1v3Runner, async3v3Runner
from core.dataset import DATASETS_CLS, DATASETS_DET, DATASETS_PCD, DATASETS_SEG, build_dataloader
from core.utils.logger import get_root_logger
from core.evaluation import EvalHook, DistEvalHook

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img']))

    return outputs


def train_algorithm(dataset,
                   cfg,
                   distributed=False,
                   validate=True,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta,
            distributed=distributed
        )
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta,
            distributed = distributed
        )


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(dataset,
                cfg,
                validate=True,
                logger=None,
                timestamp=None,
                meta=None,
                distributed=True):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dist_type = cfg.Policy.broadcast.type
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=distributed,
            round_up=False,
            seed=meta['seed'],
            dist_type=dist_type) for ds in dataset
    ]
    # put model on gpus
    if cfg.get('Critic', False):
        critic = builder.build_critic(cfg.Critic, **cfg.Space)
    else:
        raise Error('No Critic Config!!!')
    if cfg.get('Actor', False):
        actor = builder.build_actor(cfg.Actor, **cfg.Space)
    else:
        actor = None
    model = {
        'actor': MMDistributedDataParallel(actor.cuda()) if actor else None,
        'critic': MMDistributedDataParallel(critic.cuda()),
    }
    # build runner
    c_optim = build_optimizer(critic, cfg.c_optim.optimizer)
    a_optim = build_optimizer(actor, cfg.a_optim.optimizer) if actor else None
    optimizer = {
        'a_optim': a_optim,
        'c_optim': c_optim,
    }
    if dist_type == 'one2many':
        runner = async1v3Runner(model, batch_processor, optimizer, cfg.work_dir, logger, meta)
    elif dist_type == 'many2many':
        runner = async3v3Runner(model, batch_processor, optimizer, cfg.work_dir, logger, meta)
    else:
        raise Error("runner type error")
        # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    #fp16 setting
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
    #                                          **fp16_cfg)
    # else:
    #     optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    if cfg.get('Policy', False):
        policy_config = builder.build_policy(cfg.Policy)
    else:
        policy_config = None

    optimizer_config = DistOptimizerHook(**cfg.c_optim.optimizer_config)
    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config, policy_config)
    runner.register_hook(DistSamplerSeedHook())
    # Refers to https://github.com/open-mmlab/mmcv/issues/1261
    # register eval hooks
    # if validate:
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_cfg['by_epoch'] = True
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     # `EvalHook` needs to be executed after `IterTimerHook`.
    #     # Otherwise, it will cause a bug if use `IterBasedRunner`.
    #     # Refers to https://github.com/open-mmlab/mmcv/issues/1261
    #     runner.register_hook(
    #         eval_hook(data_loaders[1], **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):
    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, logger=logger)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
