#!/usr/bin/env python3
# Copyright (c) Tongyao Bai. and its affiliates. All Rights Reserved
# @Author: Tongyao Bai

import torch, torch.nn as nn
from ..registry import FPN

@FPN.register("wrapperFPN")
class WrapperFPN(nn.Module):
    '''
    @ WrapperFpn make downsampling for each topo-block.
    '''
    def __init__(self,
                 num_stages,
                 C_in,
                 C_out,
                 stride,
                 affine,
                 track_running_stats
                ):
        super(WrapperFPN, self).__init__()
        self.num_stages = num_stages
        self.stride = stride
        self.fpn_module = self._build_arch_fpn(num_stages, C_in, C_out, stride, affine, track_running_stats)

    def _build_fpn_layer(self, c_in, c_out, stride, affine, track_running_stats, downsampling=False):
        if downsampling and stride==2:
            out = nn.Sequential(
                nn.Conv2d(
                    c_in, c_out,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=not affine
                ),
                nn.BatchNorm2d(c_out, track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
            )
        elif downsampling and stride==1:
            out = nn.Sequential(
                    nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
                    nn.Conv2d(
                        c_in, c_out,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=not affine
                    ),
                    nn.BatchNorm2d(c_out, track_running_stats=track_running_stats),
                    nn.ReLU(inplace=True),
                )
        else:
            out = nn.Sequential(
                    nn.Conv2d(
                        c_in, c_out,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=not affine
                    ),
                    nn.BatchNorm2d(c_out, track_running_stats=track_running_stats),
                    nn.ReLU(inplace=True),
            )
        return out

    def _build_arch_fpn(self, stages, C_in, C_out, stride, affine, track_running_stats):
        fpn_module = nn.ModuleDict()
        for i, stage in enumerate(stages):
            stage_name = str(stage[0]) + '->' + str(stage[1])
            if i == 0:
                fpn_module[stage_name] = self._build_fpn_layer(C_in, C_out, 1, affine, track_running_stats, False)
            elif i == len(stages) - 1 and len(stage) > 2:
                fpn_module[stage_name] = self._build_fpn_layer(C_out, C_out, 1, affine, track_running_stats, False)
            else:
                fpn_module[stage_name] = self._build_fpn_layer(C_out, C_out, stride, affine, track_running_stats, True)
        return fpn_module

    def forward(self, inputs):
        out = {}
        for stage in self.num_stages:
            stage_name = str(stage[0]) + '->' + str(stage[1])
            inputs = self.fpn_module[stage_name](inputs)
            out.update({stage_name: inputs})
        return out




