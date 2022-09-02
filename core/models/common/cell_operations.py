##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import core.utils.registry as registry
from core.models.registry import CELL

OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
    "nor_conv_7x7": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (2, 2),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "dil_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (4, 4),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity() \
                            if stride == 1 and C_in == C_out \
                            else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
    "meta_dynamic_cell": lambda C_in, C_out, stride, affine, track_running_stats, kernel_size, act_func_list: DynamicMetaLayer(
        C_in,
        C_out,
        kernel_size,
        stride,
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
        act_func_list,
    ),

    "ResNetBasicblock":lambda C_in, C_out, stride, affine, track_running_stats: ResNetBasicblock(
            C_in,
            C_out,
            stride,
            affine,
            track_running_stats
    )
}

ACTIVATE_FUNC = ['relu', 'h_swish', 'h_sigmoid']
ACTIVATE_FUNC2 = ['relu', 'relu6', 'tanh', 'sigmoid', 'h_swish', 'h_sigmoid', 'none']

CONNECT_NAS_BENCHMARK = ["none", "skip_connect", "nor_conv_3x3"]

NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
MARCO_MICRO_SIZE = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3", "meta_dynamic_cell"]

DARTS_SPACE = [
    "none",
    "skip_connect",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "avg_pool_3x3",
    "max_pool_3x3",
]

SearchSpaceNames = {
    "connect-nas": CONNECT_NAS_BENCHMARK,
    "nats-bench": NAS_BENCH_201,
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
    "bind-space": MARCO_MICRO_SIZE,
    "activate-func": ACTIVATE_FUNC,
    "activate-func2": ACTIVATE_FUNC2
}

SearchSpaceFactory = registry.Registry("SearchSpaceNames")
SearchSpaceFactory.register_dict(SearchSpaceNames)


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.part = int(np.log2(stride[0]))
        if stride[0] == 1:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    C_in,
                    C_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not affine,
                ),
                nn.BatchNorm2d(
                    C_out, affine=affine, track_running_stats=track_running_stats
                ),
            )
        else:
            self.op = nn.ModuleList()
            for i in range(self.part - 1):
                self.op.append(
                    nn.Sequential(
                        nn.ReLU(inplace=False),
                        nn.Conv2d(
                            C_in,
                            C_out,
                            1,
                            stride=2,
                            bias=not affine,
                        ),
                        nn.BatchNorm2d(
                            C_out, affine=affine, track_running_stats=track_running_stats
                        )
                    )
                )
            self.op.append(
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
                    nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, dilation=dilation, bias=not affine),
                    nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
                )
            )

    def forward(self, x):
        if self.part:
            for i in range(self.part):
                x = self.op[i](x)
            return x
        else:
            return self.op(x)


class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats
        )

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x

@CELL.register("ResNetBasicblock")
class ResNetBasicblock(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            C_in, C_out, 3, (stride, stride), 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            C_out, C_out, 3, (1, 1), 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False
                )
            )
        elif C_in != C_out:
            self.downsample = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = C_in
        self.out_dim = C_out
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        self.part = int(np.log2(stride))
        if mode == "avg":
            if stride == 1:
                self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
            else:
                self.op = nn.ModuleList()
                for _ in range(self.part):
                    self.op.append(
                        nn.Sequential(
                            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
                            nn.Conv2d(
                                C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False
                            )
                        )
                    )
        elif mode == "max":
            if stride == 1:
                self.op = nn.MaxPool2d(3, stride=stride, padding=1)
            else:
                self.op = nn.ModuleList()
                for _ in range(self.part):
                    self.op.append(
                        nn.Sequential(
                            nn.MaxPool2d(3, stride=2, padding=1, count_include_pad=False),
                            nn.Conv2d(
                                C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False
                            )
                        )
                    )
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        if self.part:
            for i in range(self.part):
                x = self.op[i](x)
        else:
            x = self.op(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.part = int(np.log2(stride))
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            self.convs = nn.ModuleList()
            C_outs = [C_out // 2, C_out - C_out // 2]
            for _ in range(self.part):
                for i in range(2):
                    self.convs.append(
                        nn.Conv2d(
                            C_in, C_outs[i], 1, stride=2, padding=0, bias=not affine
                        )
                    )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        elif self.stride == 1:
            out = self.conv(x)
        else:
            for i in range(self.part):
                x = self.relu(x)
                y = self.pad(x)
                x = torch.cat([self.convs[i*2](x), self.convs[i*2+1](y[:, :, 1:, 1:])], dim=1)
            out = x
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):
    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(C_in, self.hidden, 1),
                    nn.BatchNorm2d(self.hidden, affine=True),
                )
            )
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)

        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError("Invalid Stride : {:}".format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, "input size too small : {:} vs {:}".format(
            x.shape, self.part
        )
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i + 1) * (float(H) / self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i] : IHs[i + 1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(local_feat_list, dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(attention, part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i : i + 1].expand(
                batch, self.hidden, IHs[i + 1] - IHs[i]
            )
            feature = feature.view(batch, self.hidden, IHs[i + 1] - IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x.mul_(mask)
    return x


# Searching for A Robust Neural Architecture in Four GPU Hours
class GDAS_Reduction_Cell(nn.Module):
    def __init__(
        self, C_prev_prev, C_prev, C, reduction_prev, affine, track_running_stats
    ):
        super(GDAS_Reduction_Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(
                C_prev_prev, C, 2, affine, track_running_stats
            )
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, 1, affine, track_running_stats
            )
        self.preprocess1 = ReLUConvBN(
            C_prev, C, 1, 1, 0, 1, affine, track_running_stats
        )

        self.reduction = True
        self.ops1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        C,
                        C,
                        (1, 3),
                        stride=(1, 2),
                        padding=(0, 1),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.Conv2d(
                        C,
                        C,
                        (3, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        C,
                        C,
                        (1, 3),
                        stride=(1, 2),
                        padding=(0, 1),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.Conv2d(
                        C,
                        C,
                        (3, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
            ]
        )

        self.ops2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
                nn.Sequential(
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
            ]
        )

    @property
    def multiplier(self):
        return 4

    def forward(self, s0, s1, drop_prob=-1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        X0 = self.ops1[0](s0)
        X1 = self.ops1[1](s1)
        if self.training and drop_prob > 0.0:
            X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

        # X2 = self.ops2[0] (X0+X1)
        X2 = self.ops2[0](s0)
        X3 = self.ops2[1](s1)
        if self.training and drop_prob > 0.0:
            X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
        return torch.cat([X0, X1, X2, X3], dim=1)


# To manage the useful classes in this file.
RAW_OP_CLASSES = {"gdas_reduction": GDAS_Reduction_Cell}

# Add once-for-all Micro Search Speace, i.e. Dynamic Conv filiter transform.
# meta function
def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)

def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_channels if isinstance(target_bn, nn.GroupNorm) else target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])

def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]

def build_activation(act_func, inplace=False):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

# meta-cell, dynamic search layer
class DynamicMetaLayer(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
        act_func_list=None,
    ):
        super(DynamicMetaLayer, self).__init__()
        assert isinstance(kernel_size, list) and kernel_size[0] <=7 and kernel_size[1] <=7, "Invalid kernel_size"
        assert isinstance(act_func_list, list) and len(act_func_list) <=3, 'large meta cell size'
        # self.op = nn.ModuleList()
        act = build_activation(act_func_list[0])
        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
        self.op = nn.Sequential(
                            act,
                            nn.Conv2d(
                                C_in,
                                C_out,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=not affine,
                            ),
                            nn.BatchNorm2d(
                                C_out, affine=affine, track_running_stats=track_running_stats
                            )
        )

    def forward(self, x):
        return self.op(x)


# meta op
class MyConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.WS_EPS
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(MyConv2d, self).forward(x)
        else:
            return F.conv2d(x, self.weight_standardization(self.weight), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        return super(MyConv2d, self).__repr__()[:-1] + ', ws_eps=%s)' % self.WS_EPS

# activation func
class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hsigmoid()'

class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction, divisor=8)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return 'SE(channel=%d, reduction=%d)' % (self.channel, self.reduction)

# DSC operator
class DynamicSeparableConv2d(nn.Module):

    def __init__(self, max_in_channels, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = [3,5,7]
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]

        # register scaling parameters
        # 7to5_matrix, 5to3_matrix
        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            # noinspection PyArgumentList
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, in_channel
        )
        return y

class DynamicConv2d(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicGroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size_list, groups_list, stride=1, dilation=1):
        super(DynamicGroupConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, max(self.kernel_size_list), self.stride,
            groups=min(self.groups_list), bias=False,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = self.in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start:start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        filters = self.get_active_filter(kernel_size, groups).contiguous()
        padding = get_same_padding(kernel_size)
        filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, groups,
        )
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicGroupNorm(nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=None):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps)

    @property
    def bn(self):
        return self


class DynamicSE(SEModule):

    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1)
            return torch.cat([
                sub_filter[:, :sub_in_channels, :, :] for sub_filter in sub_filters
            ], dim=1)

    def get_active_reduce_bias(self, num_mid):
        return self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0)
            return torch.cat([
                sub_filter[:sub_in_channels, :, :, :] for sub_filter in sub_filters
            ], dim=0)

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.bias[:in_channel] if self.fc.expand.bias is not None else None
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat([
                sub_bias[:sub_in_channels] for sub_bias in sub_bias_list
            ], dim=0)

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(num_mid, in_channel, groups=groups).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(num_mid, in_channel, groups=groups).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicLinear(nn.Module):

    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y

class DynamicMBConvLayer(nn.Module):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list)), 8)
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_middle_channel, self.stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), 8)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)

    @property
    def config(self):
        return {
            'name': DynamicMBConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(round(in_channel * self.active_expand_ratio), 8)

    ############################################################################################

    # def get_active_subnet(self, in_channel, preserve_weight=True):
    #     # build the new layer
    #     sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
    #     sub_layer = sub_layer.to(get_net_device(self))
    #     if not preserve_weight:
    #         return sub_layer
    #
    #     middle_channel = self.active_middle_channel(in_channel)
    #     # copy weight from current layer
    #     if sub_layer.inverted_bottleneck is not None:
    #         sub_layer.inverted_bottleneck.conv.weight.data.copy_(
    #             self.inverted_bottleneck.conv.get_active_filter(middle_channel, in_channel).data,
    #         )
    #         copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)
    #
    #     sub_layer.depth_conv.conv.weight.data.copy_(
    #         self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
    #     )
    #     copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)
    #
    #     if self.use_se:
    #         se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
    #         sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
    #             self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).data
    #         )
    #         sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
    #             self.depth_conv.se.get_active_reduce_bias(se_mid).data
    #         )
    #
    #         sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
    #             self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).data
    #         )
    #         sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
    #             self.depth_conv.se.get_active_expand_bias(middle_channel).data
    #         )
    #
    #     sub_layer.point_linear.conv.weight.data.copy_(
    #         self.point_linear.conv.get_active_filter(self.active_out_channel, middle_channel).data
    #     )
    #     copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)
    #
    #     return sub_layer
    #
    # def get_active_subnet_config(self, in_channel):
    #     return {
    #         'name': MBConvLayer.__name__,
    #         'in_channels': in_channel,
    #         'out_channels': self.active_out_channel,
    #         'kernel_size': self.active_kernel_size,
    #         'stride': self.stride,
    #         'expand_ratio': self.active_expand_ratio,
    #         'mid_channels': self.active_middle_channel(in_channel),
    #         'act_func': self.act_func,
    #         'use_se': self.use_se,
    #     }
    #
    # def re_organize_middle_weights(self, expand_ratio_stage=0):
    #     importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
    #     if isinstance(self.depth_conv.bn, DynamicGroupNorm):
    #         channel_per_group = self.depth_conv.bn.channel_per_group
    #         importance_chunks = torch.split(importance, channel_per_group)
    #         for chunk in importance_chunks:
    #             chunk.data.fill_(torch.mean(chunk))
    #         importance = torch.cat(importance_chunks, dim=0)
    #     if expand_ratio_stage > 0:
    #         sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
    #         sorted_expand_list.sort(reverse=True)
    #         target_width_list = [
    #             make_divisible(round(max(self.in_channel_list) * expand), 8)
    #             for expand in sorted_expand_list
    #         ]
    #
    #         right = len(importance)
    #         base = - len(target_width_list) * 1e5
    #         for i in range(expand_ratio_stage + 1):
    #             left = target_width_list[i]
    #             importance[left:right] += base
    #             base += 1e5
    #             right = left
    #
    #     sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    #     self.point_linear.conv.conv.weight.data = torch.index_select(
    #         self.point_linear.conv.conv.weight.data, 1, sorted_idx
    #     )
    #
    #     adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
    #     self.depth_conv.conv.conv.weight.data = torch.index_select(
    #         self.depth_conv.conv.conv.weight.data, 0, sorted_idx
    #     )
    #
    #     if self.use_se:
    #         # se expand: output dim 0 reorganize
    #         se_expand = self.depth_conv.se.fc.expand
    #         se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
    #         se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
    #         # se reduce: input dim 1 reorganize
    #         se_reduce = self.depth_conv.se.fc.reduce
    #         se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
    #         # middle weight reorganize
    #         se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
    #         se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)
    #
    #         se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
    #         se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
    #         se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)
    #
    #     if self.inverted_bottleneck is not None:
    #         adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
    #         self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
    #             self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
    #         )
    #         return None
    #     else:
    #         return sorted_idx


class DynamicConvLayer(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6'):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    # @property
    # def module_str(self):
    #     return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)
    #
    # @property
    # def config(self):
    #     return {
    #         'name': DynamicConvLayer.__name__,
    #         'in_channel_list': self.in_channel_list,
    #         'out_channel_list': self.out_channel_list,
    #         'kernel_size': self.kernel_size,
    #         'stride': self.stride,
    #         'dilation': self.dilation,
    #         'use_bn': self.use_bn,
    #         'act_func': self.act_func,
    #     }
    #
    # @staticmethod
    # def build_from_config(config):
    #     return DynamicConvLayer(**config)
    #
    # ############################################################################################
    #
    # @property
    # def in_channels(self):
    #     return max(self.in_channel_list)
    #
    # @property
    # def out_channels(self):
    #     return max(self.out_channel_list)
    #
    # ############################################################################################
    #
    # def get_active_subnet(self, in_channel, preserve_weight=True):
    #     sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
    #     sub_layer = sub_layer.to(get_net_device(self))
    #
    #     if not preserve_weight:
    #         return sub_layer
    #
    #     sub_layer.conv.weight.data.copy_(self.conv.get_active_filter(self.active_out_channel, in_channel).data)
    #     if self.use_bn:
    #         copy_bn(sub_layer.bn, self.bn.bn)
    #
    #     return sub_layer
    #
    # def get_active_subnet_config(self, in_channel):
    #     return {
    #         'name': ConvLayer.__name__,
    #         'in_channels': in_channel,
    #         'out_channels': self.active_out_channel,
    #         'kernel_size': self.kernel_size,
    #         'stride': self.stride,
    #         'dilation': self.dilation,
    #         'use_bn': self.use_bn,
    #         'act_func': self.act_func,
    #     }

class DynamicResNetBottleneckBlock(nn.Module):

    def __init__(self, in_channel_list, out_channel_list, expand_ratio_list=0.25,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='avgpool_conv'):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), 8)

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, kernel_size, stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = Identity()
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list))),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, Identity):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x
