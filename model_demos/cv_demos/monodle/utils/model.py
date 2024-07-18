"""
Code adapted from https://github.com/xinzhuma/monodle.git

MIT License

Copyright (c) 2021 XINZHU.MA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def get_model_url(data="imagenet", name="dla34", hash="ba72cf86"):
    return os.path.join("http://dl.yf.io/dla/models", data, "{}-{}.pth".format(name, hash))


BatchNorm = nn.BatchNorm2d


class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_szie=3, stride=1, bias=True):

        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernal_szie,
            stride=stride,
            padding=kernal_szie // 2,
            bias=bias,
        )
        self.bn = BatchNorm(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# weight init for up-sample layers [tranposed conv2d]
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    """
    input: features map of different layers
    output: up-sampled features
    """

    def __init__(self, in_channels_list, up_factors_list, out_channels):

        super(IDAUp, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = Conv2d(in_channels, out_channels, kernal_szie=3, stride=1, bias=False)
            node = Conv2d(out_channels * 2, out_channels, kernal_szie=3, stride=1, bias=False)
            up = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=up_factors * 2,
                stride=up_factors,
                padding=up_factors // 2,
                output_padding=0,
                groups=out_channels,
                bias=False,
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):

        assert len(self.in_channels_list) == len(layers), "{} vs {} layers".format(
            len(self.in_channels_list), len(layers)
        )

        for i in range(1, len(layers)):
            upsample = getattr(self, "up_" + str(i))
            project = getattr(self, "proj_" + str(i))
            node = getattr(self, "node_" + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i - 1], layers[i]], 1))

        return layers


class DLAUp(nn.Module):
    def __init__(self, in_channels_list, scales_list=(1, 2, 4, 8, 16)):

        super(DLAUp, self).__init__()
        scales_list = np.array(scales_list, dtype=int)

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(
                    in_channels_list=in_channels_list[j:],
                    up_factors_list=scales_list[j:] // scales_list[j],
                    out_channels=in_channels_list[j],
                ),
            )
            scales_list[j + 1 :] = scales_list[j]
            in_channels_list[j + 1 :] = [in_channels_list[j] for _ in in_channels_list[j + 1 :]]

    def forward(self, layers):

        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, "ida_{}".format(i))
            layers[-i - 2 :] = ida(layers[-i - 2 :])
        return layers[-1]


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):

        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):

        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):

        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):

        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels),
            )

    def forward(self, x, residual=None, children=None):

        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        return_levels=False,
        pool_size=7,
        linear_root=False,
    ):

        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    BatchNorm(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):

        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self, data="imagenet", name="dla34", hash="ba72cf86"):
        fc = self.fc
        if name.endswith(".pth"):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.load_state_dict(model_weights, strict=False)
        self.fc = fc


def dla34(pretrained=False, **kwargs):  # DLA-34

    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data="imagenet", name="dla34", hash="ba72cf86")
    return model


class CenterNet3D(nn.Module):
    def __init__(self, backbone="dla34", neck="DLAUp", num_class=3, downsample=4):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """

        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.heads = {
            "heatmap": num_class,
            "offset_2d": 2,
            "size_2d": 2,
            "depth": 2,
            "offset_3d": 2,
            "size_3d": 3,
            "heading": 24,
        }
        self.backbone = dla34(pretrained=True, return_levels=True)
        channels = self.backbone.channels  # channels list for feature maps generated by backbone
        self.first_level = int(np.log2(downsample))
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.neck = DLAUp(channels[self.first_level :], scales_list=scales)  # feature fusion [such as DLAup, FPN]

        # initialize the head of pipeline, according to heads setting.
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            )

            # initialization
            if "heatmap" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def forward(self, input):

        feat = self.backbone(input)
        feat = self.neck(feat[self.first_level :])

        ret = OrderedDict()
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)

        return ret

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
