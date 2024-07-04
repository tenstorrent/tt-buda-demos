# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNetV1 Demo Script - Basic

import os

import pybuda
import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Conv block is convolutional layer followed by batch normalization and ReLU activation
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        use_relu6=False,
    ):
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
        ]

        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.model(input)


class Conv_dw_Conv(nn.Module):
    """
    Conv_dw is depthwise (dw) convolution layer followed by batch normalization and ReLU activation.
    Conv_dw_Conv is Conv_dw block followed by Conv block.
    Implemented Conv_dw_Conv instead of Conv_dw since in MobleNet, every Conv_dw is followed by Conv
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=1,
        use_relu6=False,
    ):
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size,
                stride,
                padding,
                bias=False,
                groups=in_channel,
            ),
            nn.BatchNorm2d(in_channel),
        ]
        if use_relu6:
            self.layers.append(nn.ReLU6(inplace=True))
        else:
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(
            Conv(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                use_relu6=use_relu6,
            )
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.model(input)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, use_relu6=False):
        super().__init__()

        self.num_classes = num_classes

        self.model = nn.Sequential(
            Conv(3, 32, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(32, 64, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(64, 128, kernel_size=3, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(128, 128, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(128, 256, kernel_size=3, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(256, 256, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(256, 512, kernel_size=3, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(512, 512, kernel_size=3, stride=1, use_relu6=use_relu6),
            Conv_dw_Conv(512, 1024, kernel_size=3, stride=2, use_relu6=use_relu6),
            Conv_dw_Conv(1024, 1024, kernel_size=3, stride=1, use_relu6=use_relu6),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input):
        x = self.model(input)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        out = self.fc(x)

        return out


def run_mobilenetv1_basic(batch_size=1):

    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Create PyBUDA module from PyTorch model
    model = MobileNetV1(9)
    tt_model = pybuda.PyTorchModule("mobilenet_v1", model)

    # Run inference on Tenstorrent device
    input_shape = (1, 3, 64, 64)
    input_tensor = [torch.rand(*input_shape)] * batch_size
    batch_tensor = torch.cat(input_tensor, dim=0)
    output_q = pybuda.run_inference(tt_model, inputs=([batch_tensor]))
    output = output_q.get(timeout=0.5)

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    for sample in range(batch_size):
        print(f"Sampled ID: {sample} | Output: {output[0].value()[sample]}")


if __name__ == "__main__":
    run_mobilenetv1_basic()
