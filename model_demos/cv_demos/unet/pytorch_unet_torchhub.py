# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# U-Net TorchHub Demo

import os

import numpy as np
import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision import transforms


def run_unet_torchhub_pytorch(batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.default_dram_parameters = False
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.balancer_op_override(
                "conv2d_transpose_174.dc.conv2d.17.dc.matmul.11",
                "grid_shape",
                (4, 4),
            )

    # Create PyBuda module from PyTorch model
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_unet_torchhub", model)

    # Download an example input image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/unet_tcga_cs_4944.png"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ]
    )
    input_tensor = preprocess(input_image)
    img_batch = [input_tensor.unsqueeze(0)] * batch_size
    batch_input = torch.cat(img_batch, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Print output
    for sample in range(batch_size):
        print("Sample ID: ", sample, "| Result: ", output[0].value()[sample], "\n")


if __name__ == "__main__":
    run_unet_torchhub_pytorch()
