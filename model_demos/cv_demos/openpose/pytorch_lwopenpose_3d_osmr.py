# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# LW-OpenPose 2D Demo Script

import os

import pybuda
import requests
import torch
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def get_image_tensor():
    # Image processing
    url = "https://raw.githubusercontent.com/axinc-ai/ailia-models/master/pose_estimation_3d/blazepose-fullbody/girl-5204299_640.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def run_lwopenpose_3d_osmr_pytorch(batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16

    # Create PyBuda module from PyTorch model
    model = ptcv_get_model("lwopenpose3d_mobilenet_cmupan_coco", pretrained=True)
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_lwopenpose_3d_osmr", model)

    # Get sample input

    input_batch = [get_image_tensor()] * batch_size
    batch_input = torch.cat(input_batch, dim=0)

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
        print(f"Sample ID: {sample} | Output: {output[0].value()[sample]}")


if __name__ == "__main__":
    run_lwopenpose_3d_osmr_pytorch()
