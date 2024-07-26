# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNetV2 Demo Script - Basic

import os

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from transformers import AutoImageProcessor


def run_mobilenetv2_basic(batch_size=1):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.balancer_policy = "Ribbon"
            compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        elif available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.balancer_policy = "CNN"

    # STEP 2: Create PyBuda module from PyTorch model
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
    tt_model = pybuda.PyTorchModule("mobilenet_v2", model)

    # Image preprocessing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision,
    # to make a compatible postprocessing of the predicted class
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    n_sample_images = [image] * batch_size
    img_tensor = preprocessor(images=n_sample_images, return_tensors="pt").pixel_values

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
    output = output_q.get(timeout=0.5)

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1)
    for sample in range(batch_size):
        print(f" Sampled ID: {sample} | Predicted class: ", predicted_class_idx[sample])


if __name__ == "__main__":
    run_mobilenetv2_basic()
