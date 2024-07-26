# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


# MobileNetV1 Demo Script - 192x192

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def run_mobilenetv1_hf(variant="google/mobilenet_v1_0.75_192", batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    preprocessor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModelForImageClassification.from_pretrained(model_ckpt)
    tt_model = pybuda.PyTorchModule("mobilenet_v1_hf", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    input_tensor = [inputs.pixel_values] * batch_size
    batch_tensor = torch.cat(input_tensor, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_tensor]))
    output = output_q.get(timeout=0.5)

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1)
    # Print output
    for idx, class_idx in enumerate(predicted_class_idx):
        print("Sampled ID: ", idx, " | Predicted class: ", (model.config.id2label[class_idx]))


if __name__ == "__main__":
    run_mobilenetv1_hf()
