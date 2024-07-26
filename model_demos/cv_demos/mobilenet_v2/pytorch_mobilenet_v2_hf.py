# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


# MobileNetV2 Demo Script - 96x96

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def run_mobilenetv2_hf(variant="google/mobilenet_v2_0.35_96", batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    if variant == "google/mobilenet_v2_1.0_224":
        os.environ["PYBUDA_RIBBON2"] = "1"

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    preprocessor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModelForImageClassification.from_pretrained(model_ckpt)
    tt_model = pybuda.PyTorchModule("mobilenet_v2__hf", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
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
        print(
            f" Sampled ID: {sample} | Predicted class index: {predicted_class_idx[sample]} | Predicted class: {model.config.id2label[predicted_class_idx[sample]]}"
        )


if __name__ == "__main__":
    run_mobilenetv2_hf()
