# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MLP-Mixer - TIMM Demo Script

import os
import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def run_mlpmixer_timm(batch_size=1):

    # Load MLP-Mixer feature extractor and model from TIMM
    # "mixer_b16_224", "mixer_b16_224_in21k", "mixer_b16_224_miil", "mixer_b16_224_miil_in21k",
    # "mixer_b32_224", "mixer_l16_224", "mixer_l16_224_in21k",
    # "mixer_l32_224", "mixer_s16_224", "mixer_s32_224"
    variant = "mixer_b16_224"
    model = timm.create_model(variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    label = ["tiger"] * batch_size

    # Data preprocessing
    pixel_values = [transform(image).unsqueeze(0)] * batch_size
    batch_tensor = torch.cat(pixel_values, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(pybuda.PyTorchModule(f"timm_{variant}", model), inputs=[(batch_tensor,)])
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0].value(), dim=1)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Get top-k prediction
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    predicted_label = [categories[idx] for idx in top1_catid]

    # Print outputs
    for idx, pred in enumerate(predicted_label):
        print(
            f"Sampled ID: {idx} | True Label: {label[idx]} | Predicted Label: {pred} | Predicted Probabilty: {top1_prob[idx]}"
        )


if __name__ == "__main__":
    run_mlpmixer_timm()
