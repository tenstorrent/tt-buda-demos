# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Ghostnet

import os
import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image


def run_ghostnet_timm(batch_size=1):
    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    model = timm.create_model("ghostnet_100", pretrained=True)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("ghostnet_100_timm_pt", model)

    data_config = timm.data.resolve_data_config({}, model=model)
    transforms = timm.data.create_transform(**data_config)

    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_tensor = [transforms(img).unsqueeze(0)] * batch_size
    batch_tensor = torch.cat(img_tensor, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_tensor]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    top5_probabilities, top5_class_indices = torch.topk(output[0].value().softmax(dim=1) * 100, k=5)

    # Get imagenet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    for sample in range(batch_size):
        print("Sample ID: ", sample)
        for i in range(top5_probabilities.size(1)):
            class_idx = top5_class_indices[sample, i]
            class_prob = top5_probabilities[sample, i]
            class_label = categories[class_idx]
            print(f"{class_label} : {class_prob}")
        print("\n")


if __name__ == "__main__":
    run_ghostnet_timm()
