# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# VoVNet Model V2

import os
import urllib

import pybuda
import requests
import timm
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Source
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vovnet.py
# 9 architecture variants are there, but loaded pre-trained weights for 2 variants only
# Vovnet V2 has a prefeix ese_* and Vovnet V1 without that prefix
"""
vovnet_timm_model_list =   [
      'ese_vovnet19b_dw',       # Good
      'ese_vovnet19b_slim',     # untrained
      'ese_vovnet19b',          # untrained
      'ese_vovnet19b_slim_dw',  # untrained
      'vovnet39a',              # untrained
      'ese_vovnet39b',          # Good
      'vovnet57a',              # untrained
      'ese_vovnet57b',          # untrained
      'ese_vovnet99b'           # Good
      ]
"""


def preprocess_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    return model, img_tensor


def run_vovnet_ese_19bdw_timm_pytorch(variant="ese_vovnet19b_dw", batch_size=1):

    model_name = variant
    model, img_tensor = preprocess_timm_model(model_name)
    input_batch = [img_tensor] * batch_size  # create a mini-batch as expected by the model
    batch_input = torch.cat(input_batch, dim=0)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(model_name + "_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0].value(), dim=1)

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for sample in range(batch_size):
        result = {}  # reset at the start of each new sample
        for i in range(top5_prob.size(1)):
            result[categories[top5_catid[sample][i]]] = top5_prob[sample][i].item()
        print("Sample ID: ", sample, "| Result: ", result)


if __name__ == "__main__":
    run_vovnet_ese_19bdw_timm_pytorch()
