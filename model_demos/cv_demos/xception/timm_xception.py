# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Xception

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

torch.multiprocessing.set_sharing_strategy("file_system")


def run_xception_timm(variant="xception", batch_size=1):
    """
    Variants = {
     'xception',
     'xception41',
     'xception65',
     'xception71'
    }
    """

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    available_devices = pybuda.detect_available_devices()

    if variant == "xception":
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.balancer_policy = "CNN"
        elif available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"
            compiler_cfg.amp_level = 1
    if available_devices[0] == BackendDevice.Grayskull:
        compiler_cfg.balancer_policy = "Ribbon"

    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.default_dram_parameters = False

    model_name = variant
    model = timm.create_model(model_name, pretrained=True)

    # preprocessing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    tensor = [transform(img).unsqueeze(0)] * batch_size  # transform and add batch dimension
    batch_input = torch.cat(tensor, dim=0)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(f"{variant}_timm_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # postprocessing
    probabilities = torch.nn.functional.softmax(output[0].value(), dim=1)
    # Get imagenet class mappings
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
    run_xception_timm()
