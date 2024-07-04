# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResneXt Demo Script

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms


def get_image_tensor():
    # Image processing
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
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


def run_resnext_pytorch(variant=("resnext14_32x4d", "osmr"), batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if not available_devices:
        raise NotImplementedError("No device detected")

    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    if available_devices[0] == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    model_ckpt = variant[0]
    impl = variant[1]
    model_name = f"pt_{model_ckpt.replace('/', '_')}"
    if model_ckpt == "resnext14_32x4d":
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{24*1024}"
    elif model_ckpt == "resnext26_32x4d":
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"
    elif model_ckpt == "resnext50_32x4d":
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.default_dram_parameters = False
        elif available_devices[0] == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"
    elif model_ckpt == "resnext101_32x8d_wsl":
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.enable_auto_transposing_placement = True
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
    elif model_ckpt == "resnext101_32x8d":
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"
    elif model_ckpt == "resnext101_64x4d":
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.default_dram_parameters = False
            os.environ["PYBUDA_BALANCER_PREPASS_DISABLED"] = "1"
        elif available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.enable_auto_transposing_placement = True
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"

    # Create PyBuda module from PyTorch model
    if impl == "osmr":
        model = ptcv_get_model(model_ckpt, pretrained=True)
    else:
        model = torch.hub.load(impl, model_ckpt)

    model.eval()
    tt_model = pybuda.PyTorchModule(model_name, model)
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
    run_resnext_pytorch()
