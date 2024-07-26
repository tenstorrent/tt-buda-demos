# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DenseNet Demo Script

import os
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision.transforms import CenterCrop, Compose, ConvertImageDtype, Normalize, PILToTensor, Resize

torch.multiprocessing.set_sharing_strategy("file_system")


def get_input_img(batch_size):

    # Get image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Preprocessing
    img_tensor = [transform(img).unsqueeze(0)] * batch_size
    batch_tensor = torch.cat(img_tensor, dim=0)
    return batch_tensor


def run_densenet_pytorch(variant="densenet121", batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    if variant == "densenet121":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_dram_parameters = False
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    elif variant == "densenet161":
        os.environ["PYBUDA_RIBBON2"] = "1"
        compiler_cfg.balancer_policy = "Ribbon"
        compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    elif variant == "densenet169":
        compiler_cfg.balancer_policy = "CNN"
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            else:
                os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"

    elif variant == "densenet201":
        compiler_cfg.balancer_policy = "CNN"
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Wormhole_B0:
                compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
            else:
                os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"

    # Create PyBuda module from PyTorch model
    model_ckpt = variant
    model = torch.hub.load("pytorch/vision:v0.10.0", model_ckpt, pretrained=True)
    tt_model = pybuda.PyTorchModule("densnet121_pt", model)

    # Run inference on Tenstorrent device
    img_tensor = get_input_img(batch_size=batch_size)
    output_q = pybuda.run_inference(tt_model, inputs=([img_tensor]))
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
    predicted_label = [categories[cat_id] for cat_id in top1_catid]

    # Results
    for sample in range(batch_size):
        print(
            f"Sample ID: {sample} | Predicted Label: {predicted_label[sample]} | Predicted Probability: {top1_prob[sample].item():.2f}"
        )


if __name__ == "__main__":
    run_densenet_pytorch()
