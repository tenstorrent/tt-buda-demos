# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Inception-v4

import os
import sys
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


def preprocess_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    return model, img_tensor


def run_inception_v4_timm_pytorch(batch_size=1):

    model_name = "inception_v4"
    model, img_tensor = preprocess_timm_model(model_name)
    batch_tensor = torch.cat([img_tensor] * batch_size, dim=0)

    # Configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{694:704, 676:704, 167:182, 158:160, 39:48}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "158"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.balancer_op_override("_fused_op_4", "t_stream_shape", (158, 1))  # TM error
    compiler_cfg.balancer_op_override("_fused_op_7", "t_stream_shape", (158, 1))  # TM error

    if pybuda.detect_available_devices()[0] == BackendDevice.Grayskull:
        # STEP 1: Set PyBuda configuration parameters
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676, 1))  # TM error (ref pybuda#1527)
    elif pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        # STEP 1: Set PyBuda configuration parameters
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676, 1))  # TM error
        compiler_cfg.balancer_op_override(
            "conv2d_551.dc.sparse_matmul.10.dc.sparse_matmul.1.lc2",
            "grid_shape",
            (1, 4),
        )
        compiler_cfg.default_dram_parameters = False
    else:
        print("not a supported device!")
        sys.exit()

    # STEP 2: Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule(model_name + "_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        tt_model,
        inputs=([batch_tensor]),
        _verify_cfg=pybuda.VerifyConfig(
            verify_post_placer=False,
            verify_post_autograd_passes=False,
        ),
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0].value(), dim=1)
    # Get imagenet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for sample in range(batch_size):
        result = {}  # reset at the start of each new sample
        for i in range(top5_prob.size(1)):
            result[categories[top5_catid[sample][i]]] = top5_prob[sample][i].item()
        print("Sample ID: ", sample, "| Result: ", result)


if __name__ == "__main__":
    run_inception_v4_timm_pytorch()
