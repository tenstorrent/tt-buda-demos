# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Inception-v4 Demo

import os
import sys
import urllib

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms

torch.multiprocessing.set_sharing_strategy("file_system")


def get_image():

    # Get image
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(input_image)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def run_inception_v4_osmr_pytorch(batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_PAD_SPARSE_MM"] = "{694:704, 676:704, 167:182, 158:160, 39:48}"
    os.environ["PYBUDA_MANUAL_SPLICE_DECOMP_TH"] = "158"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    compiler_cfg.balancer_op_override("_fused_op_4", "t_stream_shape", (158, 1))  # TM error
    compiler_cfg.balancer_op_override("_fused_op_7", "t_stream_shape", (158, 1))  # TM error

    if pybuda.detect_available_devices()[0] == BackendDevice.Grayskull:
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676, 1))  # TM error (ref pybuda#1527)
    elif pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        compiler_cfg.balancer_op_override("_fused_op_2", "t_stream_shape", (676, 1))  # TM error
        compiler_cfg.balancer_op_override(
            "conv2d_551.dc.sparse_matmul.10.dc.sparse_matmul.1.lc2",
            "grid_shape",
            (1, 4),
        )

    else:
        print("not a supported device!")
        sys.exit()

    # STEP 2: Create PyBuda module from PyTorch model
    model = ptcv_get_model("inceptionv4", pretrained=True)
    tt_model = pybuda.PyTorchModule("inception_v4_osmr_pt", model)

    # Run inference on Tenstorrent device
    img_tensor = [get_image()] * batch_size
    batch_tensor = torch.cat(img_tensor, dim=0)
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
    run_inception_v4_osmr_pytorch()
