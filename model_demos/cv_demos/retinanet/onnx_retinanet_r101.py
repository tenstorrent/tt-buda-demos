# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# import PyBuda library

import os

import numpy as np
import onnx
import pybuda
import requests
import torch
from PIL import Image


def img_preprocess(scal_val=1):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    scale = scal_val
    w, h = pil_img.size
    print("----", w, h)
    newW, newH = int(scale * w), int(scale * h)
    newW, newH = 640, 480
    assert newW > 0 and newH > 0, "Scale is too small, resized images would have no pixel"
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img, dtype=np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img


def run_retinanet_r101_640x480_onnx(batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.graph_solver_self_cut_type = "ConsumerOperandDataEdgesFirst"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_356"] = 3

    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_DISABLE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{75*1024}"
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == pybuda.BackendDevice.Grayskull:
            os.environ["PYBUDA_RIBBON2"] = "1"
            os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
        elif available_devices[0] == pybuda.BackendDevice.Wormhole_B0:
            compiler_cfg.place_on_new_epoch("conv2d_1557.dc.matmul.11")

    # Download model weights
    url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx?download="
    load_path = "cv_demos/retinanet/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(load_path, "wb") as f:
        f.write(response.content)

    # Create PyBuda module from PyTorch model
    model = onnx.load(load_path)
    tt_model = pybuda.OnnxModule("onnx_retinanet", model, load_path)

    # Image preprocessing
    img_tensor = [img_preprocess()] * batch_size
    batch_input = torch.cat(img_tensor, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Print outputs
    for sample in range(batch_size):
        print("Sample ID: ", sample, "| Result: ", output[sample], "\n")

    # Remove weight file
    os.remove(load_path)


if __name__ == "__main__":
    run_retinanet_r101_640x480_onnx()
