# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# EfficientNet-Lite0 1x1 demo

import os
import shutil
import tarfile

import pybuda
import requests
import torch
from pybuda import TFLiteModule
from pybuda._C.backend_api import BackendDevice


def run_efficientnet_lite0_1x1():

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] != BackendDevice.Wormhole_B0:
            raise NotImplementedError("Model not supported on Grayskull")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Download model weights
    MODEL = "efficientnet-lite0"
    url = f"https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/{MODEL}.tar.gz"
    extract_to = "cv_demos/efficientnet_lite"
    file_name = url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(file_name, "wb") as f:
        f.write(response.content)
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_to)
    os.remove(file_name)

    # Load model path
    tflite_path = f"cv_demos/efficientnet_lite/{MODEL}/{MODEL}-fp32.tflite"
    tt_model = TFLiteModule("tflite_efficientnet_lite0", tflite_path)

    # Run inference on Tenstorrent device
    input_shape = (1, 224, 224, 3)
    input_tensor = torch.rand(input_shape)

    output_q = pybuda.run_inference(tt_model, inputs=([input_tensor]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    print(output[0].value().detach().float().numpy())

    # Remove remanent files
    shutil.rmtree(extract_to + "/" + MODEL, ignore_errors=True)


if __name__ == "__main__":
    run_efficientnet_lite0_1x1()
