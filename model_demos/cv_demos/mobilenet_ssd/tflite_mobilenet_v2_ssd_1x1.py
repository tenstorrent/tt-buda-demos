# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNet SSD 1x1 Demo Script

import os

import pybuda
import requests
import torch
from PIL import Image
from pybuda import TFLiteModule
from pybuda._C.backend_api import BackendDevice
from torchvision import transforms


def run_mobilenetv2_ssd_1x1_tflite(batch_size=1):

    # Set PyBUDA configuration parameters
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] != BackendDevice.Wormhole_B0:
            raise NotImplementedError("Model not supported on Grayskull")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.graph_solver_self_cut_type = "FastCut"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.cpu_fallback_ops = set(["concatenate"])

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Download model weights
    url = "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/latest/ssd_mobilenet_v2.tflite"
    tflite_path = "cv_demos/mobilenet_ssd/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(tflite_path, "wb") as f:
        f.write(response.content)

    # Load model path
    tt_model = TFLiteModule("tflite_mobilenet_ssd", tflite_path)

    # Image preprocessing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = [transform(image).permute((1, 2, 0)).unsqueeze(0)] * batch_size
    batch_tensor = torch.cat(img_tensor, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_tensor]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    for sample in range(batch_size):
        print(f"Sampled ID: {sample} | Output: {output[0].value()[sample]}")

    # Remove weight file
    os.remove(tflite_path)


if __name__ == "__main__":
    run_mobilenetv2_ssd_1x1_tflite()
