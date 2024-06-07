# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import urllib

import onnx
import pybuda
import requests
import torchvision.transforms as transforms
from PIL import Image
from pybuda._C.backend_api import BackendDevice


def run_dla_onnx(variant):
    # Load model function
    model_name = f"dla_{variant}_pytorch"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        arch = available_devices[0]
        if arch == BackendDevice.Grayskull:
            if variant == "dla102x2":
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tiger"

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    # Download Model
    onnx_dir_path = "dla"
    onnx_model_path = f"{onnx_dir_path}/{variant}_Opset18.onnx"
    if not os.path.exists(onnx_model_path):
        if not os.path.exists("dla"):
            os.mkdir("dla")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset18_timm/{variant}_Opset18.onnx?download="
        response = requests.get(url, stream=True)
        with open(onnx_model_path, "wb") as f:
            f.write(response.content)

    # Load model and prepare for evaluation (inference)
    model_name = f"dla_{variant}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    tt_model = pybuda.OnnxModule(model_name, onnx_model, onnx_model_path)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(img_tensor,)])
    output = output_q.get()[0].value()

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Post processing
    predicted_value = output.argmax(-1).item()
    predicted_label = categories[predicted_value]

    # Print outputs
    print(f"True Label: {label} | Predicted Label: {predicted_label}")

    # Cleanup model files
    os.remove(onnx_model_path)
    os.rmdir(onnx_dir_path)


if __name__ == "__main__":
    run_dla_onnx("dla34")
