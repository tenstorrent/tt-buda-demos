# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Perceiver IO Learned Demo Script

import os

import pybuda
import requests
from PIL import Image
from transformers import AutoImageProcessor, PerceiverForImageClassificationLearned


def run_perceiverio_learned_pytorch():

    # Load feature extractor and model checkpoint from HuggingFace
    model_ckpt = "deepmind/vision-perceiver-learned"
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = PerceiverForImageClassificationLearned.from_pretrained(model_ckpt)
    model.eval()

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    compiler_cfg.enable_auto_fusing = False
    os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == pybuda.BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"
        elif available_devices[0] == pybuda.BackendDevice.Wormhole_B0:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "40960"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tabby, tabby cat"

    # Data preprocessing
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_" + str(model_ckpt.split("/")[-1].replace("-", "_")), model), inputs=[(pixel_values,)]
    )
    output = output_q.get()  # return last queue object

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1).item()
    predicted_label = model.config.id2label[predicted_value]

    print(f"True Label: {label} | Predicted Label: {predicted_label}")


if __name__ == "__main__":
    run_perceiverio_learned_pytorch()
