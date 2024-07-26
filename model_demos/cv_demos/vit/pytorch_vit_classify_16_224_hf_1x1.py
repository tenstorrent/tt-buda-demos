# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ViT 1x1 Demo

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification


def run_vit_classify_224_hf_pytorch_1x1(variant="google/vit-base-patch16-224", batch_size=1):

    # Check target hardware
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] != pybuda._C.backend_api.BackendDevice.Wormhole_B0:
            raise NotImplementedError("This model demo is only supported on Wormhole_B0")

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    # Set PyBDUA environment variable
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    if "large" in variant:
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "20000"

    # Create PyBuda module from PyTorch model
    image_processor = AutoImageProcessor.from_pretrained(variant)
    model = ViTForImageClassification.from_pretrained(variant)
    tt_model = pybuda.PyTorchModule("pt_vit_classif_16_224_1x1", model)

    # Load sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    img_tensor = [image_processor(sample_image, return_tensors="pt").pixel_values] * batch_size
    batch_input = torch.cat(img_tensor, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Postprocessing
    predicted_class_idx = output[0].value().detach().float().numpy().argmax(-1)

    # Print output
    for sample in range(batch_size):
        print("Sampled ID: ", sample, "| Predicted class: ", model.config.id2label[predicted_class_idx[sample]])


if __name__ == "__main__":
    run_vit_classify_224_hf_pytorch_1x1()
