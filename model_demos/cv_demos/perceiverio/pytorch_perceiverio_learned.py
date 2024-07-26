# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Perceiver IO Learned Demo Script

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, PerceiverForImageClassificationLearned


def run_perceiverio_learned_pytorch(batch_size=1):

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
    label = ["tabby, tabby cat"] * batch_size

    # Data preprocessing
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = [inputs["pixel_values"]] * batch_size
    batch_input = torch.cat(pixel_values, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_" + str(model_ckpt.split("/")[-1].replace("-", "_")), model), inputs=[(batch_input,)]
    )
    output = output_q.get()  # return last queue object

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1)
    # Print output
    for idx, class_idx in enumerate(predicted_value):
        print("Sampled ID: ", idx, " | Predicted class: ", (model.config.id2label[class_idx.item()]))


if __name__ == "__main__":
    run_perceiverio_learned_pytorch()
