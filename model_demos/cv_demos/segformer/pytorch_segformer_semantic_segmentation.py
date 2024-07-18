# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

torch.multiprocessing.set_sharing_strategy("file_system")


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


def run_segformer_semseg_pytorch(variant="nvidia/segformer-b0-finetuned-ade-512-512", batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == pybuda.BackendDevice.Wormhole_B0:
            if variant in [
                "nvidia/segformer-b1-finetuned-ade-512-512",
                "nvidia/segformer-b2-finetuned-ade-512-512",
                "nvidia/segformer-b3-finetuned-ade-512-512",
                "nvidia/segformer-b4-finetuned-ade-512-512",
            ]:

                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        elif available_devices[0] == pybuda.BackendDevice.Grayskull:
            if variant in [
                "nvidia/segformer-b2-finetuned-ade-512-512",
                "nvidia/segformer-b3-finetuned-ade-512-512",
                "nvidia/segformer-b4-finetuned-ade-512-512",
            ]:
                compiler_cfg.amp_level = 1

    # Load the model from HuggingFace
    model = SegformerForSemanticSegmentation.from_pretrained(variant)
    model.eval()

    # Load the sample image
    pixel_values = [get_sample_data(variant)] * batch_size
    batch_input = torch.cat(pixel_values, dim=0)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(batch_input,)])
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Print output
    for sample in range(batch_size):
        print("Sample ID: ", sample, "| Result: ", output[0].value()[sample], "\n")


if __name__ == "__main__":
    run_segformer_semseg_pytorch()
