# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo Script

import os
from typing import Optional, List
import pybuda
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from pybuda.tensor import Tensor
from utils.config import Config, DeviceType
from pybuda.tools.tti_data_parallel import (
    RunMode,
    RunResult,
    ForwardRunInputs,
    run_tti_data_parallel,
)

def run_resnet_single_card(model, pixel_values):
    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(pybuda.PyTorchModule("pt_resnet50", model), inputs=[(pixel_values,)])
    output = output_q.get()  # return last queue object
    return output

def run_resnet_multi_card(pixel_values, precompiled_tti_path: str, device_overrides: List[List[int]], output_dir: Optional[str]=None):
    os.environ["PYBUDA_FORCE_THREADS"] = "1"
    
    if not output_dir:
        output_dir = "./cv_demos/resnet/resnet_multi_card"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result: RunResult = run_tti_data_parallel(
        arch=pybuda.BackendDevice.Wormhole_B0,
        device_ids=device_overrides,
        run_mode=RunMode.FORWARD,
        inputs=ForwardRunInputs(inputs=[pixel_values]), 
        output_dir=output_dir, 
        num_loops=1,
        precompiled_tti_path=precompiled_tti_path,
        sync_at_run_start=True,
    )
    
    return [Tensor.create_from_torch(output_tensor) for output_tensor in result.outputs[0]]


def run_resnet_pytorch(variant="microsoft/resnet-50", config=Config(device=DeviceType.e150, multi_chip=False, batch_size=1)):

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = ResNetForImageClassification.from_pretrained(model_ckpt)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"

    # Check for dual-chip mode
    if config.n300_data_parallel():
        os.environ["PYBUDA_N300_DATA_PARALLEL"] = "1"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tiger"

    # Adjust for batch size
    image = [image] * config.batch_size

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    if config.multi_card_dp():
        output = run_resnet_multi_card(pixel_values, config.precompiled_tti_path, config.multi_card_devices)
    
    else:
        output = run_resnet_single_card(model, pixel_values)

    # Data postprocessing
    predicted_value = output[0].value().argmax(-1)
    for idx, value in enumerate(predicted_value):
        predicted_label = model.config.id2label[value.item()]

        print(f"Sample: {idx} | True Label: {label} | Predicted Label: {predicted_label}")


if __name__ == "__main__":
    # run_resnet_pytorch()
    multi_card_config = Config(
        device=DeviceType.n300,
        multi_chip=True,
        batch_size=256 * 4,
        precompiled_tti_path="./cv_demos/resnet/resnet_50_dp_new.tti",
        multi_card_devices=[[i, i + 4] for i in range(4)]
    )
    run_resnet_pytorch(config=multi_card_config)