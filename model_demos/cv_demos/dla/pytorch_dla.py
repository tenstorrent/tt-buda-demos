# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import urllib

import pybuda
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from pybuda._C.backend_api import BackendDevice

from cv_demos.dla.utils.model import (
    dla34,
    dla46_c,
    dla46x_c,
    dla60,
    dla60x,
    dla60x_c,
    dla102,
    dla102x,
    dla102x2,
    dla169,
)

variants_func = {
    "dla34": dla34,
    "dla46_c": dla46_c,
    "dla46x_c": dla46x_c,
    "dla60x_c": dla60x_c,
    "dla60": dla60,
    "dla60x": dla60x,
    "dla102": dla102,
    "dla102x": dla102x,
    "dla102x2": dla102x2,
    "dla169": dla169,
}


def run_dla_pytorch(variant, batch_size=1):

    # Load model function
    func = variants_func[variant]
    model_name = f"dla_{variant}_pytorch"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        arch = available_devices[0]
        if arch == BackendDevice.Wormhole_B0:
            if variant == ("dla60", "dla60x"):
                compiler_cfg.place_on_new_epoch("concatenate_776.dc.concatenate.0")
        elif arch == BackendDevice.Grayskull:
            if variant in ("dla102x2", "dla169"):
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = ["tiger"] * batch_size

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    n_img_tensor = [transform(image).unsqueeze(0) for _ in range(batch_size)]
    batch_tensor = torch.cat(n_img_tensor, dim=0)

    # Load model and prepare for evaluation (inference)
    pytorch_model = func(pretrained="imagenet")
    pytorch_model.eval()

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(batch_tensor,)])
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Post processing
    predicted_value = output[0].value().argmax(-1)
    predicted_label = [categories[idx] for idx in predicted_value]

    # Print outputs
    for idx, pred in enumerate(predicted_label):
        print(f"Sampled ID: {idx} | True Label: {label[idx]} | Predicted Label: {pred}")


if __name__ == "__main__":
    run_dla_pytorch("dla34")
