# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo Script - ONNX
# Uses torch and torchvision for data pre- and post-processing;
# can use other frameworks such as MXNet, TensorFlow or Numpy

import os
import urllib

import onnx
import pybuda
import requests
import torch
from PIL import Image
from torchvision import transforms


def preprocess(image: Image) -> torch.tensor:
    """Image preprocessing for ResNet50

    Parameters
    ----------
    image : PIL.Image
        PIL Image sample

    Returns
    -------
    torch.tensor
        Preprocessed input tensor
    """
    transform_fn = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transform_fn(image).unsqueeze(0)

    return pixel_values


def postprocess(predictions: torch.tensor) -> tuple:
    """Model prediction postprocessing for ResNet50

    Parameters
    ----------
    predictions : torch.tensor
        Model predictions

    Returns
    -------
    tuple
        topk probability and category ID
    """

    # Get probabilities
    probabilities = torch.nn.functional.softmax(predictions, dim=1)

    # Get top-k prediction
    top1_prob, top1_catid = torch.topk(probabilities, 1)

    return top1_prob, top1_catid


def run_resnet_onnx(batch_size=1):

    # Download model weights
    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx?download="
    load_path = "cv_demos/resnet/" + url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(load_path, "wb") as f:
        f.write(response.content)

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model = onnx.load(load_path)

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.enable_auto_fusing = False
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = ["tiger"] * batch_size

    # Data preprocessing
    pixel_values = [preprocess(image)] * batch_size
    batch_input = torch.cat(pixel_values, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.OnnxModule("onnx_resnet50", model, load_path),
        inputs=[(batch_input,)],
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    top1_prob, top1_catid = postprocess(output[0].value())

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]
    predicted_label = [categories[cat_id] for cat_id in top1_catid]

    # Results
    for sample in range(batch_size):
        print(
            f"True Label: {label[sample]} | Predicted Label: {predicted_label[sample]} | Predicted Probability: {top1_prob[sample].item():.2f}"
        )

    # Remove weight file
    os.remove(load_path)


if __name__ == "__main__":
    run_resnet_onnx()
