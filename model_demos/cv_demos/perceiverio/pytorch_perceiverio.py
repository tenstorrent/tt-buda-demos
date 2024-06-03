# Perceiver IO Demo Script

import os

import pybuda
import requests
from PIL import Image
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
)


def run_perceiverio_pytorch(variant="deepmind/vision-perceiver-conv"):

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    if variant == "deepmind/vision-perceiver-learned":
        model = PerceiverForImageClassificationLearned.from_pretrained(model_ckpt)

    elif variant == "deepmind/vision-perceiver-conv":
        model = PerceiverForImageClassificationConvProcessing.from_pretrained(model_ckpt)

    elif variant == "deepmind/vision-perceiver-fourier":
        model = PerceiverForImageClassificationFourier.from_pretrained(model_ckpt)

    else:
        print(f"The model {variant} is not supported")

    model.eval()

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    compiler_cfg.enable_auto_fusing = False

    if model_ckpt == "deepmind/vision-perceiver-conv":
        compiler_cfg.default_dram_parameters = False
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"

    if model_ckpt in ["deepmind/vision-perceiver-learned", "deepmind/vision-perceiver-fourier"]:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == pybuda.BackendDevice.Wormhole_B0:
            if model_ckpt == "deepmind/vision-perceiver-conv":
                compiler_cfg.balancer_op_override(
                    "max_pool2d_33.dc.reshape.10.dc.sparse_matmul.13.lc2", "t_stream_shape", (1, 1)
                )

            if model_ckpt == "deepmind/vision-perceiver-fourier":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"

        if available_devices[0] == pybuda.BackendDevice.Grayskull:

            if variant in ["deepmind/vision-perceiver-learned", "deepmind/vision-perceiver-fourier"]:
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"

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
    run_perceiverio_pytorch()
