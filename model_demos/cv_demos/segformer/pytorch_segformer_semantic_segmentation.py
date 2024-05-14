import pybuda
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
import os
import requests
from PIL import Image

torch.multiprocessing.set_sharing_strategy("file_system")


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


def run_segformer_semseg_pytorch(variant="nvidia/segformer-b0-finetuned-ade-512-512"):

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
    pixel_values = get_sample_data(variant)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(pixel_values,)])
    output = output_q.get()[0].value()

    # Print output
    print("output=", output)


if __name__ == "__main__":
    run_segformer_semseg_pytorch()
