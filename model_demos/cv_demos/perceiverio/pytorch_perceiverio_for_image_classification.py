import pybuda
import torch
import os
import requests
from PIL import Image
from loguru import logger
from transformers import AutoImageProcessor, PerceiverForImageClassificationConvProcessing


def get_sample_data(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        height = image_processor.to_dict()["size"]["height"]
        width = image_processor.to_dict()["size"]["width"]
        pixel_values = torch.rand(1, 3, height, width).to(torch.float32)
    return pixel_values


def run_perceiverio_for_image_classification_pytorch(model_name):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"

    # Sample Image
    pixel_values = get_sample_data(model_name)

    # Load the model
    model = PerceiverForImageClassificationConvProcessing.from_pretrained(model_name)
    model.eval()

    # Create PyBuda module from Pytorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(model_name.split("/")[-1].replace("-", "_")), model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        tt_model,
        inputs=([pixel_values]),
    )
    output = output_q.get()
    predicted_value = output[0].value().argmax(-1).item()
    predicted_label = model.config.id2label[predicted_value]
    print("Predicted label : ", predicted_label)


if __name__ == "__main__":
    run_perceiverio_for_image_classification_pytorch(model_name="deepmind/vision-perceiver-conv")
