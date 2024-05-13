import pybuda
from transformers import AutoImageProcessor, SegformerForImageClassification, SegformerConfig
import os
import requests
from PIL import Image


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


def run_segformer_image_classification_pytorch(variant="nvidia/mit-b0"):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == pybuda.BackendDevice.Wormhole_B0:
            if variant in ["nvidia/mit-b1", "nvidia/mit-b2", "nvidia/mit-b3", "nvidia/mit-b4", "nvidia/mit-b5"]:

                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    model = SegformerForImageClassification.from_pretrained(variant, config=config)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)

    # Create PyBuda module from PyTorch model
    tt_model = pybuda.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(pixel_values,)])
    output = output_q.get()

    # Print output
    predicted_value = output[0].value().argmax(-1).item()
    predicted_label = model.config.id2label[predicted_value]
    print("Predicted label : ", predicted_label)


if __name__ == "__main__":
    run_segformer_image_classification_pytorch()
