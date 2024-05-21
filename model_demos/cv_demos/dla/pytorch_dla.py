import os
import urllib

import pybuda
import requests
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


def run_dla_pytorch(variant):

    # Load model function
    func = variants_func[variant]
    model_name = f"dla_{func.__name__}_pytorch"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        arch = available_devices[0]
        if arch == BackendDevice.Grayskull:
            if func.__name__ == "dla102x2":
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        elif arch == BackendDevice.Wormhole_B0:
            if func.__name__ == "dla60x":
                compiler_cfg.place_on_new_epoch("concatenate_776.dc.concatenate.0")
            elif func.__name__ == "dla60x":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "20480"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = "tiger"

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    # Load model and prepare for evaluation (inference)
    pytorch_model = func(pretrained="imagenet")
    pytorch_model.eval()

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(img_tensor,)])
    output = output_q.get()[0].value()

    # Get ImageNet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Post processing
    predicted_value = output.argmax(-1).item()
    predicted_label = categories[predicted_value]

    # Print outputs
    print(f"True Label: {label} | Predicted Label: {predicted_label}")


if __name__ == "__main__":
    run_dla_pytorch("dla34")
