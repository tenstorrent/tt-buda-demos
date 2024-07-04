import os

import pybuda
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from pybuda._C.backend_api import BackendDevice

from cv_demos.monodle.utils.model import CenterNet3D


def run_monodle_pytorch(batch_size=1):
    model_name = "monodle_pytorch"

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    available_devices = pybuda.detect_available_devices()
    if available_devices:
        arch = available_devices[0]
        if arch == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{88*1024}"
        elif arch == BackendDevice.Grayskull:
            os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = [transform(image).unsqueeze(0)] * batch_size
    batch_input = torch.cat(img_tensor, dim=0)

    # Load model and prepare for evaluation (inference)
    pytorch_model = CenterNet3D(backbone="dla34")
    pytorch_model.eval()

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(batch_input,)])
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    for sample in range(batch_size):
        print(f"Sampled ID: {sample} | Output: {output[0].value()[sample]}")


if __name__ == "__main__":
    run_monodle_pytorch()
