import pybuda
from pybuda._C.backend_api import BackendDevice

import os
import torch

from cv_demos.dla.utils.model import (
    dla34,
    dla46_c,
    dla46x_c,
    dla60x_c,
    dla60,
    dla60x,
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
    # PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    func = variants_func[variant]

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

    model_name = f"dla_{func.__name__}_pytorch"
    input = torch.randn(1, 3, 384, 1280)

    pytorch_model = func(pretrained="imagenet")
    pytorch_model.eval()

    # Create pybuda.PyTorchModule using the loaded Pytorch model
    tt_model = pybuda.PyTorchModule(model_name, pytorch_model)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(input,)])
    output = output_q.get()[0].value()
    print(output)
