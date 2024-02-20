import torch
import pybuda
import onnx
import os


def run_fpn_onnx():
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda._C.Float16_b

    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Load FPN model
    onnx_model_path = "model_demos/cv_demos/fpn/weights/fpn.onnx"
    model = onnx.load(onnx_model_path)
    tt_model = pybuda.OnnxModule("onnx_fpn", model, onnx_model_path)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    output_q = pybuda.run_inference(tt_model, inputs=[(feat0, feat1, feat2)])
    output = output_q.get()
    print(output[0].shape)
