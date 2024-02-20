import pytest

from cv_demos.fpn.onnx_fpn import run_fpn_onnx


@pytest.mark.fpn
def test_fpn_onnx(clear_pybuda):
    run_fpn_onnx()
