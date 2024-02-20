import pytest

from cv_demos.fpn.pytorch_fpn import run_fpn_pytorch


@pytest.mark.fpn
def test_fpn_pytorch(clear_pybuda):
    run_fpn_pytorch()
