import pytest

from cv_demos.ssd300_resnet50.pytorch_ssd300_resnet50 import run_pytorch_ssd300_resnet50


@pytest.mark.ssd300_resnet50
def test_ssd300_resnet50_pytorch(clear_pybuda):
    run_pytorch_ssd300_resnet50()
