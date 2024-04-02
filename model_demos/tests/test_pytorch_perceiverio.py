import pytest

from cv_demos.perceiverio.pytorch_perceiverio import run_perceiverio_pytorch


@pytest.mark.perceiverio
def test_perceiverio_pytorch(clear_pybuda):
    run_perceiverio_pytorch()
