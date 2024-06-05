import pytest

from cv_demos.perceiverio.pytorch_perceiverio_conv import run_perceiverio_conv_pytorch
from cv_demos.perceiverio.pytorch_perceiverio_fourier import run_perceiverio_fourier_pytorch
from cv_demos.perceiverio.pytorch_perceiverio_learned import run_perceiverio_learned_pytorch


@pytest.mark.perceiverio
def test_perceiverio_conv_pytorch(clear_pybuda):
    run_perceiverio_conv_pytorch()


@pytest.mark.perceiverio
def test_perceiverio_fourier_pytorch(clear_pybuda):
    run_perceiverio_fourier_pytorch()


@pytest.mark.perceiverio
def test_perceiverio_learned_pytorch(clear_pybuda):
    run_perceiverio_learned_pytorch()
