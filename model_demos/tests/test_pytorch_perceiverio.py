import pytest

from cv_demos.perceiverio.pytorch_perceiverio import run_perceiverio_pytorch

variants = [
    "deepmind/vision-perceiver-conv",
    "deepmind/vision-perceiver-fourier",
    "deepmind/vision-perceiver-learned",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.perceiverio
def test_perceiverio_pytorch(clear_pybuda, variant):
    run_perceiverio_pytorch(variant)
