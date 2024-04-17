import pytest

from cv_demos.hardnet.pytorch_hardnet import run_hardnet_pytorch

variants = ["hardnet68", "hardnet85", "hardnet68ds", "hardnet39ds"]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.hardnet
def test_hardnet_pytorch(variant, clear_pybuda):
    run_hardnet_pytorch(variant)
