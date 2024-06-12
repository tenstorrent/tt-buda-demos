import pytest

from cv_demos.monodle.pytorch_monodle import run_monodle_pytorch


@pytest.mark.monodle
def test_monodle_pytorch(clear_pybuda):
    run_monodle_pytorch()
