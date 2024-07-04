import pytest

from cv_demos.monodle.pytorch_monodle import run_monodle_pytorch


@pytest.mark.monodle
def test_monodle_pytorch(clear_pybuda, test_device, batch_size):
    run_monodle_pytorch(batch_size=batch_size)
