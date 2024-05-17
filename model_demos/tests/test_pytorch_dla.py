import pytest

from cv_demos.dla.pytorch_dla import run_dla_pytorch

variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60x_c",
    "dla60",
    "dla60x",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.dla
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dla_pytorch(clear_pybuda, variant):
    run_dla_pytorch(variant)
