import pytest

from cv_demos.yolo_x.pytorch_yolox import run_yolox_pytorch

variants = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.yolox
def test_yolox_pytorch(variant, clear_pybuda, test_device):
    run_yolox_pytorch(variant)
