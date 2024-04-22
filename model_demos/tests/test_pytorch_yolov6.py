import pytest

from cv_demos.yolo_v6.pytorch_yolov6 import run_yolov6_pytorch


variants = ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.yolov6
def test_yolov6_pytorch(variant, clear_pybuda):
    run_yolov6_pytorch(variant)
