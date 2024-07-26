# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.yolo_v5.pytorch_yolov5_320 import run_pytorch_yolov5_320
from cv_demos.yolo_v5.pytorch_yolov5_480 import run_pytorch_yolov5_480
from cv_demos.yolo_v5.pytorch_yolov5_640 import run_pytorch_yolov5_640

variants = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_320(clear_pybuda, test_device, variant, batch_size):
    run_pytorch_yolov5_320(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_640(clear_pybuda, test_device, variant, batch_size):
    run_pytorch_yolov5_640(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolov5
def test_pytorch_yolov5_480(clear_pybuda, test_device, variant, batch_size):
    run_pytorch_yolov5_480(variant, batch_size=batch_size)
