# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_large_basic import run_mobilenetv3_large_basic
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_large_timm import run_mobilenetv3_large_timm
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_small_basic import run_mobilenetv3_small_basic
from cv_demos.mobilenet_v3.pytorch_mobilenet_v3_small_timm import run_mobilenetv3_small_timm


@pytest.mark.mobilenetv3
def test_mobilenetv3_large_basic_pytorch(clear_pybuda, test_device, batch_size):
    run_mobilenetv3_large_basic(batch_size=batch_size)


@pytest.mark.mobilenetv3
def test_mobilenetv3_small_basic_pytorch(clear_pybuda, test_device, batch_size):
    run_mobilenetv3_small_basic(batch_size=batch_size)


@pytest.mark.mobilenetv3
def test_mobilenetv3_large_timm_pytorch(clear_pybuda, test_device, batch_size):
    run_mobilenetv3_large_timm(batch_size=batch_size)


@pytest.mark.mobilenetv3
def test_mobilenetv3_small_timm_pytorch(clear_pybuda, test_device, batch_size):
    run_mobilenetv3_small_timm(batch_size=batch_size)
