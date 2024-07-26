# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.efficientnet_lite.tflite_efficientnet_lite0_1x1 import run_efficientnet_lite0_1x1
from cv_demos.efficientnet_lite.tflite_efficientnet_lite4_1x1 import run_efficientnet_lite4_1x1


@pytest.mark.efficientnetlite
def test_efficientnet_lite0_1x1(clear_pybuda, test_device, batch_size):
    run_efficientnet_lite0_1x1(batch_size=batch_size)


@pytest.mark.efficientnetlite
def test_efficientnet_lite4_1x1(clear_pybuda, test_device, batch_size):
    run_efficientnet_lite4_1x1(batch_size=batch_size)
