# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.mobilenet_ssd.tflite_mobilenet_v2_ssd_1x1 import run_mobilenetv2_ssd_1x1_tflite


@pytest.mark.mobilenetssd
def test_mobilenetv2_ssd_1x1_tflite(clear_pybuda, test_device, batch_size):
    run_mobilenetv2_ssd_1x1_tflite(batch_size=batch_size)
