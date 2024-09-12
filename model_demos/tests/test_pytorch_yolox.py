# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.yolo_x.pytorch_yolox import run_yolox_pytorch

variants = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.yolox
def test_yolox_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_yolox_pytorch(variant, batch_size=batch_size)
