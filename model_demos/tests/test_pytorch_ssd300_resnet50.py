# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.ssd300_resnet50.pytorch_ssd300_resnet50 import run_pytorch_ssd300_resnet50


@pytest.mark.ssd300_resnet50
def test_ssd300_resnet50_pytorch(clear_pybuda, test_device, batch_size):
    run_pytorch_ssd300_resnet50(batch_size=batch_size)
