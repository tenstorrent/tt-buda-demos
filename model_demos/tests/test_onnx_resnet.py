# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.resnet.onnx_resnet import run_resnet_onnx


@pytest.mark.resnet
def test_resnet_onnx(clear_pybuda, test_device, batch_size):
    run_resnet_onnx(batch_size=batch_size)
