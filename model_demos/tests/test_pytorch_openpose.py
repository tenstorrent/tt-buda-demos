# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.openpose.pytorch_lwopenpose_2d_osmr import run_lwopenpose_2d_osmr_pytorch
from cv_demos.openpose.pytorch_lwopenpose_3d_osmr import run_lwopenpose_3d_osmr_pytorch


@pytest.mark.openpose
def test_openpose_2d_osmr(clear_pybuda, test_device, batch_size):
    run_lwopenpose_2d_osmr_pytorch(batch_size=batch_size)


@pytest.mark.openpose
def test_openpose_3d_osmr(clear_pybuda, test_device, batch_size):
    run_lwopenpose_3d_osmr_pytorch(batch_size=batch_size)
