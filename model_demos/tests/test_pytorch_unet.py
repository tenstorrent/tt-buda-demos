# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.unet.pytorch_unet_qubvel import run_unet_qubvel_pytorch
from cv_demos.unet.pytorch_unet_torchhub import run_unet_torchhub_pytorch


@pytest.mark.unet
def test_unet_qubvel(clear_pybuda, test_device, batch_size):
    run_unet_qubvel_pytorch(batch_size=batch_size)


@pytest.mark.unet
def test_unet_torchhub(clear_pybuda, test_device, batch_size):
    run_unet_torchhub_pytorch(batch_size=batch_size)
