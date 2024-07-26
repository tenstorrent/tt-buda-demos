# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.conv_autoencoder.pytorch_conv_autoencoder import run_conv_ae_pytorch
from cv_demos.linear_autoencoder.pytorch_linear_autoencoder import run_linear_ae_pytorch


@pytest.mark.autoencoder
def test_linear_ae_pytorch(clear_pybuda, test_device, batch_size):
    run_linear_ae_pytorch(batch_size=batch_size)


@pytest.mark.autoencoder
def test_conv_ae_pytorch(clear_pybuda, test_device, batch_size):
    run_conv_ae_pytorch(batch_size=batch_size)
