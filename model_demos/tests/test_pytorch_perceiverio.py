# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.perceiverio.pytorch_perceiverio_conv import run_perceiverio_conv_pytorch
from cv_demos.perceiverio.pytorch_perceiverio_fourier import run_perceiverio_fourier_pytorch
from cv_demos.perceiverio.pytorch_perceiverio_learned import run_perceiverio_learned_pytorch


@pytest.mark.perceiverio
def test_perceiverio_conv_pytorch(clear_pybuda, test_device, batch_size):
    run_perceiverio_conv_pytorch(batch_size=batch_size)


@pytest.mark.perceiverio
def test_perceiverio_fourier_pytorch(clear_pybuda, test_device, batch_size):
    run_perceiverio_fourier_pytorch(batch_size=batch_size)


@pytest.mark.perceiverio
def test_perceiverio_learned_pytorch(clear_pybuda, test_device, batch_size):
    run_perceiverio_learned_pytorch(batch_size=batch_size)
