# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.inception_v4.pytorch_inception_v4_osmr import run_inception_v4_osmr_pytorch
from cv_demos.inception_v4.pytorch_inception_v4_timm import run_inception_v4_timm_pytorch


@pytest.mark.inceptionv4
def test_inceptionv4_osmr(clear_pybuda, test_device):
    run_inception_v4_osmr_pytorch()


@pytest.mark.inceptionv4
def test_inceptionv4_timm(clear_pybuda, test_device):
    run_inception_v4_timm_pytorch()
