# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.wideresnet.pytorch_wideresnet_timm import run_wideresnet_timm_pytorch
from cv_demos.wideresnet.pytorch_wideresnet_torchhub import run_wideresnet_torchhub_pytorch

variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.wideresnet
def test_wideresnet_torchhub_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_wideresnet_torchhub_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.wideresnet
def test_wideresnet_timm_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_wideresnet_timm_pytorch(variant, batch_size=batch_size)
