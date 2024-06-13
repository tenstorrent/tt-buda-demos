# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.resnet.pytorch_resnet import run_resnet_pytorch
from cv_demos.resnet.timm_resnet import run_resnet_timm

variants = [
    "microsoft/resnet-18",
    "microsoft/resnet-26",
    "microsoft/resnet-34",
    "microsoft/resnet-50",
    "microsoft/resnet-101",
    "microsoft/resnet-152",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.resnet
def test_resnet_pytorch(clear_pybuda, test_device, variant):
    run_resnet_pytorch(variant)


@pytest.mark.resnet
def test_resnet_timm(clear_pybuda, test_device):
    run_resnet_timm()
