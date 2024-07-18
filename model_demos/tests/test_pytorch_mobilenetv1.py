# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.mobilenet_v1.pytorch_mobilenet_v1_basic import run_mobilenetv1_basic
from cv_demos.mobilenet_v1.pytorch_mobilenet_v1_hf import run_mobilenetv1_hf

variants = [
    "google/mobilenet_v1_0.75_192",
    "google/mobilenet_v1_1.0_224",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.mobilenetv1
def test_mobilenetv1_hf_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_mobilenetv1_hf(variant, batch_size=batch_size)


@pytest.mark.mobilenetv1
def test_mobilenetv1_basic_pytorch(clear_pybuda, test_device, batch_size):
    run_mobilenetv1_basic(batch_size=batch_size)
