# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.dla.onnx_dla import run_dla_onnx

variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60x_c",
    "dla60",
    "dla60x",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.dla
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dla_onnx(clear_pybuda, test_device, variant):
    run_dla_onnx(variant)
