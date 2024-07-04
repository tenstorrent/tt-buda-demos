# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.dla.pytorch_dla import run_dla_pytorch

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
def test_dla_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_dla_pytorch(variant, batch_size=batch_size)
