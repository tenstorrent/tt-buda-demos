# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.ghostnet.timm_ghostnet import run_ghostnet_timm


@pytest.mark.ghostnet
def test_ghostnet_timm_pytorch(clear_pybuda):
    run_ghostnet_timm()
