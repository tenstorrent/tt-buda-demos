# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.mlpmixer.timm_mlpmixer import run_mlpmixer_timm


@pytest.mark.mlpmixer
def test_mlpmixer_timm(clear_pybuda, test_device):
    run_mlpmixer_timm()
