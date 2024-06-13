# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.xception.timm_xception import run_xception_timm

variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xception
def test_xception_timm_pytorch(clear_pybuda, test_device, variant):
    run_xception_timm(variant)
