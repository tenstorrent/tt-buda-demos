# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.falcon.pytorch_falcon import run_falcon_pytorch


@pytest.mark.falcon
def test_falcon_pytorch(clear_pybuda, test_device):
    run_falcon_pytorch()
