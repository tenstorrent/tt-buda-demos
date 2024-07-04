# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.xglm.pytorch_xglm_causal_lm import run_xglm_causal_lm

variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xglm
def test_xglm_causal_lm_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_xglm_causal_lm(variant, batch_size=batch_size)
