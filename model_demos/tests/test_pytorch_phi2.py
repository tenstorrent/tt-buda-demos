# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.phi2.pytorch_phi2_text_generation import run_phi2_causal_lm

@pytest.mark.qwen1_5
def test_qwen1_5_causal_lm_pytorch(clear_pybuda, test_device, batch_size):
    run_phi2_causal_lm(batch_size=batch_size)