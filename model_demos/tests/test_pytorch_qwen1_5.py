# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.qwen1_5.pytorch_qwen1_5 import run_qwen1_5_causal_lm
from nlp_demos.qwen1_5.pytorch_qwen1_5_chat import run_qwen1_5_chat


@pytest.mark.qwen1_5
def test_qwen1_5_causal_lm_pytorch(clear_pybuda, test_device, batch_size):
    run_qwen1_5_causal_lm(batch_size=batch_size)


@pytest.mark.qwen1_5
def test_qwen1_5_chat_pytorch(clear_pybuda, test_device, batch_size):
    run_qwen1_5_chat(batch_size=batch_size)
