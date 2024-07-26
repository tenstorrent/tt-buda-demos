# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.gpt_neo.pytorch_gptneo_causal_lm import run_gptneo_causal_lm
from nlp_demos.gpt_neo.pytorch_gptneo_sequence_classification import run_gptneo_sequence_classification

variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.gptneo
def test_gptneo_causal_lm_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_gptneo_causal_lm(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.gptneo
def test_gptneo_sequence_classification_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_gptneo_sequence_classification(variant, batch_size=batch_size)
