# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.roberta.pytorch_roberta_masked_lm import run_roberta_mlm_pytorch
from nlp_demos.roberta.pytorch_roberta_sentiment import run_roberta_sentiment_pytorch


@pytest.mark.roberta
def test_roberta_mlm_pytorch(clear_pybuda, test_device, batch_size):
    run_roberta_mlm_pytorch(batch_size=batch_size)


@pytest.mark.roberta
def test_roberta_sentiment_pytorch(clear_pybuda, test_device, batch_size):
    run_roberta_sentiment_pytorch(batch_size=batch_size)
