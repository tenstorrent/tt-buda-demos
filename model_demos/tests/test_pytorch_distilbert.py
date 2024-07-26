# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.distilbert.pytorch_distilbert_masked_lm import run_distilbert_masked_lm_pytorch
from nlp_demos.distilbert.pytorch_distilbert_question_answering import run_distilbert_question_answering_pytorch
from nlp_demos.distilbert.pytorch_distilbert_sequence_classification import (
    run_distilbert_sequence_classification_pytorch,
)
from nlp_demos.distilbert.pytorch_distilbert_token_classification import run_distilbert_token_classification_pytorch

variants = ["distilbert-base-uncased", "distilbert-base-cased", "distilbert-base-multilingual-cased"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.distilbert
def test_distilbert_masked_lm_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_distilbert_masked_lm_pytorch(variant, batch_size=batch_size)


@pytest.mark.distilbert
def test_distilbert_question_answering_pytorch(clear_pybuda, test_device, batch_size):
    run_distilbert_question_answering_pytorch(batch_size=batch_size)


@pytest.mark.distilbert
def test_distilbert_sequence_classification_pytorch(clear_pybuda, test_device, batch_size):
    run_distilbert_sequence_classification_pytorch(batch_size=batch_size)


@pytest.mark.distilbert
def test_distilbert_token_classification_pytorch(clear_pybuda, test_device, batch_size):
    run_distilbert_token_classification_pytorch(batch_size=batch_size)
