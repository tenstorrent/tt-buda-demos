# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.albert.pytorch_albert_masked_lm import run_albert_masked_lm_pytorch
from nlp_demos.albert.pytorch_albert_question_answering import run_albert_question_answering_pytorch
from nlp_demos.albert.pytorch_albert_sequence_classification import run_albert_sequence_classification_pytorch
from nlp_demos.albert.pytorch_albert_token_classification import run_albert_token_classification_pytorch

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
@pytest.mark.albert
def test_albert_masked_lm_pytorch(clear_pybuda, test_device, size, variant, batch_size):
    print(f"batch size: {batch_size}")
    run_albert_masked_lm_pytorch(size, variant, batch_size=batch_size)


@pytest.mark.albert
def test_albert_question_answering_pytorch(clear_pybuda, test_device, batch_size):
    run_albert_question_answering_pytorch(batch_size=batch_size)


@pytest.mark.albert
def test_albert_sequence_classification_pytorch(clear_pybuda, test_device, batch_size):
    run_albert_sequence_classification_pytorch(batch_size=batch_size)


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
@pytest.mark.albert
def test_albert_token_classification_pytorch(clear_pybuda, test_device, size, variant, batch_size):
    run_albert_token_classification_pytorch(size, variant, batch_size=batch_size)
