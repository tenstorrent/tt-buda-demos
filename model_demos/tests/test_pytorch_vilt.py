# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.vilt.pytorch_vilt_maskedlm import run_vilt_maskedlm_pytorch
from cv_demos.vilt.pytorch_vilt_question_answering import run_vilt_for_question_answering_pytorch


@pytest.mark.vilt
def test_vilt_for_question_answering_pytorch(clear_pybuda, test_device, batch_size):
    run_vilt_for_question_answering_pytorch(batch_size=batch_size)


@pytest.mark.vilt
def test_vilt_maskedlm_pytorch(clear_pybuda, test_device, batch_size):
    run_vilt_maskedlm_pytorch(batch_size=batch_size)
