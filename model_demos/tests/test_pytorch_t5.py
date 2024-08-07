# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.t5.pytorch_t5_generation import run_t5_pybuda_pipeline

variants = ["t5-small", "t5-base", "t5-large"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.t5
def test_t5_pybuda_pipeline_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_t5_pybuda_pipeline(variant, batch_size=batch_size)
