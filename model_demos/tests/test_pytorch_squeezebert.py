# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.squeezebert.pytorch_squeezebert_sequence_classification import (
    run_squeezebert_sequence_classification_pytorch,
)


@pytest.mark.squeezebert
def test_squeezebert_sequence_classification_pytorch(clear_pybuda, test_device):
    run_squeezebert_sequence_classification_pytorch()
