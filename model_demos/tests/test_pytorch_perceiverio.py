# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.perceiverio.pytorch_perceiverio import run_perceiverio_pytorch

variants = ["deepmind/vision-perceiver-conv"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.perceiverio
def test_perceiverio_image_classification_pytorch(clear_pybuda, variant):
    run_perceiverio_pytorch(variant)
