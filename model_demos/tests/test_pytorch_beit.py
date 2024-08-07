# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.beit.pytorch_beit_classify_16_224_hf import run_beit_classify_224_hf_pytorch

variants = ["microsoft/beit-base-patch16-224", "microsoft/beit-large-patch16-224"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.beit
def test_beit_classify_224_hf_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_beit_classify_224_hf_pytorch(variant, batch_size=batch_size)
