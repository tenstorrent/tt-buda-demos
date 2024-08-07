# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.segformer.pytorch_segformer_image_classification import run_segformer_image_classification_pytorch
from cv_demos.segformer.pytorch_segformer_semantic_segmentation import run_segformer_semseg_pytorch

variants_semseg = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.parametrize("variant", variants_semseg, ids=variants_semseg)
@pytest.mark.segformer
def test_segformer_seg_pytorch(variant, clear_pybuda, test_device, batch_size):
    run_segformer_semseg_pytorch(variant, batch_size=batch_size)


variants_img_classification = [
    "nvidia/mit-b0",
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.parametrize("variant", variants_img_classification, ids=variants_img_classification)
@pytest.mark.segformer
def test_segformer_image_classification_pytorch(variant, clear_pybuda, test_device, batch_size):
    run_segformer_image_classification_pytorch(variant, batch_size=batch_size)
