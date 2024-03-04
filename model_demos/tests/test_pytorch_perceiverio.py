import pytest

from cv_demos.perceiverio.pytorch_perceiverio_for_image_classification import (
    run_perceiverio_for_image_classification_pytorch,
)

variants = ["deepmind/vision-perceiver-conv"]

variants = ["deepmind/vision-perceiver-conv"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.perceiverio
def test_perceiverio_image_classification_pytorch(clear_pybuda, variant):
    run_perceiverio_pytorch(variant)
