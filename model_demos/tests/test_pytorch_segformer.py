import pytest
from cv_demos.segformer.pytorch_segformer import run_segformer_pytorch

variants = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.segformer
def test_segformer_pytorch(variant, clear_pybuda):
    run_segformer_pytorch(variant)
