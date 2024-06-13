import pytest
from cv_demos.retinanet.pytorch_retinanet import run_retinanet_pytorch

variants = ["retinanet_rn18fpn", "retinanet_rn34fpn", "retinanet_rn50fpn", "retinanet_rn101fpn", "retinanet_rn152fpn"]


@pytest.mark.retinanet
@pytest.mark.parametrize("variant", variants)
def test_retinanet_pytorch(variant, clear_pybuda, test_device):
    run_retinanet_pytorch(variant)
