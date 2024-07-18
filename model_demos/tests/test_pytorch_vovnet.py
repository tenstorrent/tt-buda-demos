# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from cv_demos.vovnet.pytorch_vovnet_ese_19bdw_hf_timm import run_vovnet_ese_19bdw_timm_pytorch
from cv_demos.vovnet.pytorch_vovnet_ese_39b_hf_timm import run_vovnet_ese_39b_timm_pytorch
from cv_demos.vovnet.pytorch_vovnet_ese_99b_hf_timm import run_vovnet_ese_99b_timm_pytorch
from cv_demos.vovnet.pytorch_vovnet_v1_27s_osmr import run_vovnet_v1_27s_osmr_pytorch
from cv_demos.vovnet.pytorch_vovnet_v1_39_osmr import run_vovnet_v1_39_osmr_pytorch
from cv_demos.vovnet.pytorch_vovnet_v1_57_osmr import run_vovnet_v1_57_osmr_pytorch

sys.path.append("cv_demos/vovnet")


@pytest.mark.vovnet
def test_vovnet_v1_27s_osmr_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_v1_27s_osmr_pytorch(batch_size=batch_size)


@pytest.mark.vovnet
def test_vovnet_v1_39_osmr_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_v1_39_osmr_pytorch(batch_size=batch_size)


@pytest.mark.vovnet
def test_vovnet_v1_57_osmr_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_v1_57_osmr_pytorch(batch_size=batch_size)


@pytest.mark.vovnet
def test_vovnet_ese_19bdw_timm_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_ese_19bdw_timm_pytorch(batch_size=batch_size)


@pytest.mark.vovnet
def test_vovnet_ese_39b_timm_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_ese_39b_timm_pytorch(batch_size=batch_size)


@pytest.mark.vovnet
def test_vovnet_ese_99b_timm_pytorch(clear_pybuda, test_device, batch_size):
    run_vovnet_ese_99b_timm_pytorch(batch_size=batch_size)
