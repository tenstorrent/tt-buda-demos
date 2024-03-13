# TT-Buda Installation

## Overview

The TT-Buda software stack can compile models from several different frameworks and execute them in many different ways on Tenstorrent hardware.

This user guide is intended to help you setup your system with the appropriate device drivers, firmware, system dependencies, and compiler / runtime software.

**Note on terminology:**

While TT-Buda is the official Tenstorrent AI/ML compiler stack, PyBuda is the Python interface for TT-Buda. TT-Buda is the core technology; however, PyBuda allows users to access and utilize TT-Buda's features directly from Python. This includes directly importing model architectures and weights from PyTorch, TensorFlow, ONNX, and TFLite.

### OS Compatibility

Presently, Tenstorrent software is only supported on the **Ubuntu 20.04 LTS (Focal Fossa)** operating system.

## Release Versions

To access the PyBuda software and associated files, please navigate to the [Releases](https://github.com/tenstorrent/tt-Buda/releases) section of this repository.

Once you have identified the release version you would like to install, you can download the individual files by clicking on their name.

## Table Of Contents

1. [Installation Instructions](#installation-instructions)
   1. [Setup HugePages](#setup-hugepages)
   2. [PCI Driver Installation](#pci-driver-installation)
   3. [Device Firmware Update](#device-firmware-update)
   4. [Backend Compiler Dependencies](#backend-compiler-dependencies)
   5. [TT-SMI](#tt-smi)
2. [PyBuda Installation](#pybuda-installation)
    1. [Python Environment Installation](#python-environment-installation)
    2. [Docker Container Installation](#docker-container-installation)
3. [Tests](#tests)
   1. [Smoke Test](#smoke-test)

## Installation Instructions

PyBuda can be installed using two methods: Docker or Python virtualenv.

If you would like to run PyBuda in a Docker container, then follow the instructions for [PCI Driver Installation](#pci-driver-installation) and [Device Firmware Update](#device-firmware-update) and followed by [Docker Container Installation](#docker-container-installation).

If you would like to run PyBuda in a Python virtualenv, then follow the instructions for the [Setup HugePages](#setup-hugepages), [PCI Driver Installation](#pci-driver-installation), [Device Firmware Update](#device-firmware-update), and [Backend Compiler Dependencies](#backend-compiler-dependencies), followed by the [Tenstorrent Software Package](#tenstorrent-software-package).

### Setup HugePages

```bash
NUM_DEVICES=$(lspci -d 1e52: | wc -l)
sudo sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=.*$/GRUB_CMDLINE_LINUX_DEFAULT=\"hugepagesz=1G hugepages=${NUM_DEVICES} nr_hugepages=${NUM_DEVICES} iommu=pt\"/g" /etc/default/grub
sudo update-grub
sudo sed -i "/\s\/dev\/hugepages-1G\s/d" /etc/fstab; echo "hugetlbfs /dev/hugepages-1G hugetlbfs pagesize=1G,rw,mode=777 0 0" | sudo tee -a /etc/fstab
sudo reboot
```

### PCI Driver Installation

Please navigate to [tt-kmd](https://github.com/tenstorrent/tt-kmd) homepage and follow instructions within the README. 
*Pro-Tip: ensure that you are within the home directory of the local clone version of [tt-kmd](https://github.com/tenstorrent/tt-kmd) when performing the installation steps*

### Device Firmware Update

The [tt-firmware](https://github.com/tenstorrent/tt-firmware) file needs to be installed using the [tt-flash](https://github.com/tenstorrent/tt-flash) utility, for more details visit [TT-Flash homepage](https://github.com/tenstorrent/tt-flash?tab=readme-ov-file#firmware-files:~:text=Firmware%20files,of%20the%20images.) and follow instructions within the README.

### Backend Compiler Dependencies

Instructions to install the Tenstorrent backend compiler dependencies on a fresh install of Ubuntu Server 20.04.

*You may need to **append** each `apt-get` command with `sudo` if you do not have root permissions.*

```bash
apt-get update -y
apt-get upgrade -y --no-install-recommends
apt-get install -y software-properties-common
apt-get install -y python3.8-venv libboost-all-dev libgoogle-glog-dev libgl1-mesa-glx libyaml-cpp-dev ruby
apt-get install -y build-essential clang-6.0 libhdf5-serial-dev libzmq3-dev
```

### TT-SMI

Please navigate to [tt-smi](https://github.com/tenstorrent/tt-smi) homepage and follow instructions within the README.

## PyBuda Installation

There are two ways to install PyBuda within the host environment: using Python virtual environment or Docker container.


### Python Environment Installation

It is strongly recommended to use virtual environments for each project utilizing PyBuda and
Python dependencies. Creating a new virtual environment with PyBuda and libraries is very easy.

#### Step 1. Navigate to [Releases](https://github.com/tenstorrent/tt-buda/releases)

#### Step 2. Under the latest release, download the `pybuda` and `tvm` wheel files

#### Step 3. Create your Python environment in desired directory

```bash
python3 -m venv env
```

#### Step 4. Activate environment

```bash
source env/bin/activate
```

#### Step 5. Upgade the pip installer 

```bash
pip install --upgrade pip==24.0
```
The `pybuda-<version>.whl` file contains the PyBuda and `tvm-<version>.whl` file contains the latest TVM downloaded release(s).

---

#### Step 6. Pip install PyBuda, TVM and Torchvision whl files    

```bash
pip install pybuda-<version>.whl tvm-<version>.whl torchvision-<version>.whl
```
The `pybuda-<version>.whl` file contains the PyBuda and `tvm-<version>.whl` file contains the latest TVM downloaded release(s).

---

### Docker Container Installation

Alternatively, PyBuda and its dependencies are provided as Docker images which can run in separate containers.

#### Step 1. Setup a personal access token (classic)

Create a personal access token from: [GitHub Tokens](https://github.com/settings/tokens).
Give the token the permissions to read packages from the container registry `read:packages`.

#### Step 2. Login to Docker Registry

```bash
GITHUB_TOKEN=<your token>
echo $GITHUB_TOKEN | sudo docker login ghcr.io -u <your github username> --password-stdin
```

#### Step 3. Pull the image

```bash
sudo docker pull ghcr.io/tenstorrent/tt-buda/<TAG>
```

#### Step 4. Run the container

```bash
sudo docker run --rm -ti --cap-add=sys_nice --shm-size=4g --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -v `pwd`/:/home/ ghcr.io/tenstorrent/tt-buda/<TAG> bash
```

#### Step 5. Change root directory

```bash
cd home/
```

## Tests

Verify the correct installation of the PyBuda library and environment by conducting a smoke test.

### Smoke Test

With your Python environment with PyBuda install activated, run the following Python script:

```python
import pybuda
import torch


# Sample PyTorch module
class PyTorchTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights1 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
        self.weights2 = torch.nn.Parameter(torch.rand(32, 32), requires_grad=True)
    def forward(self, act1, act2):
        m1 = torch.matmul(act1, self.weights1)
        m2 = torch.matmul(act2, self.weights2)
        return m1 + m2, m1


def test_module_direct_pytorch():
    input1 = torch.rand(4, 32, 32)
    input2 = torch.rand(4, 32, 32)
    # Run single inference pass on a PyTorch module, using a wrapper to convert to PyBuda first
    output = pybuda.PyTorchModule("direct_pt", PyTorchTestModule()).run(input1, input2)
    print(output)
    print("PyBuda installation was a success!")


if __name__ == "__main__":
    test_module_direct_pytorch()
```
