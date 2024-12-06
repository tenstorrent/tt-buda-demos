# TT-Buda Installation

## Overview

The TT-Buda software stack can compile models from several different frameworks and execute them in many different ways on Tenstorrent hardware.

This user guide is intended to help you setup your system with the appropriate device drivers, firmware, system dependencies, and compiler / runtime software.

**Note on terminology:**

While TT-Buda is the official Tenstorrent AI/ML compiler stack, PyBuda is the Python interface for TT-Buda. TT-Buda is the core technology; however, PyBuda allows users to access and utilize TT-Buda's features directly from Python. This includes directly importing model architectures and weights from PyTorch, TensorFlow, ONNX, and TFLite.

### OS Compatibility

Currently, Tenstorrent software is supported on **Ubuntu 20.04 LTS (Focal Fossa)** and **Ubuntu 22.04 LTS (Jammy Jellyfish)** operating systems.

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
   6. [Topology (TT-LoudBox/TT-QuietBox Only)](#tt-topology-tt-loudboxtt-quietbox-systems-only)
2. [PyBuda Installation](#pybuda-installation)
   1. [Python Environment Installation](#python-environment-installation)
   2. [Docker Container Installation](#docker-container-installation)
3. [Tests](#tests)
   1. [Smoke Test](#smoke-test)

## Installation Instructions

PyBuda can be installed using two methods: Docker or Python virtualenv.

If you would like to run PyBuda in a Docker container, then follow the instructions for [PCI Driver Installation](#pci-driver-installation) and [Device Firmware Update](#device-firmware-update) and followed by [Docker Container Installation](#docker-container-installation).

If you would like to run PyBuda in a Python virtualenv, then follow the instructions for the [Setup HugePages](#setup-hugepages), [PCI Driver Installation](#pci-driver-installation), [Device Firmware Update](#device-firmware-update), and [Backend Compiler Dependencies](#backend-compiler-dependencies), followed by the [Python Environment Installation](#python-environment-installation).

### Setup HugePages

```bash
# Clone System Tools Repo
git clone https://github.com/tenstorrent/tt-system-tools.git
cd tt-system-tools
chmod +x hugepages-setup.sh
sudo ./hugepages-setup.sh

# Install `.deb`
wget https://github.com/tenstorrent/tt-system-tools/releases/download/upstream%2F1.1/tenstorrent-tools_1.1-5_all.deb
sudo dpkg -i tenstorrent-tools_1.1-5_all.deb

# Start Services
sudo systemctl enable --now tenstorrent-hugepages.service
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount'

# System Reboot
sudo reboot
```

### PCI Driver Installation

Please navigate to [tt-kmd](https://github.com/tenstorrent/tt-kmd) homepage and follow instructions within the README.
_Pro-Tip: ensure that you are within the home directory of the local clone version of [tt-kmd](https://github.com/tenstorrent/tt-kmd) when performing the installation steps_

### Device Firmware Update

The [tt-firmware](https://github.com/tenstorrent/tt-firmware) file needs to be installed using the [tt-flash](https://github.com/tenstorrent/tt-flash) utility, for more details visit [TT-Flash homepage](https://github.com/tenstorrent/tt-flash?tab=readme-ov-file#firmware-files:~:text=Firmware%20files,of%20the%20images.) and follow instructions within the README.

### Backend Compiler Dependencies

Instructions to install the Tenstorrent backend compiler dependencies on a fresh install of Ubuntu Server 20.04 or Ubuntu Server 22.04.

_You may need to **append** each `apt` command with `sudo` if you do not have root permissions._

For both operating systems run the following commands:

```bash
apt update -y
apt upgrade -y --no-install-recommends
apt install -y build-essential curl libboost-all-dev libgl1-mesa-glx libgoogle-glog-dev libhdf5-serial-dev ruby software-properties-common libzmq3-dev clang wget python3-pip python-is-python3 python3-venv

```

For Ubuntu 20.04, add:

```bash
apt install -y libyaml-cpp-dev
```

For Ubuntu 22.04, add:

```bash
wget http://mirrors.kernel.org/ubuntu/pool/main/y/yaml-cpp/libyaml-cpp-dev_0.6.2-4ubuntu1_amd64.deb
wget http://mirrors.kernel.org/ubuntu/pool/main/y/yaml-cpp/libyaml-cpp0.6_0.6.2-4ubuntu1_amd64.deb
dpkg -i libyaml-cpp-dev_0.6.2-4ubuntu1_amd64.deb libyaml-cpp0.6_0.6.2-4ubuntu1_amd64.deb
rm libyaml-cpp-dev_0.6.2-4ubuntu1_amd64.deb libyaml-cpp0.6_0.6.2-4ubuntu1_amd64.deb
```

### TT-SMI

Please navigate to [tt-smi](https://github.com/tenstorrent/tt-smi) homepage and follow instructions within the README.

### TT-Topology (TT-LoudBox/TT-QuietBox Systems Only)

If you are running on a TT-LoudBox or TT-QuietBox system, please navigate to [tt-topology](https://github.com/tenstorrent/tt-topology) homepage and follow instructions within the README.

## PyBuda Installation

There are two ways to install PyBuda within the host environment: using Python virtual environment or Docker container.

### Python Environment Installation

It is strongly recommended to use virtual environments for each project utilizing PyBuda and
Python dependencies. Creating a new virtual environment with PyBuda and libraries is very easy.

#### Step 1. Navigate to [Releases](https://github.com/tenstorrent/tt-buda/releases)

#### Step 2. Scroll to find the latest release package in `.zip` format under "Assets" that corresponds to your device and operating system

#### Step 3. Download the `.zip` package and unzip to find the `pybuda`, `tvm` and `torchvison` wheel files

#### Step 4. Create your Python environment in desired directory

```bash
python3 -m venv env
```

#### Step 5. Activate environment

```bash
source env/bin/activate
```

#### Step 5. Pip install PyBuda, TVM and Torchvision whl files

```bash
pip install --upgrade pip==24.0
pip install torchvision-<version>.whl pybuda-<version>.whl tvm-<version>.whl 
```

The `pybuda-<version>.whl` file contains the PyBuda library, the `tvm-<version>.whl` file contains the latest TVM downloaded release, and the `torchvision-<version>.whl` file bundles the torchvision library.

#### Step 6. Pip install Debuda (Optional Step)

For enhanced debugging capabilities, you may opt to install the Debuda library:

```bash
pip install debuda-<version>.whl
```

This wheel file installs the Debuda tool designed for debugging purposes.

---

### Docker Container Installation

Alternatively, PyBuda and its dependencies are provided as Docker images which can run in separate containers. The Docker containers can be found under: <https://github.com/orgs/tenstorrent/packages?repo_name=tt-buda>

#### Step 1. Pull the docker image

To pull the Docker image, use the following command:

```bash
sudo docker pull ghcr.io/tenstorrent/tt-buda/<OS-VERSION>/<TT-DEVICE>:<TAG>
```

Supported OS `<OS-VERSION>` Versions:

- ubuntu-20-04-amd64
- ubuntu-22-04-amd64

Supported Tenstorrent `<TT-DEVICE>` Devices:

- gs
- wh_b0

For example, to run on an Ubuntu version 20.04 on a Grayskull device, use this command:

```bash
sudo docker pull ghcr.io/tenstorrent/tt-buda/ubuntu-20-04-amd64/gs:<TAG>
```

where `<TAG>` is the release version number from: <https://github.com/tenstorrent/tt-buda/tags>

#### Step 2. Run the container

```bash
sudo docker run --rm -ti --cap-add=sys_nice --shm-size=4g --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -v $(pwd)/:/home/ ghcr.io/tenstorrent/tt-buda/<OS-VERSION>/<TT-DEVICE>:<TAG> bash
```

#### Step 3. Change root directory

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
