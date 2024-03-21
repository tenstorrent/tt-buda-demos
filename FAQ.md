# FAQ & Troubleshooting Guide

## Resetting an accelerator board

If you encounter a situation where a Tenstorrent chip appears to be unresponsive or is exhibiting unusual behavior, a software reset of the board might be a viable solution.


For a software reset on a single chip use : `tt-smi -lr 0` .

For more information on performing reset on multiple chips or other specifics visit [TT-SMI Resets](https://github.com/tenstorrent/tt-smi?tab=readme-ov-file#resets:~:text=on%20the%20footer.-,Resets,-Another%20feature%20of)

If you need additional assistance, you can access a detailed explanation of all available command options by appending the help flag to the command like so: `tt-smi --help` or `tt-smi -h`.

For comprehensive insights and detailed instructions on utilizing the command line GUI, we invite you to explore the Tenstorrent System Management Interface (TT-SMI) repository on GitHub at [TT-SMI homepage](https://github.com/tenstorrent/tt-smi). TT-SMI serves as a versatile command-line utility tailored to streamline interaction with all Tenstorrent devices on host.

If the software reset fails to resolve the issue, the next step would be to power cycle the board. This typically involves rebooting the host machine that the Tenstorrent board is connected to. 
*Please note that any unsaved work may be lost during this process, so ensure all important data is saved before proceeding*

## `PermissionError` on `/tmp/*.lock` files

If multiple users are running on a system with a shared Tenstorrent device,
you may encounter a `PermissionError: [Errno 13] Permission denied: '/tmp/*.lock'`.

You would need to remove these files between user sessions.

## Python pip install dependency resolution issues

Your Python environment's pip version, if different than required version, have different [dependency resolution](https://pip.pypa.io/en/stable/topics/dependency-resolution/) and not correctly install pybuda (tt-buda) or its dependencies.
To avoid dependency resolution issues check you are using the supported and tested version of pip for your tt-buda release. For example, pip==24.0 is required for the most recent release (v0.10.5). To avoid changing the system installation of pip we generally recommend using the venv installation method (see guide in https://github.com/tenstorrent/tt-buda-demos/blob/main/first_5_steps/1_install_tt_buda.md#python-environment-installation).

```bash
# check your pip version
pip --version
# install a specific version of pip, e.g. 24.0 required for tt-buda v0.10.5
pip install pip==24.0
```

For reference `pip install` error messages for dependency resolution issues with an unsupported pip version can look like:
```
ERROR: Invalid requirement: 'torch@ https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.1.0%2Bcpu.cxx11.abi-cp38-cp38-linux_x86_64.whl; python_version == "3.8"'
```
or:
```
ERROR: Package 'networkx' requires a different Python: 3.8.10 not in '>=3.9'
```
