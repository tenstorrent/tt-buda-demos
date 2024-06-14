# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import pybuda.compile as COMPILE_INFO

import pybuda
import pytest
from pybuda._C.backend_api import BackendDevice
from pybuda._C.backend_api import BackendType, BackendDevice, DeviceMode
from pybuda.run.api import detect_available_devices


@pytest.fixture(autouse=True)
def clear_pybuda():
    yield

    # Clean up after each test
    pybuda.shutdown()
    pybuda.pybuda_reset()
    # TODO: For running on Silicon, reset tensix cores after each test
    # _ = subprocess.run(
    #     ["external_libs/pybuda/third_party/budabackend/device/bin/silicon/tensix-reset"]
    # )
    archive_files()
    _ = subprocess.run(["make", "clean_tt"])


def archive_files(src_directory=Path("./"), dest_directory=Path("archive")):
    """
    Archive files post run "_netlist.yaml" to dest_directory.

    Args:
    - src_directory (Path or str): The source directory where to look for files.
    - dest_directory (Path or str): The destination directory where to copy files. Defaults to "archive".
    """
    src_directory = Path(src_directory)
    dest_directory = Path(dest_directory)
    if not src_directory.exists():
        raise ValueError(f"Source directory {src_directory} does not exist!")

    if not dest_directory.exists():
        dest_directory.mkdir(parents=True)

    for file_path in src_directory.glob("*_netlist.yaml"):
        dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dest_path = dest_directory / f"{file_path.stem}_{dt_str}{file_path.suffix}"
        try:
            shutil.copy(file_path, dest_path)
            print(f"Copied {file_path} to {dest_directory}")
        except Exception as e:
            print(f"Failed to copy {file_path}. Reason: {e}")


DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE = {
    "gs_e150": BackendDevice.Grayskull,
    "gs_e300": BackendDevice.Grayskull,
    "wh_n150": BackendDevice.Wormhole_B0,
    "wh_n300": BackendDevice.Wormhole_B0,
    "galaxy": BackendDevice.Wormhole_B0,
}

@dataclass
class TestDevice:
    devtype: BackendType
    arch: BackendDevice
    devmode: DeviceMode
    tti_path: str = None

    @classmethod
    def from_str(cls, name: str, devmode: DeviceMode, tti_path: str = None, device_config=None) -> "TestDevice":
        if name == "Golden":
            if device_config and DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE.get(device_config, None):
                return TestDevice(devtype=BackendType.Golden, arch=DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE[device_config], devmode=devmode, tti_path=tti_path)
            elif "GOLDEN_WORMHOLE_B0" in os.environ:
                return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
            elif "PYBUDA_GOLDEN_BLACKHOLE" in os.environ:
                return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
            return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
        if name == "Model":
            return TestDevice(devtype=BackendType.Model, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
        if name == "Versim":
            # Set default versim device arch to Grayskull
            versim_backend_device = BackendDevice.Grayskull
            # If PYBUDA_VERSIM_DEVICE_ARCH is set, use that arch for Versim device
            versim_arch_name = os.environ.get("PYBUDA_VERSIM_DEVICE_ARCH", None)
            if versim_arch_name != None:
                versim_backend_device = BackendDevice.from_string(versim_arch_name)
            return TestDevice(devtype=BackendType.Versim, arch=versim_backend_device, devmode=devmode, tti_path=tti_path)
        if name == "Emulation":
            # Set default emulation device arch to Grayskull
            emulation_backend_device = BackendDevice.Grayskull
            # If PYBUDA_EMULATION_DEVICE_ARCH is set, use that arch for Emulation device
            emulation_arch_name = os.environ.get("PYBUDA_EMULATION_DEVICE_ARCH", None)
            if emulation_arch_name != None:
                emulation_backend_device = BackendDevice.from_string(emulation_arch_name)
            return TestDevice(devtype=BackendType.Emulation, arch=emulation_backend_device, devmode=devmode, tti_path=tti_path)
        if name == "Grayskull":
            return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
        if name == "Wormhole_B0":
            return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
        if name == "Blackhole":
            return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
        raise RuntimeError("Unknown test device: " + name)

    def is_available(self, device_list: List[BackendDevice], silicon_only: bool, no_silicon: bool, devtype: Optional[BackendType], devmode: DeviceMode) -> bool:
        """ 
        Return true if this kind of device is available on the current host. Expect a list of devices from 
        `detect_available_devices`.
        """
        if devtype is not None and self.devtype != devtype:
            return False

        if self.devtype == BackendType.Golden:
            return not silicon_only

        if self.devtype == BackendType.Model:
            return bool(int(os.environ.get("PYBUDA_ENABLE_MODEL_DEVICE", "0")))
        
        if self.devtype == BackendType.Versim:
            return bool(int(os.environ.get("PYBUDA_ENABLE_VERSIM_DEVICE", "0")))

        if self.devtype == BackendType.Emulation:
            return bool(int(os.environ.get("PYBUDA_ENABLE_EMULATION_DEVICE", "0")))

        if self.devtype == BackendType.Silicon:
            compiled_arch_name = os.environ.get("BACKEND_ARCH_NAME", None) or os.environ.get("ARCH_NAME", None)
            if compiled_arch_name == "wormhole_b0":
                compiled_arch = BackendDevice.Wormhole_B0
            elif compiled_arch_name == "blackhole":
                compiled_arch = BackendDevice.Blackhole
            else:
                compiled_arch = BackendDevice.Grayskull

            is_offline_silicon_compile = devmode == DeviceMode.CompileOnly and self.arch == compiled_arch
            return (self.arch in device_list and not no_silicon) or is_offline_silicon_compile

        return False

    def is_silicon(self):
        return self.devtype == BackendType.Silicon

    def is_grayskull(self):
        return self.arch == BackendDevice.Grayskull
    
    def is_wormhole_b0(self):
        return self.arch == BackendDevice.Wormhole_B0
    
    def is_blackhole(self):
        return self.arch == BackendDevice.Blackhole


device_cfg_global = None
def pytest_generate_tests(metafunc):
    global device_cfg_global

    # Configure backend runtime yaml
    device_cfg_global = metafunc.config.getoption("--device-config")

    if "test_device" in metafunc.fixturenames:

        names = ["Golden", "Model", "Versim", "Emulation", "Grayskull", "Wormhole_B0", "Blackhole"]

        # Set device-mode for the test
        compile_only = metafunc.config.getoption("--compile-only")
        run_only = metafunc.config.getoption("--run-only")
        devtype = metafunc.config.getoption("--devtype")
        devtype = BackendType.from_string(devtype.capitalize()) if devtype else None

        devmode = DeviceMode.CompileAndRun
        if compile_only:
            devmode = DeviceMode.CompileOnly
            if devtype is None:
                assert False, "Backend device type needs to be specified when running tests with compile-only mode"
        elif run_only:
            devmode = DeviceMode.RunOnly

        # Configure TTI-path only if compile/run-only is set
        tti_path = None
        if compile_only or run_only:
            tti_path = metafunc.config.getoption("--tti-path")

        devices = [(TestDevice.from_str(s, devmode, tti_path, device_cfg_global), s) for s in names]
        silicon_only = metafunc.config.getoption("--silicon-only")
        no_silicon = metafunc.config.getoption("--no-silicon")
        device_list = []
        if not no_silicon:
            device_list = detect_available_devices()
        enabled_devices = [(d, name) for (d, name) in devices if d.is_available(device_list, silicon_only, no_silicon, devtype, devmode)]
        params = [pytest.param(d) for (d, _) in enabled_devices]
        ids = [name for (_, name) in enabled_devices]

        metafunc.parametrize("test_device", params, ids=ids)


environ_before_test = None
def pytest_runtest_logreport(report):
    if report.when == "setup":
        global environ_before_test
        environ_before_test = os.environ.copy()
        
        global device_cfg_global
        if device_cfg_global:
            pybuda.set_configuration_options(device_config=device_cfg_global)

        if "PYBUDA_OVERRIDES_VETO" in os.environ:
            from pybuda.config import _set_pybuda_override_veto

            # This functionality represents one way to control general and env based compiler configuration (enable us to 
            # add/update/remove existing configs in each test with ease during runtime). In sum, it uses a dict of key-value pairs 
            # that control all PyBuda specific overrides set in test. Have in  mind that this doesn't apply for everything set 
            # outside of the test itself (e.g. env vars set before calling the specific pytest).
            #
            # Input to this function is represented as two dicts:
            # - first one is a dict of keys/value pairs that controls general compiler config settings
            # - second one is a dict of keys/value pairs that controls configurations set through environment variables
            # 
            # Also, few general notes of how to use this these dicts to control the general and env based compiler configurations:
            # - overriding value with "" will use the value set in test itself
            # - overriding with some specific value will use that config and override it (ignoring the test config)
            # - not including the key and value here, will use default compiler config value (discarding the test config if set there)
            #
            # Description of override levels:
            # - Level 0 - set by compiler;      we want to keep them            (defined during compiler runtime; not in test itself)
            # - Level 1 - set by user in test;  we want to keep them            (defined in test, but easy to use and understandable for end user)
            # - Level 2 - set by dev in test;   we want to remove them          (e.g. enable/disable by default, redefine as more user friendly, etc.)
            # - Level 3 - set by dev in test;   we want to remove them entirely (purely for testing purposes)
            #
            if "PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP" in os.environ:
                _set_pybuda_override_veto({
                    "backend_output_dir": "",
                }, {})
            else:
                _set_pybuda_override_veto({
                    "backend_output_dir": "",
                    "backend_runtime_params_path": "",
                    "harvesting_mask": "",
                    "cpu_fallback_ops": "",

                    # Level 1 overrides
                    "balancer_policy": "",
                    "enable_t_streaming": "",
                    "default_df_override": "",
                },
                {
                    # Level 2 overrides
                    "PYBUDA_DISABLE_STREAM_OUTPUT": "",
                    "PYBUDA_PAD_OUTPUT_BUFFER": "",
                    "PYBUDA_OVERRIDE_DEVICE_YAML": "" # Mostly used for 1x1 model overrides
                })

    elif report.when == "teardown":
        environ_before_test_keys = set(environ_before_test.keys())
        environ_after_test_keys = set(os.environ.keys())

        # remove
        added_flags = environ_before_test_keys ^ environ_after_test_keys
        for f in added_flags:
            os.environ.pop(f, None)

        # reset
        for key, default_value in environ_before_test.items():
            if os.environ.get(key, "") != default_value:
                os.environ[key] = default_value

    if report.failed:
        last_stage = COMPILE_INFO.LAST_SUCCESSFUL_STAGE
        if not last_stage: 
            last_stage = "failed before compile"
        print(f"\nLAST SUCCESSFUL COMPILE STAGE: {last_stage}\n")


def pytest_addoption(parser):
    parser.addoption(
        "--silicon-only", action="store_true", default=False, help="run silicon tests only, skip golden/model"
    )
    parser.addoption("--no-silicon", action="store_true", default=False, help="skip silicon tests")
    parser.addoption(
        "--no-skips", action="store_true", default=False, help="ignore pytest.skip() calls, and continue on with test"
    )
    parser.addoption(
        "--device-config", default=None, type=str, help="Runtime yaml is automatically configured based on the value"
    )
    parser.addoption(
        "--compile-only", action="store_true", default=False, help="only compiles the model and generate TTI"
    )
    parser.addoption(
        "--run-only", action="store_true", default=False, help="load the generated TTI and only runs the model"
    )
    parser.addoption(
        "--devtype", default=None, type=str, choices=("golden", "silicon"), help="Valid only if --compile-only is specified. Set the backend device type between Golden or Silicon"
    )
    parser.addoption(
        "--tti-path", default=None, type=str, help="Valid only if either --compile-only or --run-only is specified. Save/load TTI from the path"
    )
