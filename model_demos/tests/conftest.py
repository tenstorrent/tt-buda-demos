import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pybuda
import pytest
from pybuda._C.backend_api import BackendDevice, BackendType

# Environment variable storage for cleanup
environ_before_test = None

detected_devices = pybuda.detect_available_devices()
device_type = BackendType.Golden if len(detected_devices) == 0 else BackendType.Silicon

RESET_TIMEOUT = 60


def reset_board():
    """Executes the reset command on the board."""
    try:
        print("Resetting board...")
        subprocess.run(
            args=["tt-smi", "-lr", "0"],
            capture_output=True,
            timeout=RESET_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print("Reset command timed out.")
    except subprocess.CalledProcessError as e:
        print(f"Reset command failed with a non-zero exit status: {e}")
    except Exception as e:
        print(f"An error occurred during board reset: {e}")


@pytest.fixture(autouse=True)
def clear_pybuda():
    """
    Fixture to clean up and reset devices after each test.
    Automatically used before each test due to 'autouse=True'.
    """
    yield  # Yield control back to pytest for test execution

    pybuda.shutdown()
    pybuda.pybuda_reset()

    archive_files()
    subprocess.run(["make", "clean_tt"])

    # Reset the board if it's a Silicon device and in a bad state
    if device_type == BackendType.Silicon:
        print("Silicon device detected. Resetting board...")
        reset_board()


def pytest_runtest_logreport(report):
    """
    Manages environment variables by clearing any that are set during the test when it ends.
    """
    global environ_before_test
    if report.when == "setup":
        environ_before_test = os.environ.copy()
    elif report.when == "teardown":
        new_env_keys = set(os.environ.keys()) - set(environ_before_test.keys())
        for key in new_env_keys:
            os.environ.pop(key, None)


def archive_files(src_directory=Path("."), dest_directory=Path("./archive")):
    """
    Archives files ending with '_netlist.yaml' from the source directory to the archive directory,
    appending a timestamp to the filename to avoid overwriting.
    """
    src_directory = Path(src_directory).absolute()
    dest_directory = Path(dest_directory).absolute()

    if not src_directory.exists():
        print(f"Source directory {src_directory} does not exist!")
        return

    if not dest_directory.exists():
        dest_directory.mkdir(parents=True)

    for file_path in src_directory.glob("*_netlist.yaml"):
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dest_path = dest_directory / f"{file_path.stem}_{timestamp}{file_path.suffix}"

        try:
            shutil.copy(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
        except Exception as e:
            print(f"Failed to copy {file_path} to {dest_path}. Reason: {e}")
