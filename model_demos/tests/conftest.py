import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pybuda
import pytest

environ_before_test = None

@pytest.fixture(autouse=True)
def clear_pybuda():
    """
    Cleans up the pybuda environment after each test and archives test-related files.
    """
    yield
    pybuda.shutdown()
    pybuda.pybuda_reset()
    archive_files()
    subprocess.run(["make", "clean_tt"])

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
    if not src_directory.exists():
        print(f"Source directory {src_directory} does not exist!")
        return

    if not dest_directory.exists():
        dest_directory.mkdir(parents=True)

    for file_path in src_directory.glob("*_netlist.yaml"):
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dest_path = dest_directory / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        try:
            shutil.copy(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
        except Exception as e:
            print(f"Failed to copy {file_path} to {dest_path}. Reason: {e}")

