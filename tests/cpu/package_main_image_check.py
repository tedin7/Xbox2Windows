"""pe_run_package.py starts the package entry as the main image.

cpu_exe_tls_slot0 is the CRT-free /GA EXE from the main-image TLS slot 0 tests: its thread-locals
are read through slot 0 of ThreadLocalStoragePointer, and it starts two workers through its
kernel32 CreateThread import (tests/cpu/exe_tls_slot0.cpp). The check compares three runs:

- Windows launch: the expected guest log, exit 0.
- pe_run_package.py with --system kernel32.dll: the runner starts the entry with pe_map_run_main
  and binds its CreateThread import to pe_map_main_create_thread. The guest log, the exit code
  and the JSON result must match the Windows launch.
- The same package without --system kernel32.dll: refused before mapping, with no guest log.

Under the previous pe_map_run start, every slot-0 check and the process-image check in the guest
log failed (native_exe_tls_slot0_check.PLAIN, exit 9).
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "pe_run_package.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_exe_tls_slot0_check import EXPECTED, PLAIN  # noqa: E402

EXE = "cpu_exe_tls_slot0.exe"
TIMEOUT = 60


def guest_log(stdout):
    return next((line[4:].strip() for line in stdout.splitlines() if line.startswith("LOG:")), None)


def check_fixture(exe):
    with pefile.PE(str(exe)) as pe:
        if not getattr(pe, "DIRECTORY_ENTRY_TLS", None):
            raise AssertionError("fixture has no TLS directory")
        imports = {(entry.dll.decode().casefold(), imp.name.decode())
                   for entry in pe.DIRECTORY_ENTRY_IMPORT for imp in entry.imports if imp.name}
    if ("kernel32.dll", "CreateThread") not in imports:
        raise AssertionError(f"fixture does not import kernel32!CreateThread: {sorted(imports)}")


def check(mapper, exe):
    check_fixture(exe)
    if EXPECTED == PLAIN:
        raise AssertionError("expected and plain logs must differ")
    with TemporaryDirectory() as temp:
        package = Path(temp)
        shutil.copyfile(exe, package / EXE)

        native = subprocess.run([str(package / EXE)], capture_output=True, text=True, timeout=TIMEOUT)
        native_exit = native.returncode & 0xFFFFFFFF
        if (guest_log(native.stdout), native_exit, native.stderr) != (EXPECTED, 0, ""):
            raise AssertionError(f"Windows launch: {native!r}")
        print(f"windows: LOG:{EXPECTED} exit {native_exit:#x}")

        command = [sys.executable, "-B", str(TOOL), str(mapper), str(package), EXE]
        refused = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
        if (refused.returncode, refused.stdout) != (1, "") or "unresolved external symbol" not in refused.stderr:
            raise AssertionError(f"runner accepted a package without kernel32: {refused!r}")
        print("runner without --system kernel32.dll: refused before mapping, no guest log")

        done = subprocess.run(command + ["--system", "kernel32.dll"], capture_output=True, text=True,
                              timeout=TIMEOUT)
        if done.returncode or done.stderr:
            raise AssertionError(f"runner failed: {done!r}")
        log = guest_log(done.stdout)
        if log == PLAIN:
            raise AssertionError("runner started the entry without slot 0 (pe_map_run result)")
        if log != EXPECTED:
            raise AssertionError(f"runner guest log differs from Windows: {log!r}")
        result = json.loads(done.stdout[done.stdout.index("{"):])
        expected_result = {"image_order": [EXE], "entry_executed": True, "exit_code": native_exit}
        if result != expected_result:
            raise AssertionError(f"runner result {result!r}, expected {expected_result!r}")
        print(f"runner: LOG:{log} exit {result['exit_code']:#x}, equal to the Windows launch")

        # Register the real Windows library as an explicit provider. Unlike a host default
        # route, its CreateThread must survive import resolution untouched.
        kernel = Path(os.environ["SystemRoot"]) / "System32" / "kernel32.dll"
        explicit = subprocess.run(command + ["--system", "kernel32.dll",
                                             "--shim", "kernel32.dll", str(kernel)],
                                  capture_output=True, text=True, timeout=TIMEOUT, check=True)
        explicit_log = guest_log(explicit.stdout)
        explicit_result = json.loads(explicit.stdout[explicit.stdout.index("{"):])
        if (explicit.stderr or explicit_log != "m1=yyy m:yyyy a:n--- b:n--- X=2"
                or explicit_result != {**expected_result, "exit_code": 2}):
            raise AssertionError(f"runner replaced the explicit CreateThread provider: {explicit!r}")
        print("explicit kernel32 provider: entry owns slot 0; unwrapped workers remain unenrolled")
    print("package main-image checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: package_main_image_check.py PE_MAP_DLL CPU_EXE_TLS_SLOT0")
    check(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve(strict=True))
