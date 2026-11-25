"""The mapped main EXE is the process image: GetModuleHandle(NULL) and FindResource(NULL, ...).

cpu_exe_process_image is a CRT-free EXE that asks the process for its own image and reports one log
line (tests/cpu/exe_process_image.cpp). Compared runs:

- windows: launched by Windows, where it is the process image. Every check is y and the exit is 0.
- runner: tools/pe_run_package.py with --system kernel32.dll, which starts the entry with
  pe_map_run_main. The guest log and the exit code must equal the Windows launch.
- main: an in-process pe_map host maps the EXE, starts it with pe_map_run_main and releases it.
  The guest log equals the Windows log, and after the release GetModuleHandleW(NULL) in the host
  returns the host's own base again.
- plain (negative control): the same host starts the EXE with pe_map_run, which does not report
  it as the process image. The EXE then finds the host instead of itself: self, hdr and res fail.
- twice: while one mapped EXE is the process image, pe_map_run_main refuses a second mapping and
  runs nothing; after the first is released, the second starts and is reported, and its release
  restores the host's value.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "pe_run_package.py"
EXE = "cpu_exe_process_image.exe"
EXPECTED = "self=y hdr=y res=y miss=y k32=y X=0"
TIMEOUT = 60

# In-process pe_map host. argv: mapper, mode, exe [, second exe]. Prints LOG lines from the guest and
# HOST: lines from here; exits 0 unless the host itself fails.
HOST = r"""
import ctypes as C, sys
from ctypes import wintypes as W
mapper_path, mode, exe = sys.argv[1], sys.argv[2], sys.argv[3]
k = C.WinDLL("kernel32", use_last_error=True)
k.GetModuleHandleW.restype = C.c_void_p
k.GetModuleHandleW.argtypes = [W.LPCWSTR]
k.LoadLibraryW.restype = C.c_void_p
k.LoadLibraryW.argtypes = [W.LPCWSTR]
k.GetProcAddress.restype = C.c_void_p
k.GetProcAddress.argtypes = [C.c_void_p, C.c_char_p]
lib = C.WinDLL(mapper_path)
Resolver = C.WINFUNCTYPE(C.c_void_p, C.c_char_p, C.c_char_p, C.c_uint32)
lib.pe_map_image.restype = C.c_void_p
lib.pe_map_image.argtypes = [W.LPCWSTR, Resolver, C.POINTER(C.c_uint32), C.c_char_p, C.c_size_t]
for name in ("pe_map_run", "pe_map_run_main"):
    getattr(lib, name).restype = C.c_int
    getattr(lib, name).argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
lib.pe_map_release.restype = C.c_int
lib.pe_map_release.argtypes = [C.c_void_p]

@Resolver
def resolve(dll, name, ordinal):
    if dll.decode().casefold() != "kernel32.dll" or not name:
        return None
    return k.GetProcAddress(k.LoadLibraryW("kernel32.dll"), name)

def map_image(path):
    status, detail = C.c_uint32(), C.create_string_buffer(512)
    base = lib.pe_map_image(path, resolve, C.byref(status), detail, len(detail))
    if not base:
        sys.exit(f"pe_map_image refused {path}: {status.value} {detail.value!r}")
    return base

def start(run, base):
    code = C.c_uint32()
    started = run(base, C.byref(code))
    sys.stdout.flush()
    return started, code.value

host = k.GetModuleHandleW(None)
if mode in ("main", "plain"):
    base = map_image(exe)
    started, code = start(lib.pe_map_run_main if mode == "main" else lib.pe_map_run, base)
    print(f"HOST: started={started} exit={code}", flush=True)
    print(f"HOST: released={lib.pe_map_release(base)} restored={'y' if k.GetModuleHandleW(None) == host else 'n'}", flush=True)
elif mode == "twice":
    first, second = map_image(exe), map_image(sys.argv[4])
    started, code = start(lib.pe_map_run_main, first)
    print(f"HOST: first started={started} exit={code}", flush=True)
    started, code = start(lib.pe_map_run_main, second)
    print(f"HOST: second while first is the process image started={started}", flush=True)
    print(f"HOST: first released={lib.pe_map_release(first)} restored={'y' if k.GetModuleHandleW(None) == host else 'n'}", flush=True)
    started, code = start(lib.pe_map_run_main, second)
    print(f"HOST: second started={started} exit={code}", flush=True)
    print(f"HOST: second released={lib.pe_map_release(second)} restored={'y' if k.GetModuleHandleW(None) == host else 'n'}", flush=True)
"""


def logs(stdout):
    return [line[4:].strip() for line in stdout.splitlines() if line.startswith("LOG:")]


def host_lines(stdout):
    return [line[6:].strip() for line in stdout.splitlines() if line.startswith("HOST:")]


def expect(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def run_host(mapper, mode, *exes):
    done = subprocess.run([sys.executable, "-B", "-c", HOST, str(mapper), mode, *map(str, exes)],
                          capture_output=True, text=True, timeout=TIMEOUT)
    if done.returncode or done.stderr:
        raise AssertionError(f"{mode} host failed: {done!r}")
    return logs(done.stdout), host_lines(done.stdout)


def check(mapper, exe):
    with TemporaryDirectory() as temp:
        package = Path(temp) / "package"
        other = Path(temp) / "other"
        package.mkdir()
        other.mkdir()
        shutil.copyfile(exe, package / EXE)
        shutil.copyfile(exe, other / EXE)

        native = subprocess.run([str(package / EXE)], capture_output=True, text=True, timeout=TIMEOUT)
        expect("windows", (logs(native.stdout), native.returncode, native.stderr), ([EXPECTED], 0, ""))
        print(f"windows: LOG:{EXPECTED} exit 0")

        done = subprocess.run([sys.executable, "-B", str(TOOL), str(mapper), str(package), EXE,
                               "--system", "kernel32.dll"], capture_output=True, text=True, timeout=TIMEOUT)
        expect("runner stderr", (done.returncode, done.stderr), (0, ""))
        expect("runner log", logs(done.stdout), [EXPECTED])
        expect("runner result", json.loads(done.stdout[done.stdout.index("{"):]),
               {"image_order": [EXE], "entry_executed": True, "exit_code": 0})
        print(f"runner: LOG:{EXPECTED} exit 0, equal to the Windows launch")

        log, host = run_host(mapper, "main", package / EXE)
        expect("main", (log, host), ([EXPECTED], ["started=1 exit=0", "released=1 restored=y"]))
        print("main: guest log equals Windows; after release the host's GetModuleHandleW(NULL) is its own again")

        log, host = run_host(mapper, "plain", package / EXE)
        if len(log) != 1 or not log[0].startswith("self=n hdr=n res=n "):
            raise AssertionError(f"plain control: {log!r}")
        expect("plain host", host, ["started=1 exit=" + log[0].rsplit("X=", 1)[1], "released=1 restored=y"])
        print(f"plain (control): pe_map_run leaves the host as the process image: LOG:{log[0]}")

        log, host = run_host(mapper, "twice", package / EXE, other / EXE)
        expect("twice", (log, host), ([EXPECTED, EXPECTED], [
            "first started=1 exit=0",
            "second while first is the process image started=0",
            "first released=1 restored=y",
            "second started=1 exit=0",
            "second released=1 restored=y",
        ]))
        print("twice: a second main image is refused while the first is the process image, then starts after its release")
    print("process image checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: process_image_check.py PE_MAP_DLL CPU_EXE_PROCESS_IMAGE")
    check(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]).resolve(strict=True))
