"""Check the startup trap stubs.

usage: startup_stub_check.py <consumer.exe> <src/startup_stubs> <stub dll>... <xmem.dll> <d3d12_xs.dll>

1. Each stub DLL loads on its own and every name in its .def resolves.
2. The consumer traps with the documented exit code and log line, for a stub DLL and for the
   trap exports added to xmem.dll and d3d12_xs.dll. The line also lands in X2W_TRAP_LOG.
3. Controls: the 'none' scenario exits 0 and writes no trap line; the physical-page round trip
   in xmem.dll still works next to the traps.
"""
import atexit
import ctypes
import ctypes.wintypes as wt
import os
import re
import shutil
import subprocess
import sys
import tempfile

consumer, stub_src, *dlls = sys.argv[1:]
failures = []

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.LoadLibraryExW.restype = wt.HMODULE
k32.LoadLibraryExW.argtypes = [wt.LPCWSTR, wt.HANDLE, wt.DWORD]
k32.GetProcAddress.restype = ctypes.c_void_p
k32.GetProcAddress.argtypes = [wt.HMODULE, ctypes.c_char_p]
k32.FreeLibrary.argtypes = [wt.HMODULE]
LOAD_WITH_ALTERED_SEARCH_PATH = 0x8

defs = {}
for name in os.listdir(stub_src):
    if name.endswith(".def"):
        text = open(os.path.join(stub_src, name), encoding="ascii").read()
        library = re.search(r"^LIBRARY\s+(\S+)", text, re.M).group(1)
        defs[library.lower() + ".dll"] = re.findall(r"^\s{4}(\w+)(?:\s+@\d+)?\s*$", text, re.M)
extra = {"xmem.dll": ["XMemIsToolsMemAvailable", "XMemMapPhysicalPages", "XMemMapPhysicalPagesScatter",
                      "XMemTransferMemory", "XMemVirtualAlloc", "XMemAllocatePhysicalPages",
                      "XMemFreePhysicalPages"],
         "d3d12_xs.dll": ["D3D12ConfigureVirtualMemoryX"]}

seen = set()
for path in dlls:
    base = os.path.basename(path).lower()
    names = defs.get(base) or extra.get(base)
    if not names:
        failures.append(f"no export list for {base}")
        continue
    seen.add(base)
    handle = k32.LoadLibraryExW(os.path.abspath(path), None, LOAD_WITH_ALTERED_SEARCH_PATH)
    if not handle:
        failures.append(f"{base}: LoadLibrary failed ({ctypes.get_last_error()})")
        continue
    missing = [n for n in names if not k32.GetProcAddress(handle, n.encode())]
    if missing:
        failures.append(f"{base}: unresolved {missing}")
    k32.FreeLibrary(handle)
    print(f"load {base}: {len(names) - len(missing)}/{len(names)} exports")
for want in list(defs) + list(extra):
    if want not in seen:
        failures.append(f"{want} was not checked")

env = dict(os.environ)
env["PATH"] = os.pathsep.join(sorted({os.path.dirname(os.path.abspath(p)) for p in dlls})) + os.pathsep + env["PATH"]
# The trap log lives in a fresh folder that is removed when the check exits, pass or fail.
log_dir = tempfile.mkdtemp(prefix="x2w_trap_")
atexit.register(shutil.rmtree, log_dir, ignore_errors=True)
log = os.path.join(log_dir, "trap.log")
env["X2W_TRAP_LOG"] = log

cases = [
    ("chad", "chad.dll!XDspActivate", 0xE2570200, "rcx=0x11 rdx=0x22 r8=0x33 r9=0x44"),
    ("xg_xs", "xg_xs.dll!XGSuggestSwizzleMode", 0xE2570501, "rcx=0x55 rdx=0x0"),
]
consumer_name = os.path.basename(consumer).lower()
for scenario, symbol, code, regs in cases:
    run = subprocess.run([consumer, scenario], capture_output=True, text=True, env=env, timeout=20)
    line = f"x2w-trap {symbol} {regs}"
    ok = (run.returncode & 0xFFFFFFFF) == code and line in run.stderr \
        and f"ret={consumer_name}+0x" in run.stderr.lower()
    print(f"{scenario}: exit=0x{run.returncode & 0xFFFFFFFF:08x} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(f"{scenario}: exit 0x{run.returncode & 0xFFFFFFFF:08x}, stderr {run.stderr.strip()!r}")

logged = open(log, encoding="ascii").read() if os.path.exists(log) else ""
if logged.count("x2w-trap ") != len(cases):
    failures.append(f"log has {logged.count('x2w-trap ')} trap lines, expected {len(cases)}")

for scenario, stdout in (("none", ""), ("physical", "physical ok")):
    before = os.path.getsize(log) if os.path.exists(log) else 0
    run = subprocess.run([consumer, scenario], capture_output=True, text=True, env=env, timeout=20)
    after = os.path.getsize(log) if os.path.exists(log) else 0
    ok = run.returncode == 0 and stdout in run.stdout and "x2w-trap" not in run.stderr and before == after
    print(f"{scenario} control: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(f"{scenario} control: exit {run.returncode}, stderr {run.stderr.strip()!r}")

run = subprocess.run([consumer, "dstorage"], capture_output=True, text=True, env=env, timeout=20)
traps_after = open(log, encoding="ascii").read().count("x2w-trap ") if os.path.exists(log) else 0
summary = re.search(r"x2w-dstorage BCPack summary: requests formats=([^;]*); zero-filled formats=([^\r\n]*)",
                    run.stderr)
requests_by_format = {int(f): int(n) for f, n in re.findall(r"(\d+):(\d+)", summary.group(1))} if summary else {}
blank_by_format = {int(f): int(n) for f, n in re.findall(r"(\d+):(\d+)", summary.group(2))} if summary else {}
bcpack_ok = (requests_by_format.get(6) == 1 and blank_by_format.get(6) == 1
             and requests_by_format.get(1) == 2 and blank_by_format.get(1, 0) == 0
             and requests_by_format.get(3) == 3 and blank_by_format.get(3, 0) == 0
             and requests_by_format.get(4) == 4 and blank_by_format.get(4, 0) == 0
             and requests_by_format.get(5) == 3 and blank_by_format.get(5, 0) == 0
             and requests_by_format.get(7) == 4 and blank_by_format.get(7, 0) == 0)
if summary:
    print(summary.group(0))
else:
    print("BCPack summary: missing")
ok = (run.returncode == 0 and "dstorage ok" in run.stdout and "x2w-trap" not in run.stderr
      and traps_after == len(cases) and bcpack_ok)
print(f"dstorage control: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"dstorage control: exit {run.returncode}, stderr {run.stderr.strip()!r}")

# Page arrays changed between EnqueueRequest and Submit: a replaced entry, and an unreadable array.
# #1843: a new status array's slots start complete with S_OK until EnqueueStatus marks one pending.
run = subprocess.run([consumer, "dstorage-status-initial"], capture_output=True, text=True, env=env, timeout=20)
ok = run.returncode == 0 and "dstorage-status-initial ok" in run.stdout and "x2w-trap" not in run.stderr
print(f"dstorage status-array initial state: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"dstorage status-array initial state: exit {run.returncode}, stdout {run.stdout.strip()!r}")

run = subprocess.run([consumer, "dstorage-pages"], capture_output=True, text=True, env=env, timeout=20)
print(run.stdout.strip())
ok = run.returncode == 0 and "dstorage-pages ok" in run.stdout and "x2w-trap" not in run.stderr
print(f"dstorage page-array control: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"dstorage page-array control: exit {run.returncode}, stdout {run.stdout.strip()!r}")

run = subprocess.run([consumer, "dstorage-error-record"], capture_output=True, text=True, env=env, timeout=20)
ok = run.returncode == 0 and "dstorage-error-record ok" in run.stdout and "x2w-trap" not in run.stderr
print(f"dstorage error record: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"dstorage error record: exit {run.returncode}, stderr {run.stderr.strip()!r}")

store_env = dict(env, X2W_RUNTIME_PROXY="1", X2W_FAKE_USER="1")
run = subprocess.run([consumer, "store"], capture_output=True, text=True, env=store_env, timeout=20)
traps_after = open(log, encoding="ascii").read().count("x2w-trap ") if os.path.exists(log) else 0
ok = (run.returncode == 0 and "store context 5832575300000001" in run.stdout
      and "store refused 80004005" in run.stdout
      and "store user refused 80004005" in run.stdout and "x2w-trap" not in run.stderr
      and traps_after == len(cases))
print(f"store context: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"store context: exit {run.returncode}, stdout {run.stdout.strip()!r}, stderr {run.stderr.strip()!r}")

# File::GetFileInformation after Close returns ERROR_INVALID_HANDLE, with the trap log unset and set.
run = subprocess.run([consumer, "dstorage-file-information"], capture_output=True, text=True, env=env, timeout=20)
print(run.stdout.strip())
ok = run.returncode == 0 and "dstorage-file-information ok" in run.stdout and "x2w-trap" not in run.stderr
print(f"dstorage file-information control: exit={run.returncode} {'ok' if ok else 'MISMATCH'}")
if not ok:
    failures.append(f"dstorage file-information control: exit {run.returncode}, stdout {run.stdout.strip()!r}")

if failures:
    print("FAIL")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("PASS")
