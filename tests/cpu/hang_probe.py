"""Probe a CRT-free EXE for the launch hang reported in #281.

Launch the target repeatedly with a one-byte selector on stdin. For every launch
that does not exit, record what the process and its pipes are doing at that
moment: whether the process handle is already signaled, the bytes received so far
on stdout and stderr separately, and whether each stream reached EOF within two
seconds. A timeout with an unsignaled handle means the process is still running;
a signaled handle means it exited while something else still held a pipe.

`--control` runs the deterministic fixtures built by CMake instead: a native
wrapper that starts a worker which never returns, so the exit path is the only
variable. Each fixture has a documented outcome, so an intermittent result fails.

Standard library only. On Windows the handle check is WaitForSingleObject through
ctypes; nothing needs to be installed.
"""

import argparse
import ctypes
import pathlib
import subprocess
import sys
import threading
import time

WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102

# outcome each control fixture must produce on every launch
CONTROL_CASES = (
    ("original", "timeout"),
    ("returning", "timeout"),
    ("no_worker", "exit"),
)


class ThreadEntry(ctypes.Structure):
    _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                ("th32ThreadID", ctypes.c_ulong), ("th32OwnerProcessID", ctypes.c_ulong),
                ("tpBasePri", ctypes.c_long), ("tpDeltaPri", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong)]


def live_threads(process):
    """Thread ids the process still owns, from a toolhelp snapshot."""
    invalid = ctypes.c_void_p(-1).value
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    for name in ("Thread32First", "Thread32Next"):
        getattr(kernel32, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ThreadEntry)]
        getattr(kernel32, name).restype = ctypes.c_int
    snapshot = kernel32.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
    if snapshot == invalid:
        return None
    entry = ThreadEntry()
    entry.dwSize = ctypes.sizeof(ThreadEntry)
    found = []
    reached = kernel32.Thread32First(snapshot, ctypes.byref(entry))
    while reached:
        if entry.th32OwnerProcessID == process.pid:
            found.append(entry.th32ThreadID)
        reached = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    kernel32.CloseHandle(snapshot)
    return found


def wait_state(process):
    """Whether the process object is already signaled, without waiting for it."""
    handle = ctypes.c_void_p(int(process._handle))
    state = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
    return {WAIT_OBJECT_0: "signaled", WAIT_TIMEOUT: "not signaled"}.get(state, f"wait state {state}")


def launch(exe, payload, timeout, eof_timeout):
    """Run one launch and describe it. Nothing is killed before the state is read."""
    process = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    received = {"stdout": bytearray(), "stderr": bytearray()}
    reached_eof = {name: threading.Event() for name in received}

    def drain(name):
        stream = getattr(process, name)
        while True:
            # read1 returns what is already available; read() would wait for a full buffer,
            # so a short stream would look empty until EOF.
            chunk = stream.read1(4096)
            if not chunk:
                break
            received[name].extend(chunk)
        reached_eof[name].set()

    readers = [threading.Thread(target=drain, args=(name,), daemon=True) for name in received]
    for reader in readers:
        reader.start()

    started = time.perf_counter()
    try:
        process.stdin.write(payload)
        process.stdin.close()
    except OSError:
        # A fixture that exits without reading its selector breaks the pipe before or while this
        # write runs. The outcome check below still decides: it must exit or time out as documented.
        pass
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Everything below is read before the child is terminated, on purpose.
        elapsed = time.perf_counter() - started
        result = {"outcome": "timeout", "elapsed": elapsed, "handle": wait_state(process),
                  "threads": live_threads(process),
                  "bytes": {name: len(data) for name, data in received.items()},
                  "eof": {name: reached_eof[name].wait(eof_timeout) for name in received},
                  "head": {name: bytes(data[:16]).hex(" ") for name, data in received.items()}}
        process.kill()
        process.wait(timeout=eof_timeout)
        return result

    elapsed = time.perf_counter() - started
    # A terminated process says nothing about the pipes: let the readers see EOF first.
    for name in received:
        reached_eof[name].wait(eof_timeout)
    return {"outcome": "exit", "code": code, "elapsed": elapsed, "handle": "signaled",
            "bytes": {name: len(data) for name, data in received.items()},
            "eof": {name: reached_eof[name].wait(eof_timeout) for name in received},
            "head": {name: bytes(data[:16]).hex(" ") for name, data in received.items()}}


def describe(index, selector, result):
    lines = [f"launch {index} selector {selector}: {result['outcome']} after "
             f"{result['elapsed'] * 1000:.0f} ms"]
    if result["outcome"] == "exit":
        lines.append(f"  exit code {result['code']} (0x{result['code'] & 0xffffffff:x})")
    else:
        lines.append(f"  process handle: {result['handle']} (WaitForSingleObject, 0 ms)")
        threads = result.get("threads")
        if threads is not None:
            ids = ", ".join(str(thread) for thread in threads)
            lines.append(f"  threads: {len(threads)} live ({ids})")
    for name in ("stdout", "stderr"):
        eof = "reached" if result["eof"][name] else "not reached"
        lines.append(f"  {name}: {result['bytes'][name]} bytes, EOF {eof} within the EOF timeout")
        if result["head"][name]:
            lines.append(f"    first bytes: {result['head'][name]}")
    return "\n".join(lines)


def probe(targets, launches, timeout, eof_timeout):
    """Bounded comparison: alternate selectors, print one normal launch and each timeout."""
    failures = 0
    for target in targets:
        counts = {"exit": 0, "timeout": 0}
        codes = {}
        slowest = 0.0
        reported = False
        for index in range(launches):
            selector = index % 2
            result = launch(target, bytes([selector]), timeout, eof_timeout)
            counts[result["outcome"]] += 1
            slowest = max(slowest, result["elapsed"])
            if result["outcome"] == "exit":
                codes[result["code"]] = codes.get(result["code"], 0) + 1
            if result["outcome"] == "timeout" or not reported:
                print(describe(index, selector, result))
                reported = True
        print(f"{target.name}: {counts['exit']} terminated, {counts['timeout']} timed out "
              f"in {launches} launches; slowest {slowest * 1000:.0f} ms; exit codes "
              f"{ {code: count for code, count in sorted(codes.items())} }")
        failures += counts["timeout"]
    return 1 if failures else 0


def control(targets, launches, timeout, eof_timeout):
    """Deterministic fixtures: every launch must match the documented outcome."""
    failures = 0
    for (suffix, expected), target in zip(CONTROL_CASES, targets):
        if not target.stem.endswith(suffix):
            raise SystemExit(f"expected the {suffix} fixture, got {target.name}")
        outcomes = []
        shown = False
        for index in range(launches):
            result = launch(target, bytes([index % 2]), timeout, eof_timeout)
            outcomes.append(result["outcome"])
            marker = b"worker-alive control\r\n"
            if result["bytes"]["stdout"] != len(marker) or result["head"]["stdout"] != marker[:16].hex(" "):
                print("  missing ready marker before exit or timeout")
                failures += 1
            if result["outcome"] == "timeout" and not shown:
                print(describe(index, index % 2, result))
                shown = True
            if result["outcome"] == "timeout":
                signature = (result["handle"], result["eof"]["stdout"], result["eof"]["stderr"])
                if signature != ("not signaled", False, False):
                    print(f"  unexpected signature: handle {signature[0]}, stdout EOF {signature[1]}, "
                          f"stderr EOF {signature[2]}")
                    failures += 1
        wrong = [outcome for outcome in outcomes if outcome != expected]
        status = "deterministic" if not wrong else f"{len(wrong)} of {launches} differ"
        print(f"{target.name}: expected {expected} on every launch; {status} ({outcomes})")
        failures += len(wrong)
    print(f"control: {'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=pathlib.Path,
                        help="EXE to launch, or the three control fixtures with --control")
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--eof-timeout", type=float, default=2.0)
    parser.add_argument("--control", action="store_true",
                        help="assert the deterministic fixtures instead of probing a target")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("Skipped: the process handle is read with WaitForSingleObject, which needs Windows.")
        return 77
    if args.control:
        return control(args.targets, args.launches, args.timeout, args.eof_timeout)
    return probe(args.targets, args.launches, args.timeout, args.eof_timeout)


if __name__ == "__main__":
    sys.exit(main())
