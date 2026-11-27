"""Check diagnostics of tools/startup/debug_launch.py on a source-built probe.

X2W_STOP_AT stops at a chosen main-image breakpoint while --continue-asserts steps over the
others, X2W_DUMP_AT writes thread stacks at a time during the run, and X2W_VALLOC_MIN logs large
virtual-memory allocations with their callers. X2W_LOG_AT captures registers and qwords at a
function entry, then removes the breakpoint at the hit limit. A `stop` file created in the log directory
ends the run as the timeout does; one left from an earlier run is removed at start. The launcher keeps the debug heap off unless
the caller sets _NO_DEBUG_HEAP=0. X2W_NOP_SOFT_ASSERTS=1 replaces a main-image INT3 with a NOP after
the first continue, and the probe repeats one site when STARTUP_DEBUG_PROBE_ASSERTS is set.
tests/startup_debug_probe.cpp is the main input: it hits two breakpoints, optionally repeats a
third, reserves 256 MiB, prints its debug-heap flags, waits 3 s and exits 0.

The gaming-runtime and movie-player call hooks are armed only with --trace-calls 1. For that case
the probe and tests/startup_debug_calls.cpp are copied into one folder, the DLL under the
movie-player module name the launcher hooks; the probe calls its export 7 times. The launcher's
call count must be absent by default and exactly 7 with --trace-calls 1. The fixture's module and
export names must be among the ones the launcher hooks, so the check cannot pass with nothing hooked,
and --trace-calls 1 without file tracing must count nothing and say why.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DEBUG = REPO / "tools/startup/debug_launch.py"


def missing_heap_flags(logdir):
    """Explain a missing flag line rather than dumping stdout alone.

    The probe prints one line per state. A probe built before the flag line existed prints only
    "debug-probe reserved ok", which looks like a heap-flag regression; name that possibility here.
    """
    printed = text(logdir / "stdout.txt")
    if "debug-probe heap flags" in printed:
        return printed
    return ("no heap-flags line in the probe output; rebuild startup_debug_probe if it predates this "
            "check. Output was:\n" + printed)

def run(probe, work, name, variables, options=("--continue-asserts", "10")):
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") and k.upper() != "_NO_DEBUG_HEAP"}
    env.update(variables)
    command = [sys.executable, "-B", str(DEBUG), str(probe), str(work), str(logdir), "--timeout", "30", "--console", "0", *options]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    return result.stdout + result.stderr, logdir


def run_with_stop(probe, work, name):
    """Create <logdir>/stop once the launcher records the running probe."""
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") and k.upper() != "_NO_DEBUG_HEAP"}
    command = [sys.executable, "-B", str(DEBUG), str(probe), str(work), str(logdir), "--timeout", "0", "--console", "0",
               "--continue-asserts", "10"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        deadline = time.monotonic() + 10
        while f"process {probe.name} at " not in text(logdir / "events.txt"):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError("stop-file probe did not start")
            time.sleep(0.025)
        (logdir / "stop").write_text("", encoding="utf-8")
        out, _ = process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    return out, logdir


def text(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def breakpoint_sites(logdir, image):
    """The main-image RVAs of the breakpoints the debugger logged, in order."""
    return [int(m, 16) for m in re.findall(r"breakpoint at %s\+0x([0-9a-f]+) first-chance" % re.escape(image),
                                          text(logdir / "events.txt"))]


HOOKED_MODULE = "bink2scarlett.dll"  # the movie-player module name debug_launch.py hooks
HOOKED_EXPORT = "BinkNextFrame"
CALLS = 7  # kCalls in tests/startup_debug_probe.cpp
LOG_CALLS = 19  # kLogCalls in tests/startup_debug_probe.cpp
LOG_TOKEN = 0x1122334455660000
LOG_REGISTERS = set("rax rcx rdx rbx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15".split())


def require_hooked(module, export, hooked_module, hooked_exports):
    """The fixture's DLL must stand in for a module the launcher hooks, and its export must be hooked."""
    if module != hooked_module or export not in hooked_exports:
        raise AssertionError(f"{module}!{export} is not hooked by debug_launch.py ({hooked_module}: {hooked_exports})")


def log_probe_state(logdir):
    stdout = text(logdir / "stdout.txt")
    found = re.search(r"^debug-probe log site (0x[0-9a-f]+) byte (0x[0-9a-f]+) calls (\d+)$", stdout, re.M)
    assert found and int(found[3]) == LOG_CALLS, stdout
    return tuple(int(value, 16) for value in found.groups()[:2])


def log_at_records(logdir, rva, count):
    lines = [line for line in text(logdir / "files.txt").splitlines() if " LOG_AT " in line]
    assert len(lines) == count, lines
    records = []
    for hit, line in enumerate(lines, 1):
        event = re.fullmatch(rf"\s*\d+\.\d{{3}} LOG_AT tid=[1-9]\d* rva={rva:#x} hit={hit} (.+)", line)
        assert event, line
        records.append(event[1])
    return records


GROUPS = ('core', 'log', 'hooks')


def main(probe, calls_dll, group='all'):
    """Run one group of cases (core, log, hooks) or all of them; ctest runs the groups as separate tests."""
    if group != 'all' and group not in GROUPS:
        raise SystemExit(f'unknown group {group!r}; use all or one of {GROUPS}')
    groups = GROUPS if group == 'all' else (group,)
    sys.path.insert(0, str(DEBUG.parent))
    from debug_launch import MOVIE_CALL_HOOKS, MOVIE_MODULE, read_qword, read_stack
    if 'hooks' in groups:
        require_hooked(HOOKED_MODULE, HOOKED_EXPORT, MOVIE_MODULE, MOVIE_CALL_HOOKS)
        # Control: the guard rejects the fixture once its export is dropped from the hook list.
        try:
            require_hooked(HOOKED_MODULE, HOOKED_EXPORT, MOVIE_MODULE, tuple(e for e in MOVIE_CALL_HOOKS if e != HOOKED_EXPORT))
        except AssertionError:
            pass
        else:
            raise AssertionError("the hook-list guard accepted an export the launcher does not hook")
    if 'core' in groups:
        word = bytes.fromhex("ab cd ef 01 23 45 67 89")
        for readable in range(8):
            assert read_qword(lambda address, size: word[:readable], 0) is None
        assert read_qword(lambda address, size: word[:size], 0) == 0x8967452301efcdab
        assert read_qword(lambda address, size: bytes(size), 0) == 0
        # read_stack on a stack whose committed pages end at 0x7000: a read past that end fails as a whole,
        # so only the retry within RSP's page recovers the words nearest RSP.
        def stack_read(address, size):
            return bytes((address + i) & 0xff for i in range(size)) if address + size <= 0x7000 else b""
        for rsp, recovered in ((0x6000, 1600), (0x6a00, 1536), (0x6fe0, 32), (0x6ff8, 8)):
            data = read_stack(stack_read, rsp, 200)
            assert len(data) == 1600 and data[:recovered] == stack_read(rsp, recovered), hex(rsp)
            assert data[recovered:] == bytes(1600 - recovered), hex(rsp)
        assert read_stack(lambda address, size: b"", 0x6000, 200) == bytes(1600)
    probe = Path(probe).resolve()
    image = probe.name
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)

        # Baseline: both breakpoints are stepped over and the probe exits 0 (core and log both need its sites).
        if group != 'hooks':
            out, logdir = run(probe, work, "baseline", {})
            assert "process exited with 0x00000000" in out, out
            sites = [int(m, 16) for m in re.findall(r"breakpoint at %s\+0x([0-9a-f]+) first-chance" % re.escape(image),
                                                    text(logdir / "events.txt"))]
            assert len(sites) == 2 and sites[0] != sites[1], text(logdir / "events.txt")
            assert "debugger stop " not in text(logdir / "events.txt")
            assert not (logdir / "threads_stop.txt").exists() and "VALLOC" not in text(logdir / "files.txt")
            log_rva, opcode = log_probe_state(logdir)
            assert opcode != 0xcc and "LOG_AT" not in out and "LOG_AT" not in text(logdir / "files.txt")
        if 'core' in groups:
            # The launcher keeps the debug heap off by default ...
            assert "debug-probe heap flags 0x0\n" in text(logdir / "stdout.txt"), missing_heap_flags(logdir)
            # ... and a caller who sets _NO_DEBUG_HEAP=0 gets it back (control: the loader's three heap bits).
            out, logdir = run(probe, work, "debug-heap", {"_NO_DEBUG_HEAP": "0"})
            assert "process exited with 0x00000000" in out, out
            assert "debug-probe heap flags 0x70\n" in text(logdir / "stdout.txt"), missing_heap_flags(logdir)

            out, _ = run(probe, work, "unlimited", {}, options=("--timeout", "0", "--continue-asserts", "-1"))
            assert "process exited with 0x00000000" in out and "timeout after" not in out, out
            out, _ = run(probe, work, "assert-cap", {}, options=("--timeout", "0", "--continue-asserts", "1"))
            assert f"stopped at breakpoint (0x80000003) at {image}+{sites[1]:#x}" in out, out
            out, _ = run(probe, work, "finite", {}, options=("--timeout", "1", "--continue-asserts", "-1"))
            assert "timeout after" in out and "process exited with" not in out, out

            # X2W_STOP_AT on the second site: the first is still stepped over, the run stops at the second.
            out, logdir = run(probe, work, "stop", {"X2W_STOP_AT": f"{sites[1]:x}"})
            assert f"stopped at breakpoint (0x80000003) at {image}+{sites[1]:#x}" in out and "first chance" in out, out
            events = text(logdir / "events.txt")
            assert events.splitlines().count("debugger stop 0x80000003 first-chance") == 1, events
            assert f"breakpoint at {image}+{sites[0]:#x}" in events, events
            stacks = text(logdir / "threads_stop.txt")
            assert image in stacks and "rip=" in stacks, stacks
            # Control: an address that is not a breakpoint site stops nothing.
            out, logdir = run(probe, work, "stop-control", {"X2W_STOP_AT": f"{sites[1] + 0x1000:x}"})
            assert "process exited with 0x00000000" in out and not (logdir / "threads_stop.txt").exists(), out
            # Control: without --continue-asserts the first site stops the run as before, with or without the switch.
            out, _ = run(probe, work, "stop-first", {"X2W_STOP_AT": f"{sites[1]:x}"}, options=())
            assert f"stopped at breakpoint (0x80000003) at {image}+{sites[0]:#x}" in out, out

            # X2W_NOP_SOFT_ASSERTS=1: a site that traps once per iteration is patched after its first
            # hit, so the loop finishes with one logged breakpoint. Without the switch every hit is logged.
            repeats = 4
            out, logdir = run(probe, work, "nop-repeat", {"STARTUP_DEBUG_PROBE_ASSERTS": str(repeats),
                                                          "X2W_NOP_SOFT_ASSERTS": "1"})
            assert "process exited with 0x00000000" in out, out
            assert f"debug-probe repeated {repeats}\n" in text(logdir / "stdout.txt"), text(logdir / "stdout.txt")
            hits = breakpoint_sites(logdir, image)
            repeated = [site for site in hits if site not in sites]
            assert len(hits) == len(sites) + 1 and len(repeated) == 1, text(logdir / "events.txt")
            nop_line = next((line for line in out.splitlines() if "soft asserts NOP'd:" in line), "")
            assert f"{image}+{repeated[0]:#x}" in nop_line, out
            # Control: without the switch the same run logs every hit at the repeated site.
            out, logdir = run(probe, work, "nop-repeat-control", {"STARTUP_DEBUG_PROBE_ASSERTS": str(repeats)})
            assert "process exited with 0x00000000" in out, out
            assert breakpoint_sites(logdir, image).count(repeated[0]) == repeats, text(logdir / "events.txt")

            # X2W_DUMP_AT: a dump one second in, while the probe is still alive.
            out, logdir = run(probe, work, "dump", {"X2W_DUMP_AT": "1"})
            assert "process exited with 0x00000000" in out, out
            assert "rip=" in text(logdir / "threads_1.txt"), text(logdir / "threads_1.txt")
            assert not (work / "baseline" / "threads_1.txt").exists()

            # <logdir>/stop ends the run with thread stacks even when the timer is disabled, while the probe
            # (which waits 3 s) is still alive. Control: a stop file left in the directory before the run is removed
            # at start, so that run exits normally.
            out, logdir = run_with_stop(probe, work, "stop-file")
            assert "stopped by request after" in out and "process exited" not in out, out
            assert "stop file found" in text(logdir / "events.txt"), text(logdir / "events.txt")
            assert "rip=" in text(logdir / "threads.txt"), text(logdir / "threads.txt")
            stale = work / "stop-stale"
            stale.mkdir(exist_ok=True)
            (stale / "stop").write_text("", encoding="utf-8")
            out, logdir = run(probe, work, "stop-stale", {})
            assert "process exited with 0x00000000" in out and "stopped by request" not in out, out
            assert not (logdir / "stop").exists(), "stale stop file kept"
        if 'hooks' in groups:
            # X2W_VALLOC_MIN: the 256 MiB reservation is logged with a caller in the probe; a higher
            # threshold logs nothing.
            out, logdir = run(probe, work, "valloc", {"X2W_VALLOC_MIN": "10000000"})
            lines = [line for line in text(logdir / "files.txt").splitlines() if " VALLOC " in line]
            ours = [line for line in lines if "size=0x10000000" in line and "type=0x2000" in line]
            assert len(ours) == 1 and "requested=0x0" in ours[0] and image in ours[0].split("callers:")[1], lines
            out, logdir = run(probe, work, "valloc-control", {"X2W_VALLOC_MIN": "20000000"})
            assert "process exited with 0x00000000" in out and " VALLOC " not in text(logdir / "files.txt"), out
            # An empty value is the same as unset.
            out, logdir = run(probe, work, "valloc-empty", {"X2W_VALLOC_MIN": ""})
            assert "process exited with 0x00000000" in out and " VALLOC " not in text(logdir / "files.txt"), out
        if 'log' in groups:
            # Log two out of 19 calls, even with the usual file hooks off. RDX changes each call,
            # RCX points into a known array, and R8 - 1 is the unreadable null address.
            entry = f"{log_rva:#x}@2:rdx;[rcx+8];[rcx-8];[r8-1]"
            malformed = ("xyz", f"{log_rva:x}@0", f"{log_rva:x}:rdx;[rcx*8]", f"{log_rva:x}:rip", f"{log_rva:x}:rdx;rdx",
                         f"{log_rva:x}:[rcx+8];[RCX+0x8]")
            unusable = (0xffffffff, 0, sites[0])
            out, logdir = run(probe, work, "log-at",
                              {"X2W_LOG_AT": ",".join((*malformed, entry, *(f"{rva:x}:rax" for rva in unusable), entry))},
                              options=("--continue-asserts", "10", "--offline", "0"))
            assert "process exited with 0x00000000" in out and f"LOG_AT hit counts: +{log_rva:x} x2\n" in out, out
            assert re.search(r"^  hook windows: entries=\d+ windows=\d+ unprotected=0 "
                             r"deferred=\d+ timeouts=0 protection=on$", out, re.M), out
            assert log_probe_state(logdir) == (log_rva, opcode), text(logdir / "stdout.txt")
            assert log_at_records(logdir, log_rva, 2) == [
                f"rdx={LOG_TOKEN + i:#x} [rcx+8]=0x89abcdef01234567 "
                "[rcx-8]=0x1020304050607080 [r8-1]=unreadable"
                for i in range(2)
            ], text(logdir / "files.txt")
            events = text(logdir / "events.txt")
            for invalid in (*malformed, entry):
                assert f"X2W_LOG_AT ignored {invalid!r}:" in events, events
            for rva in unusable:
                assert f"X2W_LOG_AT ignored {rva:#x}:" in events, events

            # Without a limit or item list, log all 16 GPRs for the first 16 hits only. The
            # shared X2W_COUNT breakpoint must still count the calls beyond the logging limit.
            out, logdir = run(probe, work, "log-at-default",
                              {"X2W_LOG_AT": f"{log_rva:x}", "X2W_COUNT": f"{log_rva:x}"})
            assert "process exited with 0x00000000" in out, out
            assert f"LOG_AT hit counts: +{log_rva:x} x16\n" in out, out
            assert f"  hit counts: +{log_rva:x} x{LOG_CALLS}\n" in out, out
            for i, record in enumerate(log_at_records(logdir, log_rva, 16)):
                registers = dict(field.split("=") for field in record.split())
                assert set(registers) == LOG_REGISTERS, record
                assert int(registers["rdx"], 16) == LOG_TOKEN + i and registers["r8"] == "0x1", record
            assert log_probe_state(logdir) == (log_rva, 0xcc), text(logdir / "stdout.txt")

            # Eight threads hit one site 16000 times with a limit of 1. INT3 events that other threads
            # queued before the breakpoint was removed must resume, not stop the run as asserts.
            found = re.search(r"^debug-probe threaded log site (0x[0-9a-f]+) byte (0x[0-9a-f]+) calls (\d+)$",
                              text(logdir / "stdout.txt"), re.M)
            assert found, text(logdir / "stdout.txt")
            threaded_rva = int(found[1], 16)
            for attempt in range(3):
                out, logdir = run(probe, work, f"log-at-threads-{attempt}", {"X2W_LOG_AT": f"{threaded_rva:x}@1:rcx"})
                stdout = text(logdir / "stdout.txt")
                assert "process exited with 0x00000000" in out and "debugger stop " not in text(logdir / "events.txt"), out
                assert f"LOG_AT hit counts: +{threaded_rva:x} x1\n" in out, out
                assert f"debug-probe threaded log site {threaded_rva:#x} byte" in stdout and "calls 16000\n" in stdout, stdout
                assert f"breakpoint at {image}+{threaded_rva:#x}" not in text(logdir / "events.txt"), text(logdir / "events.txt")
                assert len([line for line in text(logdir / "files.txt").splitlines() if " LOG_AT " in line]) == 1
        if 'hooks' in groups:
            # Call hooks: off by default, so the 7 export calls stop nothing; with --trace-calls 1 the
            # launcher counts each of them (the positive control that the probe reaches the hook).
            folder = work / "calls-bin"
            folder.mkdir()
            hooked_probe = Path(shutil.copy2(probe, folder / image))
            dll = Path(shutil.copy2(calls_dll, folder / HOOKED_MODULE))
            variables = {"STARTUP_DEBUG_PROBE_DLL": str(dll), "STARTUP_DEBUG_PROBE_EXPORT": HOOKED_EXPORT}
            banners = {"calls-default": "call tracing: off (", "calls-on": "call tracing: on (",
                       "calls-no-file-tracing": "call tracing: off: --trace-calls 1 needs file tracing"}
            for name, extra, want in (("calls-default", (), None),
                                      ("calls-on", ("--trace-calls", "1"), f"{HOOKED_EXPORT} x{CALLS}"),
                                      # Requested without file tracing: nothing is armed, and the summary says why.
                                      ("calls-no-file-tracing", ("--offline", "0", "--trace-calls", "1"), None)):
                out, logdir = run(hooked_probe, work, name, variables, options=("--continue-asserts", "10", *extra))
                assert "process exited with 0x00000000" in out, out
                assert f"debug-probe calls {CALLS}\n" in text(logdir / "stdout.txt"), text(logdir / "stdout.txt")
                events = text(logdir / "events.txt")
                assert re.search(r" load %s at " % re.escape(HOOKED_MODULE), events, re.I), events
                found = re.search(r"bink calls: (.*)", out)
                assert (found.group(1).strip() if found else None) == want, (name, out)
                assert banners[name] in out, (name, out)

    print(f"startup_debug[{group}]: PASS (stop at a chosen assert, timed thread dump, large-allocation log, debug heap off, "
          "soft asserts NOP'd after the first hit when the core group runs, call hooks opt-in, bounded register and qword logging)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
