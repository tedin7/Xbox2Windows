"""Check that the startup debugger's function hooks see every concurrent call (#1281).

usage: startup_hook_check.py                              the step-window bookkeeping
       startup_hook_check.py <startup_launch_probe.exe>  two threads through a hooked function

A hook is one patched byte. While one thread steps over it the original byte is back, so a second
thread entering the function at that moment ran untraced. tools/startup/hook_window.py holds the
other threads for the step. The first form checks that bookkeeping with fake threads and needs no
Windows. The second runs the source-built probe, whose two threads open one file 32 times each,
entering CreateFileW together, under tools/startup/debug_launch.py: every one of the 64 opens must
appear in files.txt with its result, and the run reports no unprotected window. Control:
--suspend-others 0 must lose opens and say so.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

TOOLS = Path(__file__).resolve().parents[1] / "tools/startup"
sys.path.insert(0, str(TOOLS))
from hook_window import FAILED, GONE, HELD, STILL_ACTIVE, STATUS_INVALID_HANDLE, STATUS_THREAD_IS_TERMINATING, ReturnBook, StepWindow, classify_hold, classify_status  # noqa: E402

NAME = "x2w_hook_overlap.dat"
OPENS = 64


def fakes(results=None):
    log = {"held": [], "released": []}
    results = results or {}

    def hold(tid):
        result = results.get(tid, HELD)
        if result == HELD:
            log["held"].append(tid)
        return result

    return log, hold, lambda tid: log["released"].append(tid)


def bookkeeping():
    # Every other thread is held before the step and resumed when the owner finishes; the owner never is.
    log, hold, release = fakes()
    window = StepWindow(hold, release)
    window.open(1, [1, 2, 3], now=0)
    assert log["held"] == [2, 3] and not window.defer(1) and window.defer(2) and window.deferred == 1
    assert not window.finish(2) and log["released"] == [], "only the owner closes a window"
    assert window.finish(1) and sorted(log["released"]) == [2, 3] and not window.defer(2)
    assert window.unprotected == 0 and window.timeouts == 0
    window.open(2, [1], now=1)  # a closed window can be opened again
    window.abandon()

    # A thread created during the step is held too; one that never joins the window is not.
    log, hold, release = fakes()
    window = StepWindow(hold, release)
    window.hold(9)
    assert log["held"] == []
    window.open(1, [2], now=0)
    window.hold(9)
    window.hold(9)
    window.hold(1)
    assert log["held"] == [2, 9]
    window.finish(1)
    assert sorted(log["released"]) == [2, 9]

    # A thread that is gone needs nothing; one that cannot be held leaves the window unprotected.
    log, hold, release = fakes({2: GONE, 3: FAILED})
    window = StepWindow(hold, release)
    window.open(1, [2, 3, 4], now=0)
    assert log["held"] == [4] and window.unprotected == 1
    window.finish(1)
    assert log["released"] == [4]

    # A window that outlives its limit is closed, resumed and counted, once.
    log, hold, release = fakes()
    window = StepWindow(hold, release, limit=2.0)
    window.open(1, [2], now=10)
    assert not window.expire(11.9) and log["released"] == []
    assert window.expire(12.0) and log["released"] == [2]
    assert window.timeouts == 1 and window.unprotected == 1
    assert not window.expire(99) and not window.finish(1) and window.timeouts == 1

    # With protection off nothing is held or deferred, and a window with company is reported.
    log, hold, release = fakes()
    window = StepWindow(hold, release, protect=False)
    window.open(1, [2], now=0)
    assert log["held"] == [] and not window.defer(2) and window.unprotected == 1
    window.finish(1)
    window.open(1, [], now=0)
    assert window.windows == 2 and window.unprotected == 1

    # A thread that was not held and steps while a window is open does not take the window over: the
    # owner keeps it, the held threads stay held until the owner is done, and the window is exposed.
    log, hold, release = fakes({3: FAILED})
    window = StepWindow(hold, release)
    window.open(1, [2, 3], now=0)
    assert not window.defer(3), "a thread that is not held cannot be deferred"
    window.open(3, [1, 2], now=0)
    assert window.owner == 1 and window.unprotected == 1 and log["released"] == []
    assert not window.finish(3) and window.finish(1) and log["released"] == [2]

    returns_bookkeeping()
    hold_outcomes()

    # The summary is always there, with zeros.
    line = StepWindow(hold, release).summary(0)
    assert line == "hook windows: entries=0 windows=0 unprotected=0 deferred=0 timeouts=0 protection=on", line


def hold_outcomes():
    """A thread that called ExitThread but whose EXIT_THREAD event is not read yet cannot be suspended
    (access denied). It runs no code, so it is GONE; the same failure on a live thread is FAILED."""
    access_denied = 5
    assert classify_hold(0, signaled=False) == GONE, "exited with code 0"
    assert classify_hold(STILL_ACTIVE + 1) == GONE
    assert classify_hold(STILL_ACTIVE, signaled=True) == GONE, "exit code not stored yet, handle signaled"
    assert classify_hold(None, error=87) == GONE, "no such thread"
    assert classify_hold(STILL_ACTIVE) == FAILED, "alive and not suspendable: a real hole"
    assert classify_hold(None, error=access_denied) == FAILED, "exit code unknown: not assumed gone"
    assert classify_status(STATUS_THREAD_IS_TERMINATING) == GONE, "exiting thread, exit code still 259"
    assert classify_status(STATUS_INVALID_HANDLE) == GONE
    assert classify_status(0xC0000022) == FAILED, "access denied: not assumed gone"
    # The window counts a gone thread as nothing and a live unsuspendable one as unprotected.
    for outcome, unprotected in ((classify_hold(0), 0), (classify_status(STATUS_THREAD_IS_TERMINATING), 0),
                                 (classify_hold(STILL_ACTIVE), 1)):
        log, hold, release = fakes({2: outcome})
        window = StepWindow(hold, release)
        window.open(1, [2, 3], now=0)
        assert window.unprotected == unprotected and log["held"] == [3]
        window.finish(1)


def traced_returns(retrapped):
    """Two threads call from fixed stack slots, 32 calls each. A call in `retrapped` hits its entry
    hook twice: another thread re-patched the byte before this one ran the original instruction, so
    the same call is registered again. Returns how many lines the book writes for the 64 calls."""
    book, lines = ReturnBook(), 0
    for call in range(32):
        for tid in (1, 2):
            for _ in range(2 if (call, tid) in retrapped else 1):
                book.add(0x7000, tid, 0x1000 * tid, f"NtCreateFile call {call}", 0x5000)
            lines += len(book.take(0x7000, tid, 0x1000 * tid))
    return lines


def returns_bookkeeping():
    # One call is traced once however many times its entry trapped: a run showed "88 of 64".
    retrapped = {(call, 1) for call in range(0, 32, 4)} | {(call, 2) for call in range(0, 32, 4)} \
        | {(call, 1) for call in range(1, 32, 4)}
    assert len(retrapped) == 24
    assert traced_returns(set()) == 64
    assert traced_returns(retrapped) == 64, traced_returns(retrapped)
    book = ReturnBook()
    book.add(0x10, 1, 0x100, "a", 0x5000)
    book.add(0x10, 2, 0x100, "b", 0x5000)  # same slot value on another thread is another call
    assert book.take(0x10, 1, 0x100) == ["a"] and book.take(0x10, 1, 0x100) == []
    assert book.take(0x10, 2, 0x100) == ["b"] and book.take(0x99, 2, 0x100) == []

    # Tail-calling another hooked function shares (tid, slot) with a different entry address:
    # both return records stay, while a re-trap of the same entry replaces only that entry.
    book.add(0x20, 1, 0x200, "CreateFileW", 0x5000)
    book.add(0x20, 1, 0x200, "NtCreateFile first", 0x6000)
    book.add(0x20, 1, 0x200, "NtCreateFile retrapped", 0x6000)
    assert book.take(0x20, 1, 0x200) == ["CreateFileW", "NtCreateFile retrapped"]
    assert book.take(0x20, 1, 0x200) == []

    # A restore callback runs once, when its call returns, and only for that call.
    undone = []
    book.add(0x40, 1, 0x400, "mapped", 0x5000, restore=lambda: undone.append("first"))
    book.add(0x40, 2, 0x400, "other thread", 0x5000, restore=lambda: undone.append("second"))
    assert book.take(0x40, 1, 0x400) == ["mapped"] and undone == ["first"]
    assert book.take(0x40, 1, 0x400) == [] and undone == ["first"], "a restore must not run twice"
    # A re-trap of the same call keeps the first restore (the caller's original state).
    book.add(0x50, 1, 0x500, "x", 0x5000, restore=lambda: undone.append("original"))
    book.add(0x50, 1, 0x500, "x", 0x5000, restore=lambda: undone.append("rewritten twice"))
    book.take(0x50, 1, 0x500)
    assert undone == ["first", "second"][:1] + ["original"], undone
    assert book.take(0x40, 2, 0x400) == ["other thread"] and undone[-1] == "second"

    # The launcher asks whether any call still returns to an address before it lets a hit-limited
    # breakpoint run free: a returned call must not keep it armed, a waiting one must.
    assert not book.pending(0x20) and not book.pending(0x99)
    book.add(0x30, 1, 0x300, "waiting", 0x5000)
    assert book.pending(0x30)
    book.take(0x30, 1, 0x300)
    assert not book.pending(0x30)


def run(probe, scratch, name, options):
    logdir = scratch / name
    data = scratch / NAME
    data.write_bytes(b"x2w")
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
    env.update(X2W_LAUNCH_PROBE_MODE="overlap", X2W_LAUNCH_PROBE_FILE=str(data))
    command = [sys.executable, "-B", str(TOOLS / "debug_launch.py"), str(probe), str(scratch), str(logdir),
               "--timeout", "60", "--console", "0", "--trace-files", "1", "--continue-asserts", "0", *options]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    files = (logdir / "files.txt").read_text(encoding="utf-8", errors="replace")
    traced = sum(1 for line in files.splitlines()
                 if " NtCreateFile " in line and NAME in line.lower() and line.endswith("-> 0x00000000"))
    stdout = (logdir / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    summary = re.search(r"hook windows: entries=\d+ windows=(\d+) unprotected=(\d+) deferred=\d+ timeouts=(\d+) "
                        r"protection=(on|off)", result.stdout + files)
    assert summary, (result.returncode, result.stdout, result.stderr[-2000:])
    return traced, stdout, [int(summary.group(i)) for i in (1, 2, 3)], summary.group(4)


def launcher(probe, scratch):
    for attempt in range(3):
        traced, stdout, (windows, unprotected, timeouts), protection = run(probe, scratch, f"held{attempt}", [])
        assert "overlap opens 64 of 64" in stdout, stdout
        assert traced == OPENS, f"attempt {attempt}: {traced} of {OPENS} opens traced (more than made means double counting)"
        assert windows >= OPENS and unprotected == 0 and timeouts == 0 and protection == "on", (windows, unprotected, timeouts)
    # Control: without holding the other thread the same run loses opens, and the report says so.
    for attempt in range(8):
        traced, stdout, (windows, unprotected, timeouts), protection = run(probe, scratch, f"free{attempt}",
                                                                          ["--suspend-others", "0"])
        assert "overlap opens 64 of 64" in stdout, stdout
        assert protection == "off"
        assert traced <= OPENS, f"control attempt {attempt}: {traced} traces for {OPENS} opens: a call was counted twice"
        lost = max(0, OPENS - traced)
        if lost:
            assert unprotected > 0, "opens were lost but the run reported no unprotected window"
            return lost
    raise AssertionError(f"the control never lost an open in 8 runs: the overlap was not exercised ({traced} of {OPENS})")


if __name__ == "__main__":
    bookkeeping()
    if len(sys.argv) == 1:
        print("startup_hook_window: PASS (hold, defer, finish, new thread, gone/failed thread, deadline, protection off, summary)")
        sys.exit(0)
    with tempfile.TemporaryDirectory() as temporary:
        lost = launcher(Path(sys.argv[1]).resolve(), Path(temporary))
    print(f"startup_hook_overlap: PASS ({OPENS} of {OPENS} opens traced with the window; control lost {lost})")
