"""Check the periodic window pictures of tools/startup/debug_launch.py (--capture-every) on a source-built window.

usage: startup_debug_window_check.py <startup_window_probe.exe>

tests/startup_window_probe.cpp shows a 160x120 window that is red, then blue after 3.5 s. The run lasts 9 s with
--capture-every 1. The saved window_<sss>s.png files must show red first and blue last, the frames in between
must have been skipped as unchanged, the first change must be reported inside the window of time the probe
changes it, and the end-of-run window.png must still be written (blue). Controls: without --capture-every no
window_<sss>s.png exists and no summary line is printed; an interval below one second is refused.
Exit code 77 (skip) only when the machine shows no window at all.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from startup_window_capture_check import decode_png  # noqa: E402

DEBUG = HERE.parent / "tools/startup/debug_launch.py"
RED, BLUE = (200, 30, 30), (30, 30, 200)


def run(probe, work, name, *options, extra_env=None):
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
    env.update(extra_env or {})
    command = [sys.executable, "-B", str(DEBUG), str(probe), str(work), str(logdir), "--console", "0", *options]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    return result, logdir


def pixel(path):
    width, height, rows = decode_png(path.read_bytes())
    assert (width, height) == (160, 120), (path.name, width, height)
    return tuple(rows[60][3 * 80:3 * 80 + 3])


def main(probe):
    probe = Path(probe).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        result, logdir = run(probe, work, "periodic", "--timeout", "9", "--capture-every", "1")
        out = result.stdout + result.stderr
        if "window capture: no visible window" in out:
            print("startup_debug_window: SKIP (no window is visible on this machine)")
            return 77
        frames = sorted(logdir.glob("window_*s.png"))
        assert len(frames) >= 2, (out, [p.name for p in frames])
        assert all(re.fullmatch(r"window_\d{3}s\.png", p.name) for p in frames), [p.name for p in frames]
        first, last = pixel(frames[0]), pixel(frames[-1])
        assert first == RED and last == BLUE, (first, last, [p.name for p in frames])
        assert all(pixel(p) in (RED, BLUE) for p in frames), [p.name for p in frames]
        # Every tick is saved, repeats included: red frames, then blue frames; the first blue one is the change.
        found = re.search(r"window captures: (\d+) saved \((\d+) identical to the one before\), first change at (\d+) s", out)
        assert found, out
        saved, repeated, change = (int(found.group(i)) for i in (1, 2, 3))
        assert saved == len(frames) and repeated == saved - 2 and repeated >= 3 and 3 <= change <= 6, found.group(0)
        assert min(int(p.name[7:10]) for p in frames if pixel(p) == BLUE) == change, ([p.name for p in frames], change)
        assert pixel(logdir / "window.png") == BLUE, "the end-of-run picture must stay as before"
        assert "window capture: 160x120 -> window.png" in out, out

        control, control_dir = run(probe, work, "control", "--timeout", "4")
        assert not list(control_dir.glob("window_*s.png")), "no periodic file without --capture-every"
        assert "window captures:" not in control.stdout and (control_dir / "window.png").exists(), control.stdout

        for bad in ("0.5", "-1", "abc"):
            refused, _ = run(probe, work, "bad", "--timeout", "1", "--capture-every", bad)
            assert refused.returncode == 2 and "--capture-every" in refused.stdout, (bad, refused.returncode, refused.stdout)
        # The commit headroom (--commit-headroom-mib): more than any machine has free ends the run long before the timeout,
        # with the reason in the summary and the end-of-run window.png still written; one MiB never fires.
        started = time.monotonic()
        ended, ended_dir = run(probe, work, "headroom", "--timeout", "40", "--commit-headroom-mib", "1000000000")
        took = time.monotonic() - started
        assert "stopped by request" in ended.stdout and "(commit headroom: system commit has " in ended.stdout, ended.stdout
        assert "under the headroom of 1000000000 MiB" in ended.stdout and took < 25, (took, ended.stdout)
        assert (ended_dir / "window.png").exists() and "commit headroom: stop below 1000000000 MiB, lowest free commit" in ended.stdout
        roomy, _ = run(probe, work, "roomy", "--timeout", "4", "--commit-headroom-mib", "1")
        assert "timeout after" in roomy.stdout and "commit headroom: stop below 1 MiB, lowest free commit" in roomy.stdout, roomy.stdout
        # A warning threshold above anything free logs a warning and never ends the run; the stop stays at one MiB.
        warned, warned_dir = run(probe, work, "warned", "--timeout", "7", "--commit-headroom-mib", "1", "--commit-warn-mib", "1000000000")
        assert "timeout after" in warned.stdout and "warning below 1000000000 MiB (" in warned.stdout, warned.stdout
        assert "commit headroom warning: system commit has " in (warned_dir / "events.txt").read_text(encoding="utf-8", errors="replace")
        for option in ("--commit-headroom-mib", "--commit-warn-mib"):
            for bad in ("-1", "abc", "1.5"):
                refused, _ = run(probe, work, "badhead", "--timeout", "1", option, bad)
                assert refused.returncode == 2 and option in refused.stdout, (option, bad, refused.returncode, refused.stdout)
        # A title whose window thread is stuck (the probe stops pumping messages 1.5 s in): PrintWindow waits for it, the
        # run must still end at its timeout, with finished.txt written, the title gone and under 20 s in all.
        for name, options in (("hung_window", ()), ("hung_window_periodic", ("--capture-every", "1"))):
            started = time.monotonic()
            hung, hung_dir = run(probe, work, name, "--timeout", "5", *options, extra_env={"STARTUP_WINDOW_PROBE_HANG_MS": "1500"})
            took = time.monotonic() - started
            assert hung.returncode == 0 and (hung_dir / "finished.txt").exists(), (name, hung.returncode, hung.stdout, hung.stderr)
            assert took < 20, (name, took, hung.stdout)
            pid = int((hung_dir / "title.pid").read_text())
            alive = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True).stdout
            assert str(pid) not in alive, (name, "the title is still running", alive)
        # The tester's case (Run M): after the timeout the title's window thread raises a debug event (OutputDebugString) on every
        # pass. The debugger must keep serving debug events while it takes the last picture and ends the title.
        started = time.monotonic()
        busy, busy_dir = run(probe, work, "debug_events", "--timeout", "5", "--capture-every", "1",
                             extra_env={"STARTUP_WINDOW_PROBE_DEBUG_MS": "3000"})
        took = time.monotonic() - started
        assert busy.returncode == 0 and (busy_dir / "finished.txt").exists(), (busy.returncode, busy.stdout, busy.stderr)
        assert took < 20 and (busy_dir / "window.png").exists(), (took, busy.stdout)
        assert "window capture: 160x120 -> window.png" in busy.stdout and "skipped" not in busy.stdout, busy.stdout
        # events.txt follows the modules and threads: an unloaded DLL is logged and dropped, thread creation and exit carry
        # the live thread count.
        _, churn_dir = run(probe, work, "churn", "--timeout", "3", extra_env={"STARTUP_WINDOW_PROBE_CHURN": "1"})
        events = (churn_dir / "events.txt").read_text(encoding="utf-8", errors="replace")
        assert re.search(r"load .*msimg32\.dll at 0x", events, re.I) and re.search(r"unload .*msimg32\.dll at 0x", events, re.I), events[:2000]
        assert re.search(r"thread \d+ created \(\d+ live\)", events) and re.search(r"thread \d+ exited \(\d+ live\)", events), events[:2000]
    print("startup_debug_window: PASS (red then blue saved once each, unchanged frames skipped, change time reported, "
          "window.png unchanged, off by default, bad interval refused; commit headroom ends a run, one MiB does not)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
