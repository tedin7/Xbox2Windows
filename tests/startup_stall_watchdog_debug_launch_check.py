#!/usr/bin/env python3
"""The stall watchdog ends a run that tools/startup/debug_launch.py is hosting (Windows only).

  startup_stall_watchdog_debug_launch_check.py <startup_launch_probe.exe>

A hand kill of a debugged title can fail, so the watchdog never kills: it writes <logdir>/stop, the file debug_launch.py
checks twice a second, and the run ends as at the timeout. This check hosts the source-built probe in its `sleep` mode (a
process that is alive and never presents, reads or translates anything) under debug_launch.py with a 60 s timeout:

  1. with stall_watchdog.py beside it (frozen after 3 s) debug_launch.py ends the run by the stop file long before the
     probe's own 30 s sleep and the timeout, and stall_stop.json names the frozen rule;
  2. control: the same run with the frozen rule turned off (--stall-seconds 0) is not ended early: it lasts until the probe
     exits by itself.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools" / "startup"


def run_hosted(probe, work, name, stall_seconds):
    logdir = work / name
    logdir.mkdir()
    env = dict(os.environ, X2W_LAUNCH_PROBE_MODE="sleep")
    env.pop("X2W_LOW_ADDRESSES", None)
    started = time.monotonic()
    host = subprocess.Popen([sys.executable, "-B", str(TOOLS / "debug_launch.py"), str(probe), str(work), str(logdir),
                             "--timeout", "60", "--console", "0"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    watchdog = subprocess.Popen([sys.executable, "-B", str(TOOLS / "stall_watchdog.py"), str(logdir), "--parent", str(host.pid),
                                 "--seconds", "60", "--stall-seconds", str(stall_seconds), "--min-seconds", "0",
                                 "--memory-percent", "0"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        host_output, _ = host.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        host.kill()
        raise
    elapsed = time.monotonic() - started
    try:
        watchdog_output, _ = watchdog.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        watchdog.kill()
        watchdog_output = ""
    finished = (logdir / "finished.txt").read_text(encoding="utf-8", errors="replace") if (logdir / "finished.txt").exists() else ""
    return elapsed, host_output, watchdog_output, finished, logdir


def main(probe):
    probe = Path(probe)
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)

        elapsed, host_output, watchdog_output, finished, logdir = run_hosted(probe, work, "stalled", 3)
        assert elapsed < 25, f"the hosted run was not ended early: {elapsed:.1f} s\n{host_output[-600:]}"
        assert "(stop file)" in host_output + finished, (host_output[-600:], finished[-400:])
        assert "tester_run: STALL STOP at " in watchdog_output, watchdog_output
        stop = json.loads((logdir / "stall_stop.json").read_text(encoding="utf-8"))
        assert stop["reason"].startswith("frozen"), stop

        elapsed_control, host_control, watchdog_control, _, logdir_control = run_hosted(probe, work, "control", 0)
        assert elapsed_control >= 25, f"the control ended early ({elapsed_control:.1f} s): {host_control[-600:]}"
        assert not (logdir_control / "stall_stop.json").exists(), watchdog_control
        assert "(stop file)" not in host_control, host_control[-600:]

        print(f"stall watchdog under debug_launch: stalled probe ended by the stop file after {elapsed:.1f} s "
              f"({stop['reason']}); control without the frozen rule ran {elapsed_control:.1f} s")
    return 0


if __name__ == "__main__":
    if sys.platform != "win32":
        print("startup_stall_watchdog_debug_launch_check: Windows only, skipped")
        sys.exit(77)
    if len(sys.argv) != 2:
        sys.exit("usage: startup_stall_watchdog_debug_launch_check.py <startup_launch_probe.exe>")
    sys.exit(main(sys.argv[1]))
