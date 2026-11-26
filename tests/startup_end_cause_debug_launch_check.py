#!/usr/bin/env python3
"""tester_pack.py end-cause names why a run ended, for every way a run hosted by tools/startup/debug_launch.py can end (Windows only).

  startup_end_cause_debug_launch_check.py <startup_launch_probe.exe>

The source-built probe is hosted by the real debugger, once per way out; the log folder is then read by `tester_pack.py end-cause`:

  timeout         the title lives on, --timeout 4 ends it          -> "timeout after"
  stop file       a `stop` file is written from outside            -> "stopped by request"
  exited          the title returns 0 by itself                    -> "title exited by itself"
  killed          the title process is killed from outside         -> "title killed externally"
  crashed         the title raises an access violation             -> "title crashed (exit code 0xC0000005"
  watchdog        stall_watchdog.py writes the stop file           -> "watchdog at"

Every line ends with the last present milestone ("last present"), which a probe without a heartbeat reports as none.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools" / "startup"


def host(probe, work, name, mode, timeout, code=None):
    logdir = work / name
    logdir.mkdir()
    env = dict(os.environ, X2W_LAUNCH_PROBE_MODE=mode)
    env.pop("X2W_LOW_ADDRESSES", None)
    if code is not None:
        env["X2W_LAUNCH_PROBE_CODE"] = str(code)
    process = subprocess.Popen([sys.executable, "-B", str(TOOLS / "debug_launch.py"), str(probe), str(work), str(logdir),
                                "--timeout", str(timeout), "--console", "0"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return process, logdir


def wait_for(path, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists() and path.read_text(encoding="ascii", errors="replace").strip():
            return
        time.sleep(0.2)
    raise AssertionError(f"{path.name} never appeared")


def finish(process, seconds=60):
    try:
        output, _ = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        raise AssertionError("the hosted run did not end")
    return output


def cause(logdir):
    result = subprocess.run([sys.executable, "-B", str(TOOLS / "tester_pack.py"), "end-cause", str(logdir)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout.strip()


def main(probe):
    probe = Path(probe)
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        got = {}

        process, logdir = host(probe, work, "timeout", "sleep", 4)
        output = finish(process)
        got["timeout"] = (cause(logdir), "timeout after")

        process, logdir = host(probe, work, "stopfile", "sleep", 60)
        wait_for(logdir / "title.pid")
        time.sleep(1.5)
        (logdir / "stop").write_text("stop", encoding="ascii")
        started = time.monotonic()
        output = finish(process)
        assert time.monotonic() - started < 20, output[-500:]
        got["stop file"] = (cause(logdir), "stopped by request")

        process, logdir = host(probe, work, "exited", "report", 30, code=0)
        output = finish(process)
        got["exited"] = (cause(logdir), "title exited by itself")

        process, logdir = host(probe, work, "killed", "sleep", 60)
        wait_for(logdir / "title.pid")
        time.sleep(1.5)
        pid = int((logdir / "title.pid").read_text(encoding="ascii"))
        killed = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
        assert killed.returncode == 0, killed.stdout + killed.stderr
        output = finish(process)
        got["killed"] = (cause(logdir), "title killed externally")

        process, logdir = host(probe, work, "crashed", "crash", 30)
        output = finish(process)
        got["crashed"] = (cause(logdir), "title crashed (exit code 0xC0000005")

        process, logdir = host(probe, work, "watchdog", "sleep", 60)
        watchdog = subprocess.Popen([sys.executable, "-B", str(TOOLS / "stall_watchdog.py"), str(logdir), "--parent", str(process.pid),
                                     "--seconds", "60", "--stall-seconds", "3", "--min-seconds", "0", "--memory-percent", "0"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = finish(process)
        watchdog.communicate(timeout=30)
        got["watchdog"] = (cause(logdir), "watchdog at")

        for name, (text, needle) in got.items():
            assert needle in text, (name, needle, text)
            assert "last present" in text or "no present" in text or "heartbeat" in text, (name, "no last milestone", text)
            print(f"  {name}: {text}")
    print("startup_end_cause_debug_launch: PASS (6 ways a run ends, each named)")
    return 0


if __name__ == "__main__":
    if sys.platform != "win32":
        print("startup_end_cause_debug_launch_check: Windows only, skipped")
        sys.exit(77)
    sys.exit(main(sys.argv[1]))
