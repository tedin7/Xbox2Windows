"""Check how tools/startup/debug_launch.py writes debug events to events.txt (needs no visible window).

usage: startup_debug_events_check.py <startup_window_probe.exe>

- A 400+ character Unicode debug message with CRLF inside is ONE record: line breaks escaped (\\r\\n), nothing cut, the last
  word present.
- A __debugbreak() in the title's main image is logged with the thread and how it was answered: with --continue-asserts it
  is "continued (soft assert, 1 of 5 skipped)" and the run goes on to its timeout; with --continue-asserts 0 it is "stopped"
  and the run ends there.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEBUG = HERE.parent / "tools/startup/debug_launch.py"


def run(probe, work, name, env_extra, *options, timeout="4"):
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith(("X2W_", "STARTUP_WINDOW_PROBE"))}
    env.update(env_extra)
    command = [sys.executable, "-B", str(DEBUG), str(probe), str(work), str(logdir), "--console", "0", "--timeout", timeout, *options]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    events = (logdir / "events.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    return result, events


def main(probe):
    probe = Path(probe).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _, events = run(probe, work, "message", {"STARTUP_WINDOW_PROBE_MESSAGE": "1"})
        records = [line for line in events if "debug-output" in line and "probe-worker-message" not in line]
        assert len(records) == 1, records
        record = records[0]
        # thread id of the title's main thread and the first non-system caller on its stack (the probe itself)
        match = re.match(r" *\d+\.\d{3} debug-output thread (\d+) from (\S+): probe-msg-start ", record)
        assert match, record
        assert match.group(2) == "-" or "startup_window_probe" in match.group(2).lower(), record
        threads = {m.group(1) for m in (re.match(r" *\d+\.\d{3} thread (\d+) created", line) for line in events) if m}
        worker = [line for line in events if "probe-worker-message" in line]
        assert len(worker) == 1, worker
        found = re.match(r" *\d+\.\d{3} debug-output thread (\d+) from (\S+): probe-worker-message", worker[0])
        assert found and found.group(1) in threads and found.group(1) != match.group(1), (worker, threads, record[:60])
        assert "\\r\\nsecond line\\r\\n" in record and record.endswith("\\r\\nprobe-msg-suffix-7f3a"), record[-80:]
        assert len(record) > 400, len(record)
        assert not any(line.startswith("second line") or line.startswith("probe-msg-suffix") for line in events), "message was split"

        result, events = run(probe, work, "skip", {"STARTUP_WINDOW_PROBE_BREAK": "1"}, "--continue-asserts", "5")
        skipped = [line for line in events if "exception 0x80000003" in line and "soft assert" in line]
        assert len(skipped) == 1 and re.search(r"thread \d+: continued \(soft assert, 1 of 5 skipped\)", skipped[0]), (skipped, result.stdout)
        assert "timeout after" in result.stdout, result.stdout

        result, events = run(probe, work, "stop", {"STARTUP_WINDOW_PROBE_BREAK": "1"}, "--continue-asserts", "0", timeout="20")
        stopped = [line for line in events if "exception 0x80000003" in line and "stopped" in line]
        assert len(stopped) == 1 and re.search(r"thread \d+: stopped \(not continued", stopped[0]), (stopped, result.stdout)
        assert "stopped at breakpoint" in result.stdout, result.stdout
    print("startup_debug_events: PASS (a multi-line message is one whole record; a breakpoint says how it was answered)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
