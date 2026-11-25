"""Run the startup debugger with X2W_HOST_CPU=1 on a source-built probe that executes SHA-extension
and RDPID instructions (tests/startup_host_cpu_probe.cpp), including RDPID sites that eight threads
reach at the same time: every thread that faulted before a site was rewritten has to continue.

On a host that has the instructions the probe just runs. On a host that lacks them the debugger
has to perform them, and the same run without the switch must stop at an illegal instruction.
Either way the printed SHA-256 of "abc" must be right; the check reports which case applied.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEBUG = REPO / "tools/startup/debug_launch.py"


def run(probe, work, name, host_cpu):
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
    if host_cpu:
        env["X2W_HOST_CPU"] = "1"
    command = [sys.executable, "-B", str(DEBUG), str(probe), str(work), str(logdir), "--timeout", "60", "--console", "0"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    printed = (logdir / "stdout.txt").read_text(encoding="utf-8", errors="replace") if (logdir / "stdout.txt").exists() else ""
    return result.stdout + result.stderr, printed


def main(probe):
    probe = Path(probe).resolve()
    expected = "sha256 abc " + hashlib.sha256(b"abc").hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        out, printed = run(probe, work, "with", True)
        assert "process exited with 0x00000000" in out, out
        assert expected in printed and "rdpid agrees with rdtscp" in printed, printed
        assert "rdpid threads 8 of 8" in printed, printed  # threads that faulted on the site together
        summary = next((line for line in out.splitlines() if "host-cpu sites rewritten" in line), "")
        plain, plain_printed = run(probe, work, "without", False)
        if summary:
            # The host lacks at least one of the instructions: without the switch the run cannot finish.
            assert "stopped at illegal instruction" in plain and expected not in plain_printed, plain
            if "sha (per hit)" in summary:
                count = int(summary.split("sha (per hit) x")[1].split(",")[0].split()[0])
                assert count == 56, summary  # 32 SHA256RNDS2, 12 SHA256MSG1 and 12 SHA256MSG2 for one block
            case = "performed by the debugger: " + summary.split("rewritten:")[1].strip()
            resumed = next((line for line in out.splitlines() if "host-cpu threads resumed" in line), "")
            if resumed:
                case += "; threads resumed after another thread's rewrite: " + resumed.rsplit(":", 1)[1].strip()
        else:
            assert "process exited with 0x00000000" in plain and expected in plain_printed, plain
            case = "host has the instructions, nothing rewritten"
    print(f"startup_host_cpu_debug: PASS ({case})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
