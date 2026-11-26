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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from startup_window_capture_check import decode_png  # noqa: E402

DEBUG = HERE.parent / "tools/startup/debug_launch.py"
RED, BLUE = (200, 30, 30), (30, 30, 200)


def run(probe, work, name, *options):
    logdir = work / name
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
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
        # Only the changes are saved: one red frame, one blue frame, the others skipped.
        assert len(frames) == 2, [p.name for p in frames]
        found = re.search(r"window captures: (\d+) saved, (\d+) unchanged skipped, first change at (\d+) s", out)
        assert found, out
        saved, unchanged, change = (int(found.group(i)) for i in (1, 2, 3))
        assert saved == 2 and unchanged >= 3 and 3 <= change <= 6, found.group(0)
        assert int(frames[-1].name[7:10]) == change, (frames[-1].name, change)
        assert pixel(logdir / "window.png") == BLUE, "the end-of-run picture must stay as before"
        assert "window capture: 160x120 -> window.png" in out, out

        control, control_dir = run(probe, work, "control", "--timeout", "4")
        assert not list(control_dir.glob("window_*s.png")), "no periodic file without --capture-every"
        assert "window captures:" not in control.stdout and (control_dir / "window.png").exists(), control.stdout

        for bad in ("0.5", "-1", "abc"):
            refused, _ = run(probe, work, "bad", "--timeout", "1", "--capture-every", bad)
            assert refused.returncode == 2 and "--capture-every" in refused.stdout, (bad, refused.returncode, refused.stdout)
    print("startup_debug_window: PASS (red then blue saved once each, unchanged frames skipped, change time reported, "
          "window.png unchanged, off by default, bad interval refused)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
