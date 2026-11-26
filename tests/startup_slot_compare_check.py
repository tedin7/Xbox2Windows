#!/usr/bin/env python3
"""slot_compare.py on synthetic data: sums slow-interval lines per slot and divides by replay calls."""
import json
import os
import subprocess
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "startup", "slot_compare.py")

STDERR = "\n".join([
    "x2w-d3d12 time inside the stand-in since the last report (ms/calls): list 667 ms/6678",
    "x2w-d3d12 command-list slots over 20 ms: slot9 34 ms/867 slot12 80 ms/361",
    "x2w-d3d12 command-list slots over 20 ms: slot12 1,000 ms/639 slot139 431 ms/242",
    "unrelated line slot9 999 ms/1",
])
SUMMARY = {"slot_times": [
    {"kind": 4, "slot": 12, "calls": 500, "total_us": 1000.0},
    {"kind": 4, "slot": 9, "calls": 0, "total_us": 0.0},
    {"kind": 0, "slot": 139, "calls": 10, "total_us": 10.0},
    {"kind": 4, "slot": 7, "calls": 4, "total_us": 8.0},
]}


def main():
    with tempfile.TemporaryDirectory() as d:
        err, js = os.path.join(d, "stderr.txt"), os.path.join(d, "replay_summary.json")
        with open(err, "w") as f:
            f.write(STDERR)
        with open(js, "w") as f:
            json.dump(SUMMARY, f)
        out = subprocess.run([sys.executable, "-I", TOOL, err, js], check=True, capture_output=True, text=True).stdout
        rows = {int(line.split()[0]): line.split() for line in out.splitlines()[1:]}
        # slot 12: title 1080 ms / 1000 calls = 1080 us, replay 2 us -> ratio 540
        assert rows[12][1:] == ["1080", "1000", "1080.0", "500", "2.00", "540.0"], rows[12]
        # slot 139: title only (the replay entry is another kind); slot 9: replay saw no calls
        assert rows[139][1:] == ["431", "242", "1781.0", "0", "-", "-"], rows[139]
        assert rows[9][1:] == ["34", "867", "39.2", "0", "-", "-"], rows[9]
        # slot 7: replay only
        assert rows[7][1:] == ["0", "0", "-", "4", "2.00", "-"], rows[7]
        assert list(rows)[0] == 12, list(rows)
        bare = os.path.join(d, "bare.json")
        with open(bare, "w") as f:
            json.dump({}, f)
        assert subprocess.run([sys.executable, "-I", TOOL, err, bare], capture_output=True).returncode == 2


if __name__ == "__main__":
    main()
