#!/usr/bin/env python3
"""Check tools/startup/memory_top.py: the memory.txt lines tester_run.ps1 packs (who holds the PC's committed memory).

Pure Python with a fake process list: grouping by name, the top N, the rest, the commit outside every process, and the
sampling loop's timing and end. The Windows reader itself runs in startup_tester_run_ps1 on a Windows host.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import memory_top as M  # noqa: E402

GIB = 1024 ** 3


def check_lines():
    processes = [("game.exe", 11 * GIB), ("python.exe", GIB), ("Python.exe", GIB // 2), ("python.exe", GIB // 2),
                 ("browser.exe", 3 * GIB), ("a.exe", GIB // 4), ("b.exe", GIB // 8)]
    rows, rest = M.group_top(processes, 3)
    assert rows == [("game.exe", 1, 11 * GIB), ("browser.exe", 1, 3 * GIB), ("python.exe", 3, 2 * GIB)], rows
    assert rest == (2, GIB // 4 + GIB // 8), rest
    line = M.format_line("   30 s", (50 * GIB, 51 * GIB), processes, 3, unreadable=4)
    assert line == ("   30 s commit 50.0/51.0 GiB | game.exe 11.00 | browser.exe 3.00 | python.exe x3 2.00 | "
                    "2 other processes 0.38 | not in any process 33.62 | 4 processes not readable"), line
    # Everything shown: no "other" part; commit unknown: no "not in any process" part.
    assert M.format_line("end", None, [("x.exe", GIB)], 8) == "end commit n/a | x.exe 1.00"
    # Ties sort by name so two runs of the same state print the same line.
    assert [r[0] for r in M.group_top([("b.exe", GIB), ("a.exe", GIB)], 2)[0]] == ["a.exe", "b.exe"]


def check_loop():
    now = [0.0]
    reads = []

    def read():
        reads.append(now[0])
        return (10 * GIB, 20 * GIB), [("game.exe", GIB * len(reads))], 0

    def sleep(seconds):
        now[0] += max(seconds, 0.25)

    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "memory.txt")
        M.sample_loop(out, read, 30.0, 8, alive=lambda: now[0] < 95.0, clock=lambda: now[0], sleep=sleep)
        lines = Path(out).read_text(encoding="utf-8").splitlines()
    assert reads == [0.0, 30.0, 60.0, 90.0], reads  # one sample now, then every 30 s until tester_run is gone
    assert lines[0].startswith("    0 s commit 10.0/20.0 GiB | game.exe 1.00") and lines[3].startswith("   90 s"), lines
    assert len(lines) == 4, lines


def main():
    check_lines()
    check_loop()
    print("startup_memory_top_check: top processes by commit grouped by name, rest and outside-process commit, one line "
          "every 30 s until tester_run ends")
    return 0


if __name__ == "__main__":
    sys.exit(main())
