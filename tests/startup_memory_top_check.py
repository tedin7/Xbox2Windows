#!/usr/bin/env python3
"""Check tools/startup/memory_top.py: the memory.txt lines tester_run.ps1 packs (who holds the PC's committed memory).

Pure Python with a fake process list: grouping by name, the top N, the rest, the unattributed commit, a synthetic
SystemProcessInformation snapshot, and the sampling loop's timing and end. On Windows the real snapshot must hold this
process with the private commit psapi reports, and read() must leave no process unreadable. The Windows reader also
runs in startup_tester_run_ps1 on a Windows host.
"""
import os
import struct
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
                    "2 other processes 0.38 | unattributed 33.62 (4 processes unreadable)"), line
    # Everything read: no count in the label.
    assert M.format_line("0 s", (2 * GIB, 4 * GIB), [("x.exe", GIB)], 8) == "0 s commit 2.0/4.0 GiB | x.exe 1.00 | unattributed 1.00"
    # Everything shown: no "other" part; commit unknown: no "unattributed" part.
    assert M.format_line("end", None, [("x.exe", GIB)], 8) == "end commit n/a | x.exe 1.00"
    # Ties sort by name so two runs of the same state print the same line.
    assert [r[0] for r in M.group_top([("b.exe", GIB), ("a.exe", GIB)], 2)[0]] == ["a.exe", "b.exe"]


def record(name, pid, pagefile, following, base, name_at):
    """One SYSTEM_PROCESS_INFORMATION record (64-bit layout) whose image name lives at buffer offset name_at."""
    raw = bytearray(M.SPI_SIZE)
    struct.pack_into("<I", raw, M.SPI_NEXT, following)
    encoded = name.encode("utf-16-le")
    struct.pack_into("<HH", raw, M.SPI_NAME_LENGTH, len(encoded), len(encoded) + 2)
    struct.pack_into("<Q", raw, M.SPI_NAME_BUFFER, base + name_at if name else 0)
    struct.pack_into("<Q", raw, M.SPI_PID, pid)
    struct.pack_into("<Q", raw, M.SPI_PAGEFILE, pagefile)
    return raw, encoded


def check_snapshot():
    base, step = 0x7FF000000000, M.SPI_SIZE + 64  # each record followed by its name
    rows = [("", 0, 0), ("System", 4, 1 << 20), ("svchost.exe", 1234, 3 * GIB), ("game.exe", 5678, 11 * GIB)]
    buffer = bytearray()
    for i, (name, pid, pagefile) in enumerate(rows):
        raw, encoded = record(name, pid, pagefile, step if i < len(rows) - 1 else 0, base, i * step + M.SPI_SIZE)
        buffer += raw + encoded.ljust(64, bytes(1))
    got = M.parse_system_processes(memoryview(bytes(buffer)), base)
    assert got == [(4, "System", 1 << 20), (1234, "svchost.exe", 3 * GIB), (5678, "game.exe", 11 * GIB)], got
    # A name pointer outside the snapshot gives no name, and a truncated record ends the walk.
    raw, _ = record("x.exe", 9, 7, 0, base, 1 << 30)
    assert M.parse_system_processes(memoryview(bytes(raw)), base) == [(9, "System", 7)]
    assert M.parse_system_processes(memoryview(bytes(buffer[:100])), base) == []
    if os.name == "nt":  # the real snapshot: this process, with the private commit psapi reports for it
        import ctypes
        from ctypes import wintypes as wt
        read, _alive = M.windows_reader()
        everyone = read.snapshot()
        assert everyone and len(everyone) > 10, everyone

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
            _fields_ = [("cb", wt.DWORD), ("faults", wt.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
                "private")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.WinDLL("psapi").GetProcessMemoryInfo(ctypes.WinDLL("kernel32").GetCurrentProcess(), ctypes.byref(counters),
                                                     counters.cb)
        mine = [size for name, size in everyone if name.lower().startswith("python")]
        assert any(abs(size - counters.private) < 64 << 20 for size in mine), (counters.private, mine)
        commit, processes, unreadable = read()
        assert unreadable == 0 and any(name == "System" for name, _ in processes), (unreadable, processes[:5])
        print("snapshot: %d processes, this one %.0f MiB (psapi %.0f MiB)" % (
            len(processes), min(mine, key=lambda s: abs(s - counters.private)) / 1048576.0, counters.private / 1048576.0))


def check_left():
    first = "    0 s commit 30.3/48.9 GiB | claude.exe x3 2.09 | unattributed 11.19"
    after = "10 s after exit commit 31.1/48.9 GiB | claude.exe x3 2.10 | unattributed 11.93 (3 processes unreadable)"
    assert M.left_text(first, after) == "memory left after exit: +0.80 GB (unattributed +0.74 GB)", M.left_text(first, after)
    # Lines of runs before the label change still compare; a line without commit gives nothing.
    old = "    0 s commit 30.3/48.9 GiB | not in any process 11.19"
    assert M.left_text(old, after) == "memory left after exit: +0.80 GB (unattributed +0.74 GB)"
    assert M.left_text("end commit n/a | x.exe 1.00", after) is None
    assert M.left_text(first, "end commit 29.0/48.9 GiB") == "memory left after exit: -1.30 GB"
    # The --once --left form appends the line and returns the comparison (a fake reader, any host).
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "memory.txt")
        Path(out).write_text(first + "\n", encoding="utf-8")

        def read():
            return (int(31.1 * GIB), int(48.9 * GIB)), [("x.exe", int(19.17 * GIB))], 0
        left = M.append_once(out, "10 s after exit", read, 8, left=True)
        lines = Path(out).read_text(encoding="utf-8").splitlines()
        assert M.append_once(out + ".2", "end", read, 8) is None  # without --left nothing is compared
    assert len(lines) == 2 and lines[1].startswith("10 s after exit commit 31.1/48.9 GiB"), lines
    assert left == "memory left after exit: +0.80 GB (unattributed +0.74 GB)", left


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
    check_snapshot()
    check_left()
    check_loop()
    print("startup_memory_top_check: top processes by commit grouped by name, rest and outside-process commit, one line "
          "every 30 s until tester_run ends")
    return 0


if __name__ == "__main__":
    sys.exit(main())
