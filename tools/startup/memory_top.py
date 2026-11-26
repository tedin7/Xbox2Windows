#!/usr/bin/env python3
"""Who holds the PC's committed memory during a tester run: memory.txt in the run folder (tester_run.ps1).

  memory_top.py <memory.txt> [--every S] [--top N] [--parent PID]   append one line every S seconds (default 30) until
                                                                     the process PID (tester_run) has ended
  memory_top.py <memory.txt> --once <label> [--top N]                 append one line now, labelled <label>

One line per sample: the system commit (used / limit), then the N processes with the most committed private memory,
processes of the same name summed (`python.exe x16 6.20` = 16 processes, 6.20 GiB together), then the rest, then the
commit no process holds as private memory (shared sections, drivers, the kernel). Names and GiB only: no paths, no
command lines, no window titles. Windows only; elsewhere it writes nothing and exits 0.
"""
import argparse
import os
import sys
import time

GIB = 1024 ** 3
DEFAULT_EVERY = 30.0
DEFAULT_TOP = 8
POLL_SECONDS = 1.0  # how often the loop checks that tester_run is still there


def group_top(processes, top):
    """processes: (name, private bytes) pairs. Returns (rows, rest): rows = up to `top` (name, count, bytes) groups with
    the most bytes, processes of one name (case-insensitive) summed; rest = (processes, bytes) of the other groups."""
    groups = {}
    for name, size in processes:
        key = name.lower()
        shown, count, total = groups.get(key, (name, 0, 0))
        groups[key] = (shown, count + 1, total + size)
    ranked = sorted(groups.values(), key=lambda row: (-row[2], row[0].lower()))
    rows, others = ranked[:top], ranked[top:]
    return rows, (sum(row[1] for row in others), sum(row[2] for row in others))


def format_line(label, commit, processes, top, unreadable=0):
    """One memory.txt line. commit = (used bytes, limit bytes) or None."""
    rows, (rest_count, rest_bytes) = group_top(processes, top)
    parts = ["%s commit %.1f/%.1f GiB" % (label, commit[0] / GIB, commit[1] / GIB) if commit else "%s commit n/a" % label]
    parts += ["%s%s %.2f" % (name, " x%d" % count if count > 1 else "", size / GIB) for name, count, size in rows]
    if rest_count:
        parts.append("%d other processes %.2f" % (rest_count, rest_bytes / GIB))
    if commit:
        outside = max(0, commit[0] - sum(size for _, size in processes))
        parts.append("not in any process %.2f" % (outside / GIB))
    if unreadable:
        parts.append("%d processes not readable" % unreadable)
    return " | ".join(parts)


def sample_loop(out_path, read, every, top, alive, clock=time.monotonic, sleep=time.sleep):
    """Append a line now and then every `every` seconds while alive() holds. read() -> (commit, processes, unreadable)."""
    start = clock()
    next_at = start
    with open(out_path, "a", encoding="utf-8") as out:
        while True:
            if clock() >= next_at:
                commit, processes, unreadable = read()
                out.write(format_line("%5.0f s" % (clock() - start), commit, processes, top, unreadable) + "\n")
                out.flush()
                next_at += every
            if not alive():
                return
            sleep(min(POLL_SECONDS, max(0.0, next_at - clock())))


def windows_reader():
    """read() for Windows: (commit, [(name, private bytes)], unreadable count) through psapi."""
    import ctypes
    from ctypes import wintypes as wt

    k32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    query_limited = 0x1000

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
        _fields_ = [("cb", wt.DWORD), ("faults", wt.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]

    class Performance(ctypes.Structure):  # PERFORMANCE_INFORMATION
        _fields_ = [("cb", wt.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "commit_total", "commit_limit", "commit_peak", "phys_total", "phys_avail", "cache", "kernel_total",
            "kernel_paged", "kernel_nonpaged", "page_size")] + [(n, wt.DWORD) for n in ("handles", "processes", "threads")]

    k32.OpenProcess.restype = wt.HANDLE

    def read():
        perf = Performance()
        perf.cb = ctypes.sizeof(perf)
        commit = ((perf.commit_total * perf.page_size, perf.commit_limit * perf.page_size)
                  if psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb) else None)
        pids = (wt.DWORD * 8192)()
        used = wt.DWORD()
        if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(used)):
            return commit, [], 0
        processes, unreadable = [], 0
        for pid in pids[:used.value // ctypes.sizeof(wt.DWORD)]:
            handle = k32.OpenProcess(query_limited, False, pid) if pid else None
            if not handle:
                unreadable += 1 if pid else 0
                continue
            try:
                name = ctypes.create_unicode_buffer(1024)
                size = wt.DWORD(len(name))
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                if not (k32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size))
                        and psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)):
                    unreadable += 1
                    continue
                processes.append((os.path.basename(name.value), counters.private))
            finally:
                k32.CloseHandle(handle)
        return commit, processes, unreadable

    def alive(pid):
        synchronize = 0x00100000
        handle = k32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return k32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            k32.CloseHandle(handle)

    return read, alive


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--every", type=float, default=DEFAULT_EVERY)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--parent", type=int, default=0, help="stop when this process has ended (0: run until killed)")
    parser.add_argument("--once", default=None, help="append one line with this label and exit")
    args = parser.parse_args(argv)
    if args.every < 1 or args.top < 1:
        parser.error("--every must be at least 1 s and --top at least 1")
    if os.name != "nt":
        print("memory_top: Windows only, nothing written")
        return 0
    read, alive = windows_reader()
    if args.once is not None:
        commit, processes, unreadable = read()
        with open(args.out, "a", encoding="utf-8") as out:
            out.write(format_line(args.once, commit, processes, args.top, unreadable) + "\n")
        return 0
    sample_loop(args.out, read, args.every, args.top, (lambda: alive(args.parent)) if args.parent else (lambda: True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
