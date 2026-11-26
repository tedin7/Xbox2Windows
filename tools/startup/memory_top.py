#!/usr/bin/env python3
"""Who holds the PC's committed memory during a tester run: memory.txt in the run folder (tester_run.ps1).

  memory_top.py <memory.txt> [--every S] [--top N] [--parent PID]   append one line every S seconds (default 30) until
                                                                     the process PID (tester_run) has ended
  memory_top.py <memory.txt> --once <label> [--top N]                 append one line now, labelled <label>
  memory_top.py <memory.txt> --once <label> --left                    the same, and print the system commit and the
                                                                     unattributed commit left compared with the file's
                                                                     first line (tester_run: 10 s after the title exits)

One line per sample: the system commit (used / limit), then the N processes with the most committed private memory,
processes of the same name summed (`python.exe x16 6.20` = 16 processes, 6.20 GiB together), then the rest, then the
commit no process holds as private memory (`unattributed`: shared sections, drivers, the kernel). Every process is read
from one system snapshot (NtQuerySystemInformation), which needs no handle to it, so services and protected processes
count too; only if the snapshot fails are processes opened one by one, and those that cannot be opened are named in the
label (`unattributed 22.34 (150 processes unreadable)`). Names and GiB only: no paths, no command lines, no window
titles. Windows only; elsewhere it writes nothing and exits 0.
"""
import argparse
import os
import struct
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
    unread = " (%d processes unreadable)" % unreadable if unreadable else ""
    if commit:
        outside = max(0, commit[0] - sum(size for _, size in processes))
        parts.append("unattributed %.2f%s" % (outside / GIB, unread))
    elif unread:
        parts.append(unread.strip(" ()"))
    return " | ".join(parts)


# SYSTEM_PROCESS_INFORMATION on 64-bit Windows: the offsets of the fields read here.
SPI_NEXT, SPI_NAME_LENGTH, SPI_NAME_BUFFER, SPI_PID, SPI_PAGEFILE = 0, 56, 64, 80, 184
SPI_SIZE = 256  # the fixed part of a record (the threads follow it)


def parse_system_processes(buffer, base):
    """[(pid, name, committed private bytes)] from a SystemProcessInformation snapshot. `buffer` holds the records,
    `base` is the address the snapshot was written at (each image name is a pointer into the buffer). PagefileUsage is
    the process's committed private memory, the PrivateUsage psapi reports. Malformed input ends the walk."""
    out, offset = [], 0
    while 0 <= offset <= len(buffer) - SPI_SIZE:
        following, = struct.unpack_from("<I", buffer, offset + SPI_NEXT)
        length, = struct.unpack_from("<H", buffer, offset + SPI_NAME_LENGTH)
        pointer, = struct.unpack_from("<Q", buffer, offset + SPI_NAME_BUFFER)
        pid, = struct.unpack_from("<Q", buffer, offset + SPI_PID)
        pagefile, = struct.unpack_from("<Q", buffer, offset + SPI_PAGEFILE)
        start = pointer - base
        name = ""
        if pointer and length and 0 <= start <= len(buffer) - length:
            name = bytes(buffer[start:start + length]).decode("utf-16-le", "replace")
        if pid:  # pid 0 is the idle process
            out.append((pid, name or "System", pagefile))
        if not following:
            break
        offset += following
    return out


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
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtQuerySystemInformation.restype = ctypes.c_long
    ntdll.NtQuerySystemInformation.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    system_process_information, length_mismatch = 5, -0x3ffffffc  # STATUS_INFO_LENGTH_MISMATCH

    def snapshot():
        """Every process from one SystemProcessInformation snapshot, or None if the call fails."""
        size = 1 << 20
        for _ in range(6):
            buffer = ctypes.create_string_buffer(size)
            needed = ctypes.c_ulong()
            status = ntdll.NtQuerySystemInformation(system_process_information, buffer, size, ctypes.byref(needed))
            if status == length_mismatch:
                size = max(size * 2, needed.value + (64 << 10))
                continue
            if status < 0:
                return None
            return [(name, pagefile) for _pid, name, pagefile in
                    parse_system_processes(memoryview(buffer)[:needed.value or size], ctypes.addressof(buffer))]
        return None

    def read():
        perf = Performance()
        perf.cb = ctypes.sizeof(perf)
        commit = ((perf.commit_total * perf.page_size, perf.commit_limit * perf.page_size)
                  if psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb) else None)
        everyone = snapshot()
        if everyone:
            return commit, everyone, 0
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

    read.snapshot = snapshot
    return read, alive


def line_totals(line):
    """(commit used GiB, unattributed GiB or None) of a memory.txt line, or None when it carries no commit."""
    import re
    commit = re.search(r"commit (\d+(?:\.\d+)?)/", line)
    if not commit:
        return None
    outside = re.search(r"(?:unattributed|not in any process) (\d+(?:\.\d+)?)", line)
    return float(commit.group(1)), float(outside.group(1)) if outside else None


def left_text(first, last):
    """'memory left after exit: +X GB (unattributed +Y GB)': the last line's totals against the first line's (the run's
    start), or None when either has no commit."""
    start, now = line_totals(first or ""), line_totals(last or "")
    if not start or not now:
        return None
    text = "memory left after exit: %+.2f GB" % (now[0] - start[0])
    if start[1] is not None and now[1] is not None:
        text += " (unattributed %+.2f GB)" % (now[1] - start[1])
    return text


def append_once(path, label, read, top, left=False):
    """Append one line labelled `label` to memory.txt; with `left`, return left_text against the file's first line."""
    commit, processes, unreadable = read()
    line = format_line(label, commit, processes, top, unreadable)
    first = None
    if left and os.path.isfile(path):
        with open(path, encoding="utf-8") as existing:
            first = existing.readline()
    with open(path, "a", encoding="utf-8") as out:
        out.write(line + "\n")
    return (left_text(first, line) or "memory left after exit: n/a") if left else None


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--every", type=float, default=DEFAULT_EVERY)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--parent", type=int, default=0, help="stop when this process has ended (0: run until killed)")
    parser.add_argument("--once", default=None, help="append one line with this label and exit")
    parser.add_argument("--left", action="store_true", help="with --once: print what is left against the first line")
    args = parser.parse_args(argv)
    if args.every < 1 or args.top < 1:
        parser.error("--every must be at least 1 s and --top at least 1")
    if os.name != "nt":
        print("memory_top: Windows only, nothing written")
        return 0
    read, alive = windows_reader()
    if args.once is not None:
        left = append_once(args.out, args.once, read, args.top, args.left)
        if left:
            print(left)
        return 0
    sample_loop(args.out, read, args.every, args.top, (lambda: alive(args.parent)) if args.parent else (lambda: True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
