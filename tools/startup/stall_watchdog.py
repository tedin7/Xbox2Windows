#!/usr/bin/env python3
"""Stall watchdog of the tester run (tester_run.ps1): the run ends by itself, cleanly, when it is clearly stuck.

  stall_watchdog.py <logdir> --parent PID --seconds S [--stall-seconds 90] [--min-seconds 150] [--memory-percent 95]

Runs beside the title (started by tester_run.ps1 like memory_top.py) until the process PID has ended. When a rule fires it
prints one line `tester_run: STALL STOP at N s: <reason>`, writes <logdir>/stall_stop.json ({"at_seconds", "reason"}) and creates
<logdir>/stop, the file debug_launch.py watches: the run then ends exactly as at the normal timeout, so the zip is packed as
always. tester_run.ps1 copies the reason into manifest.json as stop_reason.

Rules (decide() below, pure and tested on recorded samples):
1. memory: the committed memory is at or above --memory-percent of the commit limit in 2 samples in a row;
2. frozen: for --stall-seconds in a row none of these moved: frames presented, storage KiB read, translator output, window picture.
   A loading screen that animates or reads the disk keeps the run alive;
3. never before --min-seconds of run time, never at or after --seconds (the normal end). 0 for --stall-seconds / --memory-percent
   turns that rule off.
Times are run seconds: 0 is the moment debug_launch.py created events.txt. Windows only for the memory reading; elsewhere that
rule never fires.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

DEFAULT_STALL_SECONDS = 90
DEFAULT_MIN_SECONDS = 150
DEFAULT_MEMORY_PERCENT = 95
MEMORY_SAMPLES = 2          # consecutive samples at or above the percent
SAMPLE_SECONDS = 5.0
TAIL_BYTES = 262144         # how much of the end of each title log is read per sample
SIGNALS = ("present", "storage", "translator", "picture")
RE_FRAMES = re.compile(r"x2w-d3d12 heartbeat: (\d{1,12}) frames presented")
RE_STORAGE = re.compile(r"x2w-dstorage progress \d{1,10} s: .*?total read (\d{1,15}) KiB")


class Watch:
    """The state of the decision: when each signal last moved, how many memory samples in a row were too high."""

    def __init__(self, stall_seconds=DEFAULT_STALL_SECONDS, min_seconds=DEFAULT_MIN_SECONDS,
                 memory_percent=DEFAULT_MEMORY_PERCENT, max_seconds=0):
        self.stall_seconds, self.min_seconds = stall_seconds, min_seconds
        self.memory_percent, self.max_seconds = memory_percent, max_seconds
        self.last = {}        # signal -> last value seen
        self.moved_at = None  # run second of the latest change of any signal (the first value seen counts as a change)
        self.high = 0

    def decide(self, t, sample):
        """sample: dict with commit_percent (float or None) and the SIGNALS (value or None = unknown). Returns the reason
        text when the run must end now, else None. Call once per sample, with t rising."""
        moved = False
        for name in SIGNALS:
            value = sample.get(name)
            if value is None:
                continue
            if name not in self.last or value != self.last[name]:
                moved = True
            self.last[name] = value
        if moved:
            self.moved_at = t
        percent = sample.get("commit_percent")
        self.high = self.high + 1 if (self.memory_percent and percent is not None and percent >= self.memory_percent) else 0
        if t < self.min_seconds or (self.max_seconds and t >= self.max_seconds):
            return None
        if self.high >= MEMORY_SAMPLES:
            return "memory: committed memory is %.0f%% of the commit limit (limit %d%%), in %d samples in a row" % (
                percent, self.memory_percent, self.high)
        if self.stall_seconds and self.moved_at is not None and t - self.moved_at >= self.stall_seconds:
            return "frozen: no change in frames presented, storage read, translator output or window picture for %d s" % (
                t - self.moved_at)
        return None


def read_tail(path):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def last_number(regex, texts):
    best = None
    for text in texts:
        for found in regex.finditer(text):
            value = int(found.group(1))
            best = value if best is None else max(best, value)
    return best


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def newest_picture_hash(logdir):
    try:
        names = sorted(n for n in os.listdir(logdir) if re.fullmatch(r"window(_\d{3}s)?\.png", n))
        newest = max((os.path.join(logdir, n) for n in names), key=os.path.getmtime, default=None)
        if newest is None:
            return None
        with open(newest, "rb") as handle:
            return hashlib.sha1(handle.read()).hexdigest()
    except OSError:
        return None


def commit_percent():
    """Committed memory as a percent of the system commit limit (GlobalMemoryStatusEx), or None."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes as wt

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wt.DWORD), ("load", wt.DWORD), ("phys", ctypes.c_uint64), ("phys_free", ctypes.c_uint64),
                    ("page", ctypes.c_uint64), ("page_free", ctypes.c_uint64), ("virt", ctypes.c_uint64),
                    ("virt_free", ctypes.c_uint64), ("ext", ctypes.c_uint64)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) or not status.page:
        return None
    return 100.0 * (status.page - status.page_free) / status.page


def take_sample(logdir):
    texts = [read_tail(os.path.join(logdir, name)) for name in ("stdout.txt", "stderr.txt")]
    translator = file_size(os.path.join(logdir, "translate.txt")) + file_size(os.path.join(logdir, "translate_err.txt"))
    return {"commit_percent": commit_percent(), "present": last_number(RE_FRAMES, texts),
            "storage": last_number(RE_STORAGE, texts), "translator": translator, "picture": newest_picture_hash(logdir)}


def parent_alive(pid):
    if pid <= 0:
        return True
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        alive = ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102
        ctypes.windll.kernel32.CloseHandle(handle)
        return alive
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--parent", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--stall-seconds", type=int, default=DEFAULT_STALL_SECONDS)
    parser.add_argument("--min-seconds", type=int, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--memory-percent", type=int, default=DEFAULT_MEMORY_PERCENT)
    parser.add_argument("--every", type=float, default=SAMPLE_SECONDS)
    args = parser.parse_args(argv)
    if min(args.stall_seconds, args.min_seconds, args.memory_percent, args.seconds) < 0 or args.memory_percent > 100:
        print("stall_watchdog: expected whole numbers of 0 or more (memory percent at most 100)")
        return 2
    watch = Watch(args.stall_seconds, args.min_seconds, args.memory_percent, args.seconds)
    events = os.path.join(args.logdir, "events.txt")
    started = None
    while parent_alive(args.parent):
        if started is None and os.path.exists(events):
            started = time.time()
        if started is not None:
            t = time.time() - started
            reason = watch.decide(t, take_sample(args.logdir))
            if reason:
                with open(os.path.join(args.logdir, "stall_stop.json"), "w", encoding="utf-8") as out:
                    json.dump({"at_seconds": round(t), "reason": reason}, out)
                with open(os.path.join(args.logdir, "stop"), "w", encoding="utf-8") as out:
                    out.write("stall watchdog\n")
                print("tester_run: STALL STOP at %d s: %s" % (t, reason), flush=True)
                return 0
        time.sleep(args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
