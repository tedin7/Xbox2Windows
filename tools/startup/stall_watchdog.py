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
3. deadlock: no frame presented for --stall-seconds and the title used under 1 s of CPU in that time (a thread deadlock, not
   loading): the watchdog first asks debug_launch.py for a threads dump (file `dump_request`, answered with threads_stall.txt),
   then stops the run with the reason "deadlock: ...";
4. never before --min-seconds of run time, never at or after --seconds (the normal end). 0 for --stall-seconds / --memory-percent
   turns that rule off.
Hard end: Stop-Process does not close a title that runs under the debugger, and a stop that debug_launch.py cannot finish
(its end-of-run thread dump waits on a stuck thread) would leave the run open. So after the stop file, or after --seconds
(the normal end), the watchdog waits --hard-grace seconds for finished.txt (debug_launch.py writes it last); if it is not there
it ends the debugger's process tree (taskkill /T /F on the debug_launch.py PID found by its command line, then the title PID from
title.pid), checks both PIDs are gone and writes hard_end.json, so tester_run.ps1 goes on to pack the zip.
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
from pathlib import Path

DEFAULT_STALL_SECONDS = 90
DEFAULT_MIN_SECONDS = 150
DEFAULT_MEMORY_PERCENT = 95
MEMORY_SAMPLES = 2          # consecutive samples at or above the percent
SAMPLE_SECONDS = 5.0
DEADLOCK_CPU_SECONDS = 1.0   # the title used less CPU than this over the stall window while presenting nothing
HARD_GRACE_SECONDS = 60     # wait for the debugger to finish by itself before ending its process tree
DUMP_WAIT_SECONDS = 15      # how long the threads dump may take before the stop goes ahead
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
        self.present_value, self.present_at = None, None
        self.cpu_history = []   # (run second, title CPU seconds)

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
        present = sample.get("present") or 0
        if present != self.present_value:
            self.present_value, self.present_at = present, t
        if sample.get("cpu") is not None:
            self.cpu_history.append((t, sample["cpu"]))
            self.cpu_history = [(at, cpu) for at, cpu in self.cpu_history if at >= t - self.stall_seconds - SAMPLE_SECONDS * 4]
        percent = sample.get("commit_percent")
        self.high = self.high + 1 if (self.memory_percent and percent is not None and percent >= self.memory_percent) else 0
        if t < self.min_seconds or (self.max_seconds and t >= self.max_seconds):
            return None
        if self.high >= MEMORY_SAMPLES:
            return "memory: committed memory is %.0f%% of the commit limit (limit %d%%), in %d samples in a row" % (
                percent, self.memory_percent, self.high)
        if self.stall_seconds and self.present_at is not None and t - self.present_at >= self.stall_seconds:
            before = [cpu for at, cpu in self.cpu_history if at <= t - self.stall_seconds]
            if before and sample.get("cpu") is not None and sample["cpu"] - before[-1] < DEADLOCK_CPU_SECONDS:
                return "deadlock: no frame presented for %d s and the title used %.1f s of CPU in that time" % (
                    t - self.present_at, sample["cpu"] - before[-1])
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


def read_pid(logdir):
    try:
        return int(Path(logdir, "title.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def process_cpu_seconds(pid):
    """User + kernel CPU time of a process in seconds (Windows GetProcessTimes), or None."""
    if os.name != "nt" or not pid:
        return None
    import ctypes
    from ctypes import wintypes as wt
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    times = [wt.FILETIME() for _ in range(4)]
    ok = ctypes.windll.kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])
    ctypes.windll.kernel32.CloseHandle(handle)
    if not ok:
        return None
    return sum((t.dwHighDateTime << 32 | t.dwLowDateTime) for t in times[2:]) / 1e7


def take_sample(logdir):
    texts = [read_tail(os.path.join(logdir, name)) for name in ("stdout.txt", "stderr.txt")]
    translator = file_size(os.path.join(logdir, "translate.txt")) + file_size(os.path.join(logdir, "translate_err.txt"))
    return {"commit_percent": commit_percent(), "present": last_number(RE_FRAMES, texts),
            "storage": last_number(RE_STORAGE, texts), "translator": translator, "picture": newest_picture_hash(logdir),
            "cpu": process_cpu_seconds(read_pid(logdir))}


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        alive = ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102
        ctypes.windll.kernel32.CloseHandle(handle)
        return alive
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"   # a zombie is already dead
    except (OSError, IndexError):
        return False


def parent_alive(pid):
    return pid <= 0 or pid_alive(pid)


def find_debugger(logdir):
    """PID of the debug_launch.py process of this run folder (found by its command line), or None."""
    if os.name == "nt":
        import subprocess
        quoted = logdir.replace("'", "''")
        command = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*debug_launch.py*' -and "
                   "$_.CommandLine -like '*%s*' } | ForEach-Object { $_.ProcessId }" % quoted)
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                                 timeout=60).stdout.split()
        except (OSError, subprocess.SubprocessError):
            return None
        return int(out[0]) if out and out[0].isdigit() else None
    for entry in os.listdir("/proc"):
        if entry.isdigit() and int(entry) != os.getpid():
            try:
                words = Path("/proc", entry, "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            text = [w.decode("utf-8", errors="replace") for w in words]
            if any(w.endswith("debug_launch.py") for w in text) and logdir in text:
                return int(entry)
    return None


def kill_tree(pid):
    """End a process and everything it started."""
    if os.name == "nt":
        import subprocess
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=60)
        return
    import signal
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                parent = int(Path("/proc", entry, "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(parent, []).append(int(entry))
    order, todo = [], [pid]
    while todo:
        order.append(todo.pop())
        todo += children.get(order[-1], [])
    for victim in order:
        try:
            os.kill(victim, signal.SIGKILL)
        except OSError:
            pass


def hard_end(logdir, run_second, say=print, wait_seconds=10.0):
    """End the debugger's process tree, then the title, and check both PIDs are gone. Writes hard_end.json; returns the note."""
    debugger, title = find_debugger(logdir), read_pid(logdir)
    if debugger:
        kill_tree(debugger)
    if title and pid_alive(title):
        kill_tree(title)
    deadline = time.time() + wait_seconds
    while time.time() < deadline and (pid_alive(debugger) or pid_alive(title)):
        time.sleep(0.2)
        for pid in (debugger, title):
            if pid_alive(pid):
                kill_tree(pid)
    gone = not pid_alive(debugger) and not pid_alive(title)
    note = "debugger %s ended with its process tree, title %s %s" % (
        debugger if debugger else "not found", title if title else "unknown", "gone" if gone else "STILL RUNNING")
    with open(os.path.join(logdir, "hard_end.json"), "w", encoding="utf-8") as out:
        json.dump({"at_seconds": round(run_second), "note": note, "title_gone": gone}, out)
    say("tester_run: HARD END at %d s: the debugger did not finish; %s" % (run_second, note))
    return note


def request_dump(logdir, say=print):
    """Ask debug_launch.py for a threads dump (threads_stall.txt) and wait for it a little."""
    target = os.path.join(logdir, "threads_stall.txt")
    with open(os.path.join(logdir, "dump_request"), "w", encoding="utf-8") as out:
        out.write("threads\n")
    deadline = time.time() + DUMP_WAIT_SECONDS
    while time.time() < deadline and not os.path.exists(target):
        time.sleep(0.2)
    say("tester_run: threads dump %s" % ("written (threads_stall.txt)" if os.path.exists(target) else "not written in time"))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--parent", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--stall-seconds", type=int, default=DEFAULT_STALL_SECONDS)
    parser.add_argument("--min-seconds", type=int, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--memory-percent", type=int, default=DEFAULT_MEMORY_PERCENT)
    parser.add_argument("--hard-grace", type=int, default=HARD_GRACE_SECONDS)
    parser.add_argument("--every", type=float, default=SAMPLE_SECONDS)
    args = parser.parse_args(argv)
    if min(args.stall_seconds, args.min_seconds, args.memory_percent, args.seconds, args.hard_grace) < 0 or args.memory_percent > 100:
        print("stall_watchdog: expected whole numbers of 0 or more (memory percent at most 100)")
        return 2
    watch = Watch(args.stall_seconds, args.min_seconds, args.memory_percent, args.seconds)
    events = os.path.join(args.logdir, "events.txt")
    finished = os.path.join(args.logdir, "finished.txt")
    started = None
    ending_at = None   # run second at which the run was told to end (stop file) or its time is up
    while parent_alive(args.parent):
        if started is None and os.path.exists(events):
            started = time.time()
        if started is not None:
            t = time.time() - started
            if os.path.exists(finished):
                return 0
            if ending_at is None:
                reason = watch.decide(t, take_sample(args.logdir))
                if reason:
                    if reason.startswith("deadlock"):
                        request_dump(args.logdir)
                    with open(os.path.join(args.logdir, "stall_stop.json"), "w", encoding="utf-8") as out:
                        json.dump({"at_seconds": round(t), "reason": reason}, out)
                    with open(os.path.join(args.logdir, "stop"), "w", encoding="utf-8") as out:
                        out.write("stall watchdog\n")
                    print("tester_run: STALL STOP at %d s: %s" % (t, reason), flush=True)
                    ending_at = t
                    if not args.hard_grace:
                        return 0
                elif args.seconds and t >= args.seconds:
                    ending_at = float(args.seconds)
            if ending_at is not None and args.hard_grace and t >= ending_at + args.hard_grace:
                hard_end(args.logdir, t)
                return 0
        time.sleep(args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
