"""Live debug console for a run: follows the run's logs and prints what the title is doing.

usage: live_view.py <rundir> [<package root>]      (started in its own console window by debug_launch.py)

It only reads the log files the debugger and the stand-in DLLs write; it does not touch the game.
File opens are shown for paths on the G: mount and under the package root, if one is given.
"""
import argparse
import os
import re
import sys
import time


def _storage_completed(line):
    """Requests the DirectStorage stand-in has finished, from a status or heartbeat line.

    New lines carry "completed=N" (raw + inflated + bcpack-blank). Older lines have "done=N" (uncompressed copies
    only) next to "inflated=N" and, in the heartbeat, "bcpack-blank=N": the same total is rebuilt from those."""
    m = re.search(r"\bcompleted=(\d+)", line)
    if m:
        return int(m.group(1))
    total = 0
    for name in ("done", "raw", "inflated", "bcpack-blank"):
        m = re.search(rf"\b{name}=(\d+)", line)
        if m:
            total += int(m.group(1))
    return total


class _LogViewer:
    """Stateful log parser and status tracker for a single live_view session."""

    def __init__(self, package=None):
        # The package root the run uses as its working directory (run.ps1 -DataDir), next to the G:\ mount.
        roots = ["G:\\"] + ([os.path.join(os.path.abspath(package), "")] if package is not None else [])
        self.game_file = re.compile(
            r"Nt(?:CreateFile|OpenFile) (?:\\\?\?\\)?((?i:"
            + "|".join(map(re.escape, roots))
            + r").+)(?= from (?:[^+]*\+)?0x[0-9a-fA-F]+ )"
        )
        self.start = time.time()
        self.counts = {"modules": 0, "files": 0, "refused": 0, "chunks": 0, "completed": 0, "gpu calls first seen": 0}
        self.last_status = 0.0
        self.state = "?"
        self.frames = "0"

    def say(self, tag, text):
        print(f"{time.time() - self.start:7.1f}s  {tag:<9} {text}", flush=True)

    def on_event(self, line):
        m = re.match(r"\s*[\d.]+ load (\S+) at", line)
        if m:
            self.counts["modules"] += 1
            self.say("module", m.group(1))
            return
        if " debug-output " in line:
            self.say("runtime", line.split(" debug-output ", 1)[1].strip()[:150])
        elif " watch " in line:
            self.say("watch", line.split(" watch ", 1)[1].strip())
        elif " exception " in line and "first-chance" in line and any(
                f" at {m}+" in line for m in ("d3d12_xs.dll", "xg_xs.dll", "GameInput.dll", "xmem.dll", "XCRDAPI.dll")):
            # A stand-in DLL probing whether an address is readable: caught and handled there (expected).
            self.counts["handled probes"] = self.counts.get("handled probes", 0) + 1
            if self.counts["handled probes"] == 1:
                self.say("probe", "stand-in read probes fault on purpose and are handled; counted in the status line from now on")
        elif " exception " in line and "C++ exception" not in line and "thread name" not in line:
            self.say("EXCEPTION", line.split(" exception ", 1)[1].strip()[:150])
        elif re.search(r" exit 0x", line):
            self.say("EXIT", line.strip())

    def on_file(self, line):
        if "DENIED (offline)" in line:
            self.counts["refused"] += 1
            target = line.split("DENIED (offline)", 1)[1].split(" from ")[0].strip()
            self.say("offline", "refused network open: " + target[:100])
            return
        if "Bink" in line:
            self.say("movie", re.sub(r" from .*", "", line.split(None, 1)[1]).strip()[:140])
            return
        if " REDIRECT " in line:
            self.say("EXPERIMENT", "forced code path: " + line.split(" REDIRECT ", 1)[1].strip()[:120])
            return
        if " HIT " in line:
            self.say("hit", line.split(" HIT ", 1)[1].strip()[:140])
            return
        m = self.game_file.search(line)
        if m:
            self.counts["files"] += 1
            result = line.rsplit("->", 1)[-1].strip()
            self.say("file", m.group(1) + ("" if result == "0x00000000" else f"   (failed {result})"))

    def on_shim(self, line):
        line = line.strip()
        if not line:
            return
        if line.startswith("x2w-trap"):
            self.say("TRAP", line[9:200] + "   <- unimplemented console function: the run stops here")
        elif "Factory::OpenFile" in line:
            m = re.search(r'path="([^"]+)"', line)
            self.say("storage", "opened " + (m.group(1) if m else line[:120]))
        elif "Queue::EnqueueStatus" in line:
            m = re.search(r"inflated=(\d+) failed=(\d+)", line)
            if m:
                self.counts["chunks"] = int(m.group(1))
                completed = _storage_completed(line)
                self.counts["completed"] = completed
                self.say("storage", f"read batch complete: {completed} requests completed so far, "
                                    f"{m.group(1)} compressed chunks decompressed, {m.group(2)} failed")
        elif "EnqueueRequest" in line or "options=" in line or "Queue::Submit" in line or "IsComplete" in line:
            return
        elif line.startswith("x2w-dstorage heartbeat "):
            m = re.search(r"\binflated=(\d+)", line)
            if m:
                self.counts["chunks"] = int(m.group(1))
                self.counts["completed"] = _storage_completed(line)
            self.say("storage", line)
        elif line.startswith("x2w-render totals:"):
            self.say("render", line)
        elif "heartbeat" in line:
            m = re.search(r"heartbeat: (\d+) frames presented in ([\d.]+) s \(([\d.]+) fps", line)
            if m:
                self.frames = m.group(1)
                self.say("frames", f"{m.group(1)} frames presented, {m.group(3)} fps")
        elif "title main state" in line:
            m = re.search(r"state (-?\d+) -> (-?\d+)", line)
            if m:
                self.state = m.group(2)
                meaning = {"0": "starting", "1": "loading", "3": "start-up flow, waiting"}.get(self.state, "")
                self.say("STATE", f"title main state {m.group(1)} -> {m.group(2)} {meaning}")
        elif line.startswith("x2w-audio"):
            self.say("audio", line[10:170])
        elif line.startswith("x2w-xmem"):
            self.say("memory", line[9:170])
        elif line.startswith("x2w-noop"):
            self.counts["gpu calls first seen"] += 1
            m = re.search(r"!(\S+)", line)
            self.say("gpu", "first use of an untranslated call: " + (m.group(1) if m else line[9:80]))
        elif "DEVICE REMOVED" in line or "failed" in line.lower() and "failed=0" not in line:
            self.say("WARNING", line[:170])
        elif line.startswith("x2w-d3d12") and ("placeholder" in line or "Create" in line or "slot" in line):
            self.say("gpu", line[10:170])
        elif line.startswith("x2w-xg"):
            self.say("texture", line[7:150])


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rundir", help="Run log directory to follow")
    parser.add_argument("package", nargs="?", default=None, help="Optional package root directory")
    return parser


def main(argv=None) -> int:
    """Follow a run's log directory until finished.txt appears and return the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not os.path.isdir(args.rundir):
        parser.exit(2, f"live_view: run directory does not exist or is not a directory: {args.rundir}\n")
    if args.package is not None and os.path.isfile(args.package):
        parser.exit(2, f"live_view: package root must be a directory, not a file: {args.package}\n")

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        viewer = _LogViewer(args.package)
        os.system("title Xbox2Windows debug log - " + os.path.basename(os.path.abspath(args.rundir)))
        sources = {"events.txt": viewer.on_event, "files.txt": viewer.on_file, "trap.log": viewer.on_shim}
        positions = {name: 0 for name in sources}
        partial = {name: b"" for name in sources}  # bytes: a character can be split between two reads
        print("Xbox2Windows local startup experiment: live log of " + os.path.abspath(args.rundir))
        print("Columns: time since this window opened, category, detail. Network refusals are shown when observed.")
        print("-" * 110, flush=True)
        finished = os.path.join(args.rundir, "finished.txt")
        ended = False
        while True:
            progressed = False
            for name, handler in sources.items():
                path = os.path.join(args.rundir, name)
                try:
                    with open(path, "rb") as handle:
                        handle.seek(positions[name])
                        chunk = handle.read()
                        positions[name] += len(chunk)
                except OSError:
                    continue
                if not chunk:
                    continue
                progressed = True
                lines = (partial[name] + chunk).split(b"\n")
                partial[name] = lines.pop()
                for raw in lines:
                    try:
                        handler(raw.decode("utf-8", errors="replace").rstrip("\r"))
                    except Exception as error:  # a viewer bug must not hide the rest of the log
                        viewer.say("viewer", f"could not format a line: {error}")
            if time.time() - viewer.last_status >= 10 and not ended:
                viewer.last_status = time.time()
                viewer.say(
                    "status",
                    f"state {viewer.state}, {viewer.frames} frames, {viewer.counts['modules']} modules, "
                    f"{viewer.counts['files']} game-file opens, {viewer.counts['chunks']} chunks decompressed, "
                    f"{viewer.counts['completed']} storage requests completed, "
                    f"{viewer.counts['refused']} network opens refused, "
                    f"{viewer.counts.get('handled probes', 0)} handled probes",
                )
            if ended and not progressed:
                break
            if os.path.exists(finished):
                ended = True
                continue
            time.sleep(0.25)
        print("-" * 110)
        try:
            with open(finished, encoding="utf-8", errors="replace") as handle:
                print(handle.read().strip())
        except OSError:
            pass
        print("\nRun ended. Full logs are in " + os.path.abspath(args.rundir) + " (events.txt, files.txt, trap.log).")
        # Closes by itself (the full logs stay in the run folder).
        for left in range(10, 0, -1):
            print(f"\rThis window closes in {left} s ", end="", flush=True)
            time.sleep(1)
    except OSError as exc:
        parser.exit(1, f"live_view: {exc}\n")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
