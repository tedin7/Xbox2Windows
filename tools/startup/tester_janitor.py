#!/usr/bin/env python3
"""Janitor of the tester run (tester_run.ps1): if the tester stops the script (Ctrl-C) or closes its window, the run is still
ended and the zip is still packed.

  tester_janitor.py <logdir> --parent PID --pack-json FILE --cache DIR

Started by tester_run.ps1 in a window of its own (hidden), so closing the tester's console does not end it. It waits for the
process PID (tester_run) to end. Then:
  1. if <logdir>/pack_done is missing, the run was cut short: the debugger is asked to stop (the `stop` file) and given
     20 s to write finished.txt, else its process tree and the title are ended (stall_watchdog.hard_end);
  2. the translator (gfx2hlsl.py --watch on this cache) and its workers are ended, whether or not the run was packed
     (a normal finish has already ended it; what is left is a leftover of an interrupted script);
  3. if the run was cut short: the report is packed from FILE ({"manifest", "zip", "limit_mb", "cache", "map"}, written by
     tester_run.ps1 before the run) with end_cause and interrupted=true added, the same zip a normal end would write.
Everything it does is written to <logdir>/janitor.txt. tester_run.ps1 writes pack_done after its own pack and stops the janitor
when it ends normally, so a normal run never meets it.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stall_watchdog as W  # noqa: E402
import tester_pack as T  # noqa: E402

DEBUGGER_GRACE_SECONDS = 20


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--pack-json", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--grace", type=int, default=DEBUGGER_GRACE_SECONDS)
    args = parser.parse_args(argv)
    note_path = os.path.join(args.logdir, "janitor.txt")

    def note(text):
        with open(note_path, "a", encoding="utf-8") as handle:
            handle.write("%s %s\n" % (time.strftime("%H:%M:%S"), text))

    note("janitor started, waiting for tester_run (pid %d)" % args.parent)
    while W.parent_alive(args.parent):
        time.sleep(args.poll)
    note("tester_run (pid %d) has ended" % args.parent)
    # pack_done holds the PID of the tester_run that packed: one left by an earlier run in the same folder does not count
    try:
        with open(os.path.join(args.logdir, "pack_done"), "r", encoding="utf-8", errors="replace") as handle:
            packed = handle.read().strip() == str(args.parent)
    except OSError:
        packed = False
    if not packed:
        note("tester_run ended before it packed the report (Ctrl-C, or its window was closed): ending the run and packing it")
        if W.find_debugger(args.logdir) or W.pid_alive(W.read_pid(args.logdir)):
            with open(os.path.join(args.logdir, "stop"), "w", encoding="utf-8") as out:
                out.write("tester_janitor\n")
            deadline = time.time() + args.grace
            while time.time() < deadline and not os.path.exists(os.path.join(args.logdir, "finished.txt")):
                time.sleep(0.5)
            if not os.path.exists(os.path.join(args.logdir, "finished.txt")):
                note(W.hard_end(args.logdir, 0, say=lambda text: None))
            else:
                note("the debugger finished by itself")
    if args.cache:
        for pid in W.find_processes("gfx2hlsl.py", "--watch", args.cache):
            W.kill_tree(pid)
            note("ended the translator (pid %d) and its workers" % pid)
    if packed:
        return 0
    try:
        with open(args.pack_json, "r", encoding="utf-8") as handle:
            job = json.load(handle)
        manifest = dict(job["manifest"])
        manifest["interrupted"] = True
        manifest["end_cause"] = "interrupted by the tester (Ctrl-C or the window was closed); " + T.end_cause(args.logdir)
        manifest["exit_code"] = -1
        parts = T.pack(args.logdir, manifest, job["zip"], int(float(job.get("limit_mb", T.DEFAULT_LIMIT_MB)) * 1024 * 1024),
                       cache_dir=job.get("cache") or None, map_file=job.get("map") or None)
        for path, size in parts:
            note("packed %s (%d bytes)" % (path, size))
        with open(os.path.join(args.logdir, "pack_done"), "w", encoding="utf-8") as out:
            out.write("%d\n" % args.parent)
    except Exception as error:  # the logs stay in the run folder
        note("packing failed: %s; the logs stay in %s" % (error, args.logdir))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
