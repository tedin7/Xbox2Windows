#!/usr/bin/env python3
"""Per-slot command-list cost of two stand-in logs (tester run vs replay run), and the 25 % check of the replay design.

Input: the stand-in's own stderr of each run. Lines read:
  x2w-d3d12 command-list slots over 20 ms: slot12 15827 ms/1271 slot14 5698 ms/11042 ...   (ms, calls since the last report; only slots over 20 ms)
  x2w-d3d12 heartbeat: 4971 frames presented in 580.1 s (...)                              (frames, for calls per frame)
The window is every report line (or --from-report/--to-report, 1-based, to compare the same part of the run). Per slot: total ms, calls,
ms per call, and ms per frame. Verdict: every slot that costs at least --min-ms-per-frame (default 1 ms) in the tester run must have the
replay's ms per call within --tolerance (default 0.25) of the tester's. Slots below the threshold are listed but never fail the check.
Exit 0 match, 1 mismatch, 2 unusable input.
"""
import argparse
import re
import sys
from pathlib import Path

SLOTS = re.compile(r"command-list slots over 20 ms:(.*)")
SLOT = re.compile(r"slot(\d+) (\d+) ms/(\d+)")
FRAMES = re.compile(r"heartbeat: (\d+) frames presented")


def read_run(path, first=1, last=None):
    reports = []
    frames_first = frames_last = None
    for line in Path(path).read_text(errors="replace").splitlines():
        found = SLOTS.search(line)
        if found:
            reports.append({int(s): (int(ms), int(calls)) for s, ms, calls in SLOT.findall(found.group(1))})
        beat = FRAMES.search(line)
        if beat:
            frames_last = int(beat.group(1))
            if frames_first is None:
                frames_first = frames_last
    chosen = reports[first - 1:last]
    total = {}
    for report in chosen:
        for slot, (ms, calls) in report.items():
            old = total.get(slot, (0, 0))
            total[slot] = (old[0] + ms, old[1] + calls)
    frames = max((frames_last or 0) - (frames_first or 0), 0) if chosen else 0
    return total, len(chosen), frames


def compare(tester, replay, tolerance=0.25, min_ms_per_frame=1.0):
    (t_slots, t_reports, t_frames), (r_slots, r_reports, r_frames) = tester, replay
    rows, failed = [], []
    for slot in sorted(set(t_slots) | set(r_slots)):
        t_ms, t_calls = t_slots.get(slot, (0, 0))
        r_ms, r_calls = r_slots.get(slot, (0, 0))
        t_per_call = t_ms / t_calls if t_calls else 0.0
        r_per_call = r_ms / r_calls if r_calls else 0.0
        t_per_frame = t_ms / t_frames if t_frames else 0.0
        counted = t_per_frame >= min_ms_per_frame if t_frames else t_ms >= 1000
        ratio = r_per_call / t_per_call if t_per_call else float("inf") if r_per_call else 1.0
        ok = not counted or abs(ratio - 1.0) <= tolerance
        rows.append((slot, t_calls, t_per_call, r_calls, r_per_call, ratio, counted, ok))
        if not ok:
            failed.append(slot)
    return rows, failed


def format_table(rows):
    lines = ["slot  tester calls  ms/call   replay calls  ms/call   ratio  counted  verdict"]
    for slot, t_calls, t_pc, r_calls, r_pc, ratio, counted, ok in rows:
        lines.append(f"{slot:4d}  {t_calls:12d}  {t_pc:7.3f}   {r_calls:12d}  {r_pc:7.3f}  {ratio:6.2f}  {'yes' if counted else 'no ':7s}  {'ok' if ok else 'MISMATCH'}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tester")
    parser.add_argument("replay")
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--min-ms-per-frame", type=float, default=1.0)
    parser.add_argument("--tester-reports", nargs=2, type=int, metavar=("FIRST", "LAST"), default=[1, None])
    parser.add_argument("--replay-reports", nargs=2, type=int, metavar=("FIRST", "LAST"), default=[1, None])
    args = parser.parse_args(argv)
    tester = read_run(args.tester, *args.tester_reports)
    replay = read_run(args.replay, *args.replay_reports)
    if not tester[1] or not replay[1]:
        print("slot_compare: a log has no 'command-list slots over 20 ms' lines", file=sys.stderr)
        return 2
    rows, failed = compare(tester, replay, args.tolerance, args.min_ms_per_frame)
    print(format_table(rows))
    print(f"slot_compare: tester {tester[1]} reports / {tester[2]} frames, replay {replay[1]} reports / {replay[2]} frames; "
          + ("MATCH" if not failed else "MISMATCH slots " + ",".join(map(str, failed))))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
