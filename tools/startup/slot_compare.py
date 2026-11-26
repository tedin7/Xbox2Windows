#!/usr/bin/env python3
"""Compare per-slot cost (us/call) of a title run against a replay of the same slots.

Title side: the stand-in's stderr report lines
    x2w-d3d12 command-list slots over 20 ms: slot9 34 ms/867 slot12 80 ms/361 ...
are summed over the whole log per slot (ms and calls). The report lists only slots that
took over 20 ms in an interval, so the title numbers are a lower bound of calls and
an upper-biased us/call: read them as "the slow intervals".

Replay side: replay_summary.json written with X2W_REPLAY_TIMING=1, entries
{"kind", "slot", "calls", "total_us"}. Only command-list slots (kind 4) are compared.

Usage: slot_compare.py TITLE_STDERR REPLAY_SUMMARY_JSON [--top N] [--kind K]
Prints one row per slot: title ms, title calls, title us/call, replay calls,
replay us/call, ratio title/replay. Output is a table on stdout; exit 0.
"""
import argparse
import json
import re
import sys

LIST_KIND = 4
LINE = "command-list slots over"
ENTRY = re.compile(r"slot(\d+) ([\d,]+) ms/([\d,]+)")


def number(text):
    return int(text.replace(",", ""))


def title_slots(lines):
    """Sum (ms, calls) per slot over all 'command-list slots over' report lines."""
    totals = {}
    for line in lines:
        if LINE not in line:
            continue
        for slot, ms, calls in ENTRY.findall(line.split(":", 1)[-1]):
            old_ms, old_calls = totals.get(int(slot), (0, 0))
            totals[int(slot)] = (old_ms + number(ms), old_calls + number(calls))
    return totals


def replay_slots(summary, kind):
    return {e["slot"]: (e["calls"], e["total_us"]) for e in summary.get("slot_times", []) if e["kind"] == kind}


def rows(title, replay):
    out = []
    for slot in sorted(set(title) | set(replay)):
        ms, calls = title.get(slot, (0, 0))
        rcalls, rus = replay.get(slot, (0, 0.0))
        title_us = ms * 1000.0 / calls if calls else None
        replay_us = rus / rcalls if rcalls else None
        ratio = title_us / replay_us if title_us and replay_us else None
        out.append((slot, ms, calls, title_us, rcalls, replay_us, ratio))
    # biggest title time first, slots only in the replay last
    out.sort(key=lambda r: (-r[1], r[0]))
    return out


def cell(value, spec):
    return "-" if value is None else format(value, spec)


def render(table, top):
    lines = ["slot  title_ms title_calls title_us/call replay_calls replay_us/call ratio"]
    for slot, ms, calls, tus, rcalls, rus, ratio in table[:top]:
        lines.append(f"{slot:<5} {ms:>8} {calls:>11} {cell(tus, '.1f'):>13} {rcalls:>12} {cell(rus, '.2f'):>14} {cell(ratio, '.1f'):>5}")
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("title_stderr")
    parser.add_argument("replay_summary")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--kind", type=int, default=LIST_KIND)
    args = parser.parse_args(argv)
    with open(args.title_stderr, errors="replace") as f:
        title = title_slots(f)
    with open(args.replay_summary) as f:
        summary = json.load(f)
    if "slot_times" not in summary:
        print("replay summary has no slot_times: run the player with X2W_REPLAY_TIMING=1", file=sys.stderr)
        return 2
    print(render(rows(title, replay_slots(summary, args.kind)), args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
