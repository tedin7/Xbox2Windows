#!/usr/bin/env python3
"""The end of a run log for debugger_summary.out (debug_launch.py), with the last complete heartbeat lines kept whole.

debug_launch.py prints the last TAIL_CHARS characters of stderr.txt into the summary. A draw-heavy end of run pushed
the last storage heartbeat (requests completed, commit peak and current) out of that window or cut it mid-line, so the
summary lost the run's final totals. tail_text() prints, before the tail, the newest complete line of each heartbeat
kind that the tail does not already hold whole. It reads at most WINDOW_BYTES from the end of the file.
"""
import os

TAIL_CHARS = 3000
WINDOW_BYTES = 8 << 20  # how far back a heartbeat is looked for: the runtime writes one every few seconds
HEARTBEATS = ("x2w-dstorage heartbeat ", "x2w-dstorage progress ")


def last_lines(text, prefixes):
    """{prefix: the newest complete line (ending in a newline) that starts with it} in `text`."""
    found = {}
    for line in reversed(text.splitlines(keepends=True)):
        if not line.endswith("\n"):
            continue  # still being written
        for prefix in prefixes:
            if prefix not in found and line.startswith(prefix):
                found[prefix] = line.rstrip("\r\n")
        if len(found) == len(prefixes):
            break
    return found


def tail_text(path, tail_chars=TAIL_CHARS, prefixes=HEARTBEATS, window=WINDOW_BYTES):
    """The summary text for one log: the newest complete heartbeat lines the tail does not hold whole, then the last
    `tail_chars` characters. Empty when the file is missing or empty."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            handle.seek(max(0, size - window))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    if not text:
        return ""
    tail = text[-tail_chars:]
    # A line the tail holds whole starts right after a newline of the tail (or at the start of the window).
    whole = tail if len(text) <= tail_chars else tail[tail.find("\n") + 1:] if "\n" in tail else ""
    kept = [line for prefix, line in sorted(last_lines(text, prefixes).items()) if line + "\n" not in whole]
    return "".join("(newest complete line, kept from before the tail) %s\n" % line for line in kept) + tail
