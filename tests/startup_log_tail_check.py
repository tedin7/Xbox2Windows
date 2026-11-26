#!/usr/bin/env python3
"""tools/startup/log_tail.py: the end of stderr.txt that debug_launch.py prints into debugger_summary.out keeps the
newest complete storage heartbeat whole.

  - a heartbeat followed by 4000 characters of draw output: the heartbeat's totals are printed whole before the tail;
  - a heartbeat the tail already holds whole is not printed twice;
  - an older heartbeat is not printed when a newer one exists, and a line still being written (no newline) is skipped;
  - the bounded read: a heartbeat further back than the window is not looked for;
  - a missing or empty log gives no text.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "startup"))
import log_tail as L  # noqa: E402

failures = []


def expect(label, got, want):
    if got != want:
        failures.append("%s: got %r, want %r" % (label, got, want))


OLD = ("x2w-dstorage heartbeat 590 s: requests 195000 (inflated=10 failed=0); page tokens 0 at enqueue=0, still 0 at "
       "run=0, replaced after enqueue=0; commit peak=13700 MiB current=13200 MiB, live physical-page sections=4")
NEW = ("x2w-dstorage heartbeat 600 s: requests 195373 (inflated=12 failed=0); page tokens 0 at enqueue=0, still 0 at "
       "run=0, replaced after enqueue=0; commit peak=13795 MiB current=13281 MiB, live physical-page sections=4")
PROGRESS = "x2w-dstorage progress 600 s: pending 0 oldest 0 ms, enqueued 195373 completed 195373, total read 1 KiB"


def write(log, text):
    """The runtime ends its log lines with CRLF."""
    log.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))


DRAWS = "".join("x2w-draw %016x refused: unknown view token %d\n" % (i, i) for i in range(80))  # about 4000 characters

with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "stderr.txt"
    assert len(DRAWS) >= 4000, len(DRAWS)
    write(log, "start\n" + OLD + "\n" + PROGRESS + "\n" + NEW + "\n" + DRAWS + "x2w-draw half a li")
    text = L.tail_text(str(log))
    tail = text[-L.TAIL_CHARS:]
    expect("the tail is the last characters of the log", tail, log.read_bytes().decode("utf-8")[-L.TAIL_CHARS:])
    head = text[:-L.TAIL_CHARS].splitlines()
    expect("the newest heartbeat and progress lines, whole, before the tail",
           [line.split(") ", 1)[1] for line in head], [NEW, PROGRESS])
    expect("the totals survive intact", "requests 195373" in head[0] and "commit peak=13795 MiB current=13281 MiB" in head[0],
           True)
    expect("the older heartbeat is not printed", OLD in text, False)

    # The heartbeat is the last line: the tail holds it whole, nothing is printed twice.
    write(log, DRAWS + NEW + "\n")
    text = L.tail_text(str(log))
    expect("a heartbeat the tail holds whole is printed once", text.count(NEW), 1)
    # A line still being written (no newline at the end) is not a heartbeat yet: the older complete one is kept.
    write(log, OLD + "\n" + DRAWS + NEW[:90])
    expect("a half-written heartbeat is skipped", L.tail_text(str(log)).splitlines()[0].endswith(OLD), True)
    # The bounded read: a heartbeat before the window is not looked for.
    write(log, NEW + "\n" + "x" * 5000 + "\n")
    expect("the window bounds the read", NEW in L.tail_text(str(log), window=4096), False)
    expect("inside the window it is kept", NEW in L.tail_text(str(log), window=8192), True)
    write(log, "")
    expect("an empty log gives no text", L.tail_text(str(log)), "")
    expect("a missing log gives no text", L.tail_text(str(Path(tmp) / "missing.txt")), "")

if failures:
    print("\n".join("FAIL " + f for f in failures))
    sys.exit(1)
print("startup_log_tail_check passed: the newest storage heartbeat survives draw output after it")
