#!/usr/bin/env python3
"""Summarize how one startup run ended, using the existing debug-launch logs.

The report is deliberately small: an exit code, one of six fixed end-state
labels, the last present counter and the last presented surface. Input paths,
module names, raw addresses, exception parameters and free-form text are never
copied. The parser accepts the log formats written by
tools/startup/debug_launch.py, tools/startup/run.ps1 and the runtime heartbeats.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Iterable


LOG_NAMES = ("events.txt", "finished.txt", "run.out", "stderr.txt", "stdout.txt", "trap.log")
MAX_BYTES = 16 * 1024 * 1024
FILTERED_FIRST_CHANCE = {"0x80000003", "0x80000004", "0x406d1388", "0x40010006"}

RE_EXIT_EVENT = re.compile(r"^\s*(\d{1,10}(?:\.\d{1,6})?)\s+exit\s+(0x[0-9a-fA-F]{8})\s*$")
RE_EXIT_SUMMARY = re.compile(r"^process exited with (0x[0-9a-fA-F]{8}) after (\d{1,10}(?:\.\d{1,6})?) s$")
RE_STOP_SUMMARY = re.compile(
    r"^stopped at [^(]+\((0x[0-9a-fA-F]{8})\) at .+ after (\d{1,10}(?:\.\d{1,6})?) s "
    r"\((first|second) chance\)"
)
RE_DEBUGGER_STOP = re.compile(r"^debugger stop (0x[0-9a-fA-F]{8}) (first|second)-chance$")
RE_EXCEPTION = re.compile(
    r"^\s*(\d{1,10}(?:\.\d{1,6})?)\s+exception\s+(0x[0-9a-fA-F]{8})\b.*?\b(first|second)-chance\b"
)
RE_TIMEOUT = re.compile(r"^timeout after (\d{1,10}(?:\.\d{1,6})?) s$")
RE_STOP_REQUEST = re.compile(r"^stopped by request after (\d{1,10}(?:\.\d{1,6})?) s \(stop file\)$")
RE_HEARTBEAT = re.compile(
    r"x2w-d3d12 heartbeat: (\d{1,10}) frames presented in (\d{1,10}(?:\.\d{1,6})?) s "
    r"\((\d{1,10}(?:\.\d{1,6})?) fps overall\)"
)
RE_SURFACE = re.compile(
    r"x2w-render presented surface (\d{1,5})x(\d{1,5}) format (\d{1,3}), readback (\d{1,10}), "
    r"mean level (\d{1,3}(?:\.\d{1,6})?)/255"
)
RE_CAPTURE = re.compile(r"^present frame (\d{1,10}) surface \S+$")


@dataclass
class ExitSummary:
    exit_code: str | None = None
    end_elapsed_s: float | None = None
    timed_out: bool = False
    stop_requested: bool = False
    confirmed_stop: tuple[str, str] | None = None
    first_exception: tuple[str, str] | None = None
    last_exception: tuple[str, str] | None = None
    exception_events: int = 0
    last_present_frames: int | None = None
    last_present_s: float | None = None
    last_present_fps: float | None = None
    last_surface: tuple[int, int, int, int, str] | None = None
    last_capture_frame: int | None = None
    lines_scanned: int = 0

    def classification(self) -> str:
        if self.confirmed_stop is not None:
            return "exception stop"
        if self.timed_out:
            return "timeout"
        if self.stop_requested:
            return "stopped by request"
        if self.exit_code is not None:
            return "clean exit" if self.exit_code == "0x00000000" else "nonzero exit"
        return "missing exit event"

    def observe(self, line: str) -> None:
        self.lines_scanned += 1
        match = RE_HEARTBEAT.search(line)
        if match:
            elapsed = float(match.group(2))
            if self.last_present_s is None or elapsed >= self.last_present_s:
                self.last_present_frames = int(match.group(1))
                self.last_present_s = elapsed
                self.last_present_fps = float(match.group(3))
            return
        match = RE_SURFACE.search(line)
        if match:
            self.last_surface = (
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), match.group(5),
            )
            return
        match = RE_CAPTURE.match(line)
        if match:
            self.last_capture_frame = int(match.group(1))
            return
        match = RE_EXIT_EVENT.match(line)
        if match:
            self.exit_code = match.group(2).lower()
            self.end_elapsed_s = float(match.group(1))
            return
        match = RE_EXIT_SUMMARY.match(line)
        if match:
            self.exit_code = match.group(1).lower()
            self.end_elapsed_s = float(match.group(2))
            return
        match = RE_TIMEOUT.match(line)
        if match:
            self.timed_out = True
            self.end_elapsed_s = float(match.group(1))
            return
        match = RE_STOP_REQUEST.match(line)
        if match:
            self.stop_requested = True
            self.end_elapsed_s = float(match.group(1))
            return
        match = RE_DEBUGGER_STOP.match(line)
        if match:
            self.confirmed_stop = (match.group(1).lower(), match.group(2))
            return
        match = RE_STOP_SUMMARY.match(line)
        if match:
            self.confirmed_stop = (match.group(1).lower(), match.group(3))
            self.end_elapsed_s = float(match.group(2))
            return
        match = RE_EXCEPTION.match(line)
        if match and (match.group(3) == "second" or match.group(2).lower() not in FILTERED_FIRST_CHANCE):
            exception = (match.group(2).lower(), match.group(3))
            self.exception_events += 1
            if self.first_exception is None:
                self.first_exception = exception
            self.last_exception = exception


def _read_logs(root: Path) -> dict[str, list[str]]:
    logs: dict[str, list[str]] = {}
    for name in LOG_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()[:MAX_BYTES]
        except OSError:
            continue
        logs[name] = data.decode("utf-8", errors="replace").splitlines()
    return logs


def summarize(root: Path) -> ExitSummary:
    logs = _read_logs(root)
    summary = ExitSummary()
    # finished.txt is authoritative for the final state, so parse it after the other logs and
    # let its exit code and end time overwrite an earlier duplicate event.
    for name in LOG_NAMES:
        if name in ("finished.txt", "run.out"):
            continue
        for line in logs.get(name, ()):
            summary.observe(line)
    for name in ("finished.txt", "run.out"):
        for line in logs.get(name, ()):
            summary.observe(line)
    return summary


def _exception_text(exception: tuple[str, str] | None) -> str:
    return "not reported" if exception is None else f"`{exception[0]}` ({exception[1]}-chance)"


def _exit_text(code: str | None) -> str:
    return "not reported" if code is None else f"`{code}` ({int(code, 16)})"


def _time_text(value: float | None) -> str:
    return "not reported" if value is None else f"`{value:.1f}` s"


def _present_text(summary: ExitSummary) -> str:
    if summary.last_present_frames is not None and summary.last_present_s is not None:
        fps = "" if summary.last_present_fps is None else f" (`{summary.last_present_fps:g}` fps overall)"
        return f"`{summary.last_present_frames}` frames at `{summary.last_present_s:g}` s{fps}"
    if summary.last_capture_frame is not None:
        return f"capture frame `{summary.last_capture_frame}`"
    return "not reported"


def _surface_text(summary: ExitSummary) -> str:
    if summary.last_surface is None:
        return "not reported"
    width, height, format_code, readback, mean = summary.last_surface
    return f"`{width}x{height}` format `{format_code}`, readback `{readback}`, mean level `{mean}/255`"


def format_markdown(summary: ExitSummary) -> str:
    return "\n".join([
        "## Startup Exit Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Exit code | {_exit_text(summary.exit_code)} |",
        f"| Classification | {summary.classification()} |",
        f"| End time | {_time_text(summary.end_elapsed_s)} |",
        f"| Last present | {_present_text(summary)} |",
        f"| Last presented surface | {_surface_text(summary)} |",
        f"| First observed exception | {_exception_text(summary.first_exception)} |",
        f"| Last observed exception | {_exception_text(summary.last_exception)} |",
        f"| Observed exceptions | `{summary.exception_events}` |",
        "",
    ])


def as_json(summary: ExitSummary) -> dict[str, object]:
    return {
        "exit_code": summary.exit_code,
        "classification": summary.classification(),
        "end_elapsed_s": summary.end_elapsed_s,
        "last_present_frames": summary.last_present_frames,
        "last_present_s": summary.last_present_s,
        "last_present_fps": summary.last_present_fps,
        "last_surface": None if summary.last_surface is None else {
            "width": summary.last_surface[0],
            "height": summary.last_surface[1],
            "format": summary.last_surface[2],
            "readback": summary.last_surface[3],
            "mean_level": summary.last_surface[4],
        },
        "first_exception": None if summary.first_exception is None else {
            "code": summary.first_exception[0], "chance": summary.first_exception[1],
        },
        "last_exception": None if summary.last_exception is None else {
            "code": summary.last_exception[0], "chance": summary.last_exception[1],
        },
        "exception_events": summary.exception_events,
        "lines_scanned": summary.lines_scanned,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize how one startup run ended without copying private text.")
    parser.add_argument("logdir", type=Path, help="directory written by debug_launch.py or run.ps1")
    parser.add_argument("--json", type=Path, help="also write the allowlisted fields as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.logdir.is_dir():
        print(f"exit_report: not a directory: {args.logdir}", file=sys.stderr)
        return 2
    summary = summarize(args.logdir)
    print(format_markdown(summary))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(as_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
