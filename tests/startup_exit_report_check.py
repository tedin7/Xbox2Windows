#!/usr/bin/env python3
"""Checks tools/startup/exit_report.py against synthetic debug-launch logs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "startup"))
import exit_report  # noqa: E402


def report(logs: dict[str, str]) -> tuple[str, exit_report.ExitSummary]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text in logs.items():
            (root / name).write_text(text, encoding="utf-8")
        summary = exit_report.summarize(root)
        return exit_report.format_markdown(summary), summary


def expect(report_text: str, summary: exit_report.ExitSummary, want: list[str], label: str) -> None:
    for snippet in want:
        assert snippet in report_text, f"{label}: missing {snippet!r}\n{report_text}"
    assert summary.classification() in {
        "clean exit", "nonzero exit", "exception stop", "timeout", "stopped by request", "missing exit event",
    }, label


def main() -> int:
    clean, clean_summary = report({
        "events.txt": "  301.800 exit 0x00000000\n",
        "stderr.txt": (
            "x2w-d3d12 heartbeat: 7395 frames presented in 301.8 s (24.53 fps overall)\n"
            "x2w-render presented surface 1600x900 format 28, readback 1, mean level 0.0/255\n"
        ),
    })
    expect(clean, clean_summary, [
        "| Exit code | `0x00000000` (0) |",
        "| Classification | clean exit |",
        "| End time | `301.8` s |",
        "| Last present | `7395` frames at `301.8` s (`24.53` fps overall) |",
        "| Last presented surface | `1600x900` format `28`, readback `1`, mean level `0.0/255` |",
    ], "clean exit")

    nonzero, nonzero_summary = report({
        "events.txt": "  307.600 exit 0x00000001\n",
    })
    expect(nonzero, nonzero_summary, [
        "| Exit code | `0x00000001` (1) |",
        "| Classification | nonzero exit |",
        "| End time | `307.6` s |",
        "| First observed exception | not reported |",
    ], "nonzero exit")

    stopped, stopped_summary = report({
        "events.txt": (
            "  250.250 exception 0xE06D7363 C++ exception at C:\\Users\\alice\\game.exe+0x55 first-chance params=['0x1']\n"
            "  299.900 exception 0xC0000005 access violation at KERNELBASE.dll+0x9000 second-chance params=['0x0']\n"
            "debugger stop 0xc0000005 second-chance\n"
        ),
        "finished.txt": (
            "stopped at access violation (0xc0000005) at game.exe+0x4da6341 after 253.49 s "
            "(second chance) params=['0x1']\n"
        ),
    })
    expect(stopped, stopped_summary, [
        "| Classification | exception stop |",
        "| End time | `253.5` s |",
        "| First observed exception | `0xe06d7363` (first-chance) |",
        "| Last observed exception | `0xc0000005` (second-chance) |",
        "| Observed exceptions | `2` |",
    ], "exception stop")
    for secret in ("alice", "game.exe", "KERNELBASE", "0x4da6341", "0x55", "0x9000"):
        assert secret not in stopped, f"exception stop leaked {secret!r}\n{stopped}"

    timed_out, timeout_summary = report({"finished.txt": "timeout after 300.00 s\n"})
    expect(timed_out, timeout_summary, [
        "| Classification | timeout |", "| End time | `300.0` s |",
    ], "timeout")

    requested, requested_summary = report({"finished.txt": "stopped by request after 42.0 s (stop file)\n"})
    expect(requested, requested_summary, ["| Classification | stopped by request |", "| End time | `42.0` s |"], "stop request")

    missing, missing_summary = report({"stderr.txt": "x2w-d3d12 heartbeat: 12 frames presented in 1.0 s (12.0 fps overall)\n"})
    expect(missing, missing_summary, ["| Classification | missing exit event |", "| Last present | `12` frames at `1` s (`12` fps overall) |"], "missing exit")

    filtered, filtered_summary = report({
        "events.txt": (
            "  1.000 exception 0x80000003 breakpoint at ntdll.dll+0x1 first-chance params=[]\n"
            "  2.000 exception 0x406d1388 thread name at game.exe+0x2 first-chance params=[]\n"
        ),
    })
    expect(filtered, filtered_summary, ["| Observed exceptions | `0` |"], "filtered first chance")
    assert "0x80000003" not in filtered and "0x406d1388" not in filtered

    hostile, hostile_summary = report({
        "events.txt": (
            "  3.000 exception 0xE06D7363 C++ exception at C:\\Users\\alice\\secret\\game.exe+0x1 second-chance "
            "params=['0x00007ff7deadbeef', 'https://acme.example/leak']\n"
            "  4.000 exception 0xZZ00ZZ00 junk at X second-chance\n"
        ),
        "stderr.txt": (
            "x2w-d3d12 heartbeat: 5 frames presented in 2.0 s (2.5 fps overall) acme_widget 0x00007ff7deadbeef\n"
            "x2w-render presented surface 1x1 format 28, readback 1, mean level 0.0/255 secret\n"
        ),
    })
    for needle in ("alice", "secret", "game.exe", "acme", "widget", "deadbeef", "http", "0xZZ00ZZ00"):
        assert needle not in hostile, f"hostile input leaked {needle!r}\n{hostile}"
    assert hostile_summary.exception_events == 1

    # The 16 MiB cap applies to the lines that are parsed, not to a discarded first read.
    bounded, bounded_summary = report({
        "events.txt": "  12.500 exit 0x00000000\n",
        "stdout.txt": "x" * (exit_report.MAX_BYTES + 10) +
                      "x2w-d3d12 heartbeat: 99 frames presented in 12.0 s (8.25 fps overall)\n",
    })
    assert "| Last present | not reported |" in bounded, bounded
    assert bounded_summary.last_present_frames is None, bounded_summary

    # finished.txt is authoritative even when events.txt contains an earlier exit line.
    authority, authority_summary = report({
        "finished.txt": "process exited with 0x00000005 after 10.00 s\n",
        "events.txt": "3.000 exit 0x00000000\n",
    })
    expect(authority, authority_summary, [
        "| Exit code | `0x00000005` (5) |",
        "| Classification | nonzero exit |",
        "| End time | `10.0` s |",
    ], "finished authority")

    # The highest present sample wins regardless of file order.
    ordered, ordered_summary = report({
        "stderr.txt": "x2w-d3d12 heartbeat: 50 frames presented in 5.0 s (10.0 fps overall)\n",
        "stdout.txt": "x2w-d3d12 heartbeat: 20 frames presented in 2.0 s (10.0 fps overall)\n",
    })
    expect(ordered, ordered_summary, ["| Last present | `50` frames at `5` s (`10` fps overall) |"], "highest heartbeat")

    second_chance, second_chance_summary = report({
        "events.txt": "  1.000 exception 0x80000003 breakpoint at ntdll.dll+0x1 second-chance params=[]\n",
    })
    assert second_chance_summary.exception_events == 1, second_chance
    assert "0x80000003" in second_chance, second_chance

    payload = exit_report.as_json(clean_summary)
    assert payload["classification"] == "clean exit" and payload["last_present_frames"] == 7395
    assert json.loads(json.dumps(payload)) == payload

    print("startup_exit_report_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
