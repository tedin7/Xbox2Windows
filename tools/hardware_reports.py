"""List hardware-thread GPU reports that the README coverage table does not link yet.

Export the comments of the hardware thread with the GitHub CLI, then run the tool:

    gh api --paginate repos/R0Y00/Xbox2Windows/issues/1/comments > out/issue-1-comments.json
    python -B tools/hardware_reports.py out/issue-1-comments.json

The tool reads the layout written by tools/gpu-check/Run-GpuCheck.ps1. For each report
whose comment README.md does not link, it prints the adapter, driver, Windows build,
source commit, debug-layer status and test results, the coverage row the report
belongs to and suggested cells. Comments with results in another layout, such as an
attached report file, a Linux translation-layer run, repeated test sections or
conflicting hardware and WARP labels, are listed for manual review.
The tool makes no network requests and does not edit README.md. A maintainer still
checks every row against the original report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
THREAD_URL = "https://github.com/R0Y00/Xbox2Windows/issues/1"
HEADER = "## Xbox2Windows GPU report"
TABLE_HEADER = "| Adapter or backend | Default native lanes | Forced wave64 | BVH differential | Evidence |"
EN_DASH = "\u2013"
DUPLICATE_SECTION = "duplicate test section: "
BACKEND_CONFLICT = "conflicting backends: "

# Run-GpuCheck.ps1 reads its summary with these patterns. tests/hardware_reports_check.py
# keeps both copies equal.
FIELD_PATTERNS = {
    "adapter": r"(?m)^adapter=(.+?)\s+vendor=",
    "vendor": r"(?m)^adapter=.+?\s+vendor=([0-9A-Fa-f]+)\s+device=",
    "device": r"(?m)^adapter=.+?\s+vendor=[0-9A-Fa-f]+\s+device=([0-9A-Fa-f]+)",
    "driver": r"(?m)^driver=(.+)$",
    "debugLayer": r"(?m)^backend=\S+\s+debug_layer=(\S+)",
    "laneRange": r"(?m)^native_lane_range=(\S+)",
}

# Hand-edited copies of the report keep most of its layout, so these patterns also
# accept bold labels, code spans, short commits and other dashes. Such reports get a note.
HEADER_LINE = re.compile(r"(?m)^#{0,6}[ \t]*Xbox2Windows GPU (?:report|check)\b.*$", re.I)
COMMIT_LINE = re.compile(r"(?m)^\W*Source commit:?\W*?([0-9a-f]{7,40})\b")
BUILD_LINE = re.compile(r"(?m)Windows build:?\W*?(\d+(?:\.\d+)?)\b")
BACKEND_LINE = re.compile(r"(?m)^\W*Backend:?\W*?(hardware|warp)\b")
FILTER_LINE = re.compile(r"(?m)^Adapter filter:[ \t]*(.+?)[ \t]*$")
SECTION = re.compile(r"(?m)^(#{3,6}[ \t]*)?`?([a-z0-9_]+?)_(hardware|warp)`?"
                     r"[ \t]+([-\u2013\u2014])[ \t]+\**(PASS|FAIL|SKIP)\b.*$")
FENCE = re.compile(r"```[a-z]*[ \t]*\n(.*?)```", re.S)
COMMENT_LINK = re.compile(r"github\.com/R0Y00/Xbox2Windows/issues/1#issuecomment-(\d+)")
ATTACHED_REPORT = re.compile(r"\]\(https://github\.com/user-attachments/[^)\s]*gpu-report[^)\s]*\)", re.I)
RESULT_HINT = re.compile(r"(?m)^(?:native_wave_width|native_lane_range|dxr_tier)="
                         r"|\bbvh_waves_(?:hardware|warp)\b")
TRANSLATION_LAYER = re.compile(r"vkd3d|\bproton\b|\bwine\b|steamuser|\bradv\b|^amdgpu:", re.I | re.M)
TRADEMARKS = re.compile(r"\((?:tm|r)\)", re.I)


@dataclass
class Report:
    comment_id: int
    url: str
    author: str
    created: str
    commit: str
    windows_build: str | None
    backend: str
    adapter_filter: str | None
    fields: dict[str, str | None]
    results: dict[str, str]
    blocks: dict[str, str]
    duplicate_of: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def adapter(self) -> str:
        return self.fields.get("adapter") or "Adapter not reported"


@dataclass
class Row:
    adapter: str
    lanes: str
    wave64: str
    bvh: str
    comment_ids: set[int]


def load_comments(paths: list[Path]) -> list[dict]:
    """Read GitHub REST issue comments, including the concatenated pages of `gh api --paginate`."""
    decoder = json.JSONDecoder()
    comments: dict[int, dict] = {}
    for path in paths:
        data = path.read_bytes()
        # Windows PowerShell 5.1 redirects command output as UTF-16.
        text = data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
        index = 0
        while True:
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            value, index = decoder.raw_decode(text, index)
            for item in value if isinstance(value, list) else [value]:
                if not isinstance(item, dict) or not isinstance(item.get("id"), int) or "body" not in item:
                    raise ValueError(f"{path}: expected GitHub issue comment objects with id and body")
                comments[item["id"]] = item
    return [comments[key] for key in sorted(comments)]


def comment_author(comment: dict) -> str:
    return (comment.get("user") or {}).get("login") or "unknown"


def text_blocks(chunk: str) -> str:
    fenced = FENCE.findall(chunk)
    return "\n".join(fenced) if fenced else chunk


def report_segments(body: str) -> list[str]:
    """Split a comment that pastes several reports, for example a hardware and a WARP run."""
    starts = [match.start() for match in HEADER_LINE.finditer(body)]
    if len(starts) < 2:
        return [body]
    return [body[start:end] for start, end in zip(starts, starts[1:] + [len(body)])]


def parse_segment(comment: dict, text: str) -> tuple[Report | None, str | None]:
    sections = list(SECTION.finditer(text))
    header = HEADER_LINE.search(text)
    commit = COMMIT_LINE.search(text)
    if not sections:
        return None, "results without the generated test sections"
    if not (header or commit):
        return None, "test results without the generated header, for example a source-built run"
    if not commit:
        return None, "generated layout without a source commit"
    if not re.search(FIELD_PATTERNS["adapter"], text):
        return None, "no adapter line in the test output"
    results: dict[str, str] = {}
    blocks: dict[str, str] = {}
    for number, match in enumerate(sections):
        end = sections[number + 1].start() if number + 1 < len(sections) else len(text)
        name = match.group(2)
        if name in results:
            return None, f"{DUPLICATE_SECTION}{name}_{match.group(3)}"
        results[name] = match.group(5)
        blocks[name] = text_blocks(text[match.end():end])
    section_backends = {match.group(3) for match in sections}
    if len(section_backends) > 1:
        return None, BACKEND_CONFLICT + "test sections name both hardware and warp"
    backend = section_backends.pop()
    for label in BACKEND_LINE.finditer(text, 0, sections[0].start()):
        if label.group(1) != backend:
            return None, f"{BACKEND_CONFLICT}report says {label.group(1)}, test sections say {backend}"
    joined = "\n".join(blocks.values())
    fields = {}
    for key, pattern in FIELD_PATTERNS.items():
        found = re.search(pattern, joined)
        fields[key] = found.group(1).strip() if found else None
    build = BUILD_LINE.search(text)
    adapter_filter = FILTER_LINE.search(text)
    report = Report(
        comment_id=int(comment["id"]),
        url=comment.get("html_url") or f"{THREAD_URL}#issuecomment-{comment['id']}",
        author=comment_author(comment),
        created=comment.get("created_at") or "",
        commit=commit.group(1),
        windows_build=build.group(1) if build else None,
        backend=backend,
        adapter_filter=adapter_filter.group(1) if adapter_filter else None,
        fields=fields,
        results=results,
        blocks=blocks,
    )
    edited = (
        not header
        or header.group(0).strip() != HEADER
        or len(commit.group(1)) != 40
        or any(match.group(1) is None or match.group(4) != "-" for match in sections)
    )
    if edited:
        report.notes.append("edited report layout; compare the cells with the comment")
    return report, None


def parse_comment(comment: dict) -> tuple[list[Report], str | None]:
    """Return usable reports and any reason to review the comment by hand."""
    lines = (comment.get("body") or "").replace("\r\n", "\n").split("\n")
    # Quoted lines repeat another comment, so they never count as a report.
    body = "\n".join(line for line in lines if not line.lstrip().startswith(">"))
    if not SECTION.search(body):
        if ATTACHED_REPORT.search(body):
            return [], "report attached as a file"
        if RESULT_HINT.search(body) or HEADER_LINE.search(body) or COMMIT_LINE.search(body):
            if TRANSLATION_LAYER.search(body):
                return [], "Linux translation-layer run; README records these in prose"
            return [], "results without the generated test sections"
        return [], None
    if TRANSLATION_LAYER.search(body):
        return [], "Linux translation-layer run; README records these in prose"
    reports, reasons, pasted = [], [], set()
    for text in report_segments(body):
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized in pasted:
            continue
        pasted.add(normalized)
        report, reason = parse_segment(comment, text)
        if report:
            reports.append(report)
        elif reason:
            reasons.append(reason)
    # Keep ambiguous results visible even when another pasted run is usable.
    if reports:
        reasons = [reason for reason in reasons if reason.startswith((DUPLICATE_SECTION, BACKEND_CONFLICT))]
    return reports, next(iter(reasons), None)


def first_int(pattern: str, text: str) -> int | None:
    found = re.search(pattern, text)
    return int(found.group(1)) if found else None


def lanes_cell(report: Report) -> str | None:
    lanes = first_int(r"(?m)^native_wave_width=(\d+)", report.blocks.get("bvh_waves", ""))
    return None if lanes is None else str(lanes)


def wave64_cell(report: Report) -> str | None:
    status = report.results.get("bvh_wave64")
    block = report.blocks.get("bvh_wave64", "")
    if status == "PASS":
        lanes = first_int(r"(?m)^native_wave_width=(\d+)", block)
        return f"Passed, {lanes} lanes" if lanes else "Passed"
    if status == "SKIP":
        reasons = []
        lane_range = re.search(r"(?m)^native_lane_range=(\d+)\.\.(\d+)", block)
        if lane_range and int(lane_range.group(2)) < 64:
            reasons.append(f"lane range {lane_range.group(1)}{EN_DASH}{lane_range.group(2)}")
        model = re.search(r"shader_model_capped_at_6_6=0x([0-9a-f])([0-9a-f])\b", block, re.I)
        if model and int(model.group(1) + model.group(2), 16) < 0x66:
            reasons.append(f"Shader Model {int(model.group(1), 16)}.{int(model.group(2), 16)}")
        return "Skipped, " + ", ".join(reasons) if reasons else "Skipped"
    return "Failed" if status == "FAIL" else None


def bvh_cell(report: Report) -> str | None:
    status = report.results.get("bvh_differential")
    block = report.blocks.get("bvh_differential", "")
    if status == "PASS":
        total = sum(int(value) for value in re.findall(r"\bcomparisons=(\d+)", block))
        return f"{total} queries passed" if total else "Passed"
    if status == "SKIP":
        tier = first_int(r"\bdxr_tier=(\d+)", block)
        if tier == 0:
            return "Skipped, DXR unsupported"
        if tier is not None and tier < 11:
            return f"Skipped, DXR tier {tier // 10}.{tier % 10}"
        return "Skipped"
    return "Failed" if status == "FAIL" else None


def evidence_label(report: Report) -> str:
    driver = report.fields.get("driver")
    label = f"{driver} report" if driver and not driver.startswith("unavailable") else "Hardware report"
    if report.fields.get("debugLayer") == "enabled":
        label += ", debug layer enabled"
    if report.adapter_filter:
        label += f", adapter filter {report.adapter_filter}"
    return label


def adapter_key(name: str) -> str:
    """Compare adapter names without trademark marks or a trailing "Graphics"."""
    key = re.sub(r"\s+", " ", TRADEMARKS.sub("", name)).strip().lower()
    return re.sub(r" graphics$", "", key)


def row_key(cell: str) -> tuple[str, str | None]:
    plain = TRADEMARKS.sub("", cell).strip()
    trailing = re.match(r"(.*?)\s*\(([^()]*)\)$", plain)
    base, note = (trailing.group(1), trailing.group(2)) if trailing else (plain, "")
    device = re.search(r"\bdevice ([0-9a-f]{4})\b", note, re.I)
    return adapter_key(base), device.group(1).lower() if device else None


def coverage_rows(readme: str) -> list[Row]:
    lines = readme.replace("\r\n", "\n").split("\n")
    try:
        start = lines.index(TABLE_HEADER)
    except ValueError:
        raise ValueError("README.md has no hardware coverage table header") from None
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) != 5:
            raise ValueError(f"coverage row without five cells: {line[:80]}")
        ids = {int(number) for number in COMMENT_LINK.findall(cells[4])}
        rows.append(Row(cells[0], cells[1], cells[2], cells[3], ids))
    return rows


def linked_comment_ids(readme: str) -> set[int]:
    return {int(number) for number in COMMENT_LINK.findall(readme)}


def matching_row(report: Report, rows: list[Row]) -> Row | None:
    key = adapter_key(report.fields.get("adapter") or "")
    if report.backend == "warp" or key == "microsoft basic render driver":
        return next((row for row in rows if row_key(row.adapter)[0].startswith("microsoft warp")), None)
    device = (report.fields.get("device") or "").lower()
    candidates = [row for row in rows if row_key(row.adapter)[0] == key]
    exact = [row for row in candidates if row_key(row.adapter)[1] == device]
    general = [row for row in candidates if row_key(row.adapter)[1] is None]
    return next(iter(exact or general), None)


def row_differences(report: Report, row: Row) -> list[str]:
    notes = []
    lanes = lanes_cell(report)
    if lanes and not re.match(rf"{lanes}\b", row.lanes):
        notes.append(f"native lanes {lanes}; the row says {row.lanes!r}")
    wave64 = wave64_cell(report) or ""
    if wave64.split(",")[0].lower() not in row.wave64.lower():
        notes.append(f"forced wave64 {wave64!r}; the row says {row.wave64!r}")
    bvh = bvh_cell(report) or ""
    if bvh.split(",")[0].split(" ")[0].lower() not in row.bvh.lower():
        notes.append(f"BVH differential {bvh!r}; the row says {row.bvh!r}")
    return notes


def describe(report: Report, row: Row | None) -> dict:
    duplicate = [f"same text as comment {report.duplicate_of}"] if report.duplicate_of else []
    return {
        "comment_id": report.comment_id,
        "url": report.url,
        "author": report.author,
        "created": report.created,
        "adapter": report.adapter,
        "vendor": report.fields.get("vendor"),
        "device": report.fields.get("device"),
        "driver": report.fields.get("driver"),
        "debug_layer": report.fields.get("debugLayer"),
        "windows_build": report.windows_build,
        "commit": report.commit,
        "backend": report.backend,
        "passed": sum(status == "PASS" for status in report.results.values()),
        "skipped": [name for name, status in report.results.items() if status == "SKIP"],
        "failed": [name for name, status in report.results.items() if status == "FAIL"],
        "row": row.adapter if row else None,
        "cells": {
            "lanes": lanes_cell(report),
            "wave64": wave64_cell(report),
            "bvh": bvh_cell(report),
            "evidence": f"[{evidence_label(report)}]({report.url})",
        },
        "notes": report.notes + duplicate + (row_differences(report, row) if row else []),
    }


def summarize(comments: list[dict], readme: str, after: int | None = None) -> dict:
    rows = coverage_rows(readme)
    linked = linked_comment_ids(readme)
    first_seen: dict[str, int] = {}
    reports: list[Report] = []
    review = []
    for comment in comments:
        number = int(comment["id"])
        found, reason = parse_comment(comment)
        digest = hashlib.sha256(re.sub(r"\s+", " ", comment.get("body") or "").strip().encode()).hexdigest()
        if found:
            duplicate_of = first_seen.setdefault(digest, number)
            for report in found:
                report.duplicate_of = None if duplicate_of == number else duplicate_of
        if after is not None and number <= after:
            continue
        reports.extend(found)
        if reason and number not in linked:
            review.append({"comment_id": number, "url": comment.get("html_url"),
                           "author": comment_author(comment), "reason": reason})
    return {
        "comments": len(comments),
        "after": after,
        "reports": len(reports),
        "linked": sum(report.comment_id in linked for report in reports),
        "unrecorded": [describe(report, matching_row(report, rows))
                       for report in reports if report.comment_id not in linked],
        "review": review,
    }


def markdown(summary: dict) -> str:
    scope = f" after comment {summary['after']}" if summary["after"] is not None else ""
    out = [
        "# Hardware reports not linked from README.md",
        "",
        f"Read {summary['comments']} comments. {summary['reports']} reports{scope} use the generated "
        f"layout; README.md links {summary['linked']} of them.",
    ]
    for number, item in enumerate(summary["unrecorded"], 1):
        cells = {key: value or "?" for key, value in item["cells"].items()}
        target = f"existing row `{item['row']}`" if item["row"] else "new row"
        tests = f"{item['passed']} passed"
        if item["skipped"]:
            tests += f", skipped: {', '.join(item['skipped'])}"
        if item["failed"]:
            tests += f", failed: {', '.join(item['failed'])}"
        unknown = "not reported"
        out += [
            "",
            f"## {number}. {item['adapter']} ({target})",
            "",
            f"- Report: {item['url']} by @{item['author']}, {item['created']}",
            f"- Driver {item['driver'] or unknown}, Windows build {item['windows_build'] or unknown}, "
            f"source {item['commit'][:7]}, backend {item['backend']}, "
            f"debug layer {item['debug_layer'] or unknown}",
            f"- Tests: {tests}",
            f"- Cells: lanes {cells['lanes']} | wave64 {cells['wave64']} | BVH {cells['bvh']}",
            f"- Evidence: {cells['evidence']}",
        ]
        out += [f"- Check: {note}" for note in item["notes"]]
        if not item["row"]:
            name = re.sub(r"\s+", " ", TRADEMARKS.sub("", item["adapter"])).strip()
            row = (name, cells["lanes"], cells["wave64"], cells["bvh"], cells["evidence"])
            out += ["", "| " + " | ".join(row) + " |"]
    if summary["review"]:
        out += ["", "# Comments for manual review", ""]
        out += [f"- {item['url']} by @{item['author']}: {item['reason']}" for item in summary["review"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("comments", nargs="+", type=Path, help="issue #1 comments from the GitHub REST API")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md",
                        help="README with the coverage table (default: the repository README)")
    parser.add_argument("--after", type=int, help="list only comments with a larger comment ID")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    try:
        comments = load_comments(args.comments)
        summary = summarize(comments, args.readme.read_text(encoding="utf-8"), args.after)
    except (OSError, ValueError) as error:
        print(f"hardware_reports: {error}", file=sys.stderr)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
