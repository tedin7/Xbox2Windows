#!/usr/bin/env python3
"""Build a bounded Markdown report from a per-draw texel history.

Refs #870.  The startup draw dump writes ``pixel_history.txt`` when
``X2W_DRAW_DUMP_FRAME`` and ``X2W_DRAW_DUMP_PIXEL`` are set (see
``src/startup_stubs/pixel_history.h``).  This tool turns that history into a
small review artifact for the first scene-colour write: it selects a frame and
a pixel, names the first retained draw whose sample changes the value, and
reports the last retained writer.  Full-dump BMPs are read only to check the
selected pixel and the target dimensions; they are not merged into the history.

The report never copies raw log lines, paths, arbitrary text or game filenames.
Only allowlisted values (numbers, DXGI format names, opaque 64-bit keys and the
fixed labels below) reach the output.

Uses only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import sys
from typing import Iterable, Sequence


TOOL_VERSION = 1
HISTORY_FILE = "pixel_history.txt"
RUNTIME_SAMPLE_LIMIT = 48

MAX_HISTORY_BYTES = 1 << 20
MAX_HISTORY_RECORDS = 4096
MAX_LINE_BYTES = 4096
MAX_BMPS = 64
MAX_BMP_BYTES = 64 << 20
MAX_BMP_DIMENSION = 16384
DEFAULT_MAX_ROWS = 24
MAX_MAX_ROWS = 64
MIN_MAX_ROWS = 8
MAX_GROUPS = 8

# DXGI_FORMAT values that pixel_history_decode() can turn into RGBA channels.
# The second member is True when the 32-bit word stores B8G8R8A8.
RGBA8_FORMATS: dict[int, tuple[str, bool]] = {
    28: ("DXGI_FORMAT_R8G8B8A8_UNORM", False),
    29: ("DXGI_FORMAT_R8G8B8A8_UNORM_SRGB", False),
    87: ("DXGI_FORMAT_B8G8R8A8_UNORM", True),
    91: ("DXGI_FORMAT_B8G8R8A8_UNORM_SRGB", True),
}

CLASS_LABELS = {
    "no_draw": "aucun draw",
    "ui_only": "draw UI seulement",
    "target_write": "écriture de cible",
    "value_unchanged": "valeur inchangée",
    "history_truncated": "historique tronqué",
    "target_non_rgba8": "cible non-RGBA8",
}

HEADER_RE = re.compile(r"\Ax2w-pixel-history 1 samples=([0-9]+)\Z")
SAMPLE_RE = re.compile(
    r"\Adraw=([0-9]+) frame=(-?[0-9]+) x=([0-9]+) y=([0-9]+) "
    r"key=([0-9A-Fa-f]{16}) count=([0-9]+) format=([0-9]+) raw=([0-9A-Fa-f]{8})"
    r"(?: rgba=([0-9]{1,3}),([0-9]{1,3}),([0-9]{1,3}),([0-9]{1,3}))?\Z"
)
BMP_NAME_RE = re.compile(r"\Adraw_([0-9]+)_([0-9A-Fa-f]{16})_([0-9]+)\.bmp\Z", re.IGNORECASE)

UINT64_MAX = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT32_MAX = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1


class ReportError(Exception):
    """A controlled input error whose text is safe to print."""


@dataclass(frozen=True)
class Sample:
    draw: int
    frame: int
    x: int
    y: int
    key: int
    count: int
    format: int
    raw: int
    rgba: tuple[int, int, int, int] | None
    order: int


@dataclass
class History:
    declared: int
    samples: list[Sample]
    physical_records: int
    malformed: int
    read_truncated: bool
    header_mismatch: bool
    duplicate_draws: int
    rgba_mismatches: int
    rgba_missing: int
    rgba_ignored: int

    @property
    def cap_reached(self) -> bool:
        return (
            self.read_truncated
            or self.header_mismatch
            or self.declared >= RUNTIME_SAMPLE_LIMIT
            or len(self.samples) >= RUNTIME_SAMPLE_LIMIT
        )


@dataclass(frozen=True)
class Group:
    frame: int
    x: int
    y: int
    format: int
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class BmpInfo:
    path: Path
    draw: int
    key: int
    count: int
    width: int
    height: int
    top_down: bool
    bits_per_pixel: int
    pixel_offset: int
    row_bytes: int
    file_size: int


@dataclass
class BmpSummary:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    dimensions: tuple[tuple[int, int], ...] = ()
    matched: int = 0
    mismatched: int = 0
    out_of_range: int = 0
    unreadable: int = 0
    skipped: int = 0
    mismatch_rows: tuple[tuple[int, str, str], ...] = ()


def _format_label(fmt: int) -> str:
    entry = RGBA8_FORMATS.get(fmt)
    if entry is not None:
        return entry[0]
    return "unknown"


def _format_cell(fmt: int) -> str:
    return f"{_format_label(fmt)} (`{fmt}`)"


def _key_cell(key: int) -> str:
    return f"`{key:016x}`"


def _raw_cell(raw: int) -> str:
    return f"`0x{raw:08x}`"


def _rgba_cell(rgba: tuple[int, int, int, int] | None) -> str:
    if rgba is None:
        return "_n/a_"
    return f"`{rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]}`"


def decode_rgba(raw: int, fmt: int) -> tuple[int, int, int, int] | None:
    """Decode a raw 32-bit word the same way pixel_history_decode() does."""
    entry = RGBA8_FORMATS.get(fmt)
    if entry is None:
        return None
    bgra = entry[1]
    if bgra:
        return (raw >> 16) & 0xFF, (raw >> 8) & 0xFF, raw & 0xFF, (raw >> 24) & 0xFF
    return raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF


def parse_history(path: Path) -> History:
    """Parse pixel_history.txt with bounded reads and strict allowlisted fields."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(MAX_HISTORY_BYTES + 1)
    except OSError as exc:
        raise ReportError("pixel_history.txt is not readable") from exc

    read_truncated = size > MAX_HISTORY_BYTES
    if read_truncated:
        data = data[:MAX_HISTORY_BYTES]
        # Drop a final partial record rather than report it as malformed.
        if data and not data.endswith((b"\n", b"\r")):
            cut = data.rfind(b"\n")
            data = data[:cut] if cut >= 0 else b""

    lines = data.splitlines()
    if not lines:
        raise ReportError("pixel_history.txt is empty or has no header")

    try:
        header = lines[0].decode("ascii").rstrip("\r")
    except UnicodeDecodeError as exc:
        raise ReportError("pixel_history.txt has a non-ASCII header") from exc
    match = HEADER_RE.fullmatch(header)
    if match is None:
        raise ReportError("pixel_history.txt has a missing or unsupported header")
    declared = int(match.group(1))
    if declared > MAX_HISTORY_RECORDS:
        read_truncated = True

    records = lines[1:]
    if len(records) > MAX_HISTORY_RECORDS:
        read_truncated = True
        records = records[:MAX_HISTORY_RECORDS]
    physical_records = len(records)
    header_mismatch = read_truncated or declared != physical_records

    samples: list[Sample] = []
    malformed = 0
    duplicates = 0
    rgba_mismatches = 0
    rgba_missing = 0
    rgba_ignored = 0
    seen_draws: set[int] = set()

    for order, raw_line in enumerate(records):
        if not raw_line or len(raw_line) > MAX_LINE_BYTES:
            malformed += 1
            continue
        text = raw_line.decode("ascii", "replace").rstrip("\r")
        match = SAMPLE_RE.fullmatch(text)
        if match is None:
            malformed += 1
            continue
        try:
            draw = int(match.group(1))
            frame = int(match.group(2))
            x = int(match.group(3))
            y = int(match.group(4))
            key = int(match.group(5), 16)
            count = int(match.group(6))
            fmt = int(match.group(7))
            raw = int(match.group(8), 16)
            declared_rgba = None
            if match.group(9) is not None:
                declared_rgba = tuple(int(match.group(index)) for index in range(9, 13))
        except (TypeError, ValueError):
            malformed += 1
            continue

        if not (0 <= draw <= UINT64_MAX and INT64_MIN <= frame <= INT64_MAX
                and 0 <= x <= UINT32_MAX and 0 <= y <= UINT32_MAX
                and 0 <= count <= UINT32_MAX and 0 <= fmt <= INT32_MAX
                and 0 <= raw <= UINT32_MAX):
            malformed += 1
            continue
        if declared_rgba is not None and any(channel > 255 for channel in declared_rgba):
            malformed += 1
            continue

        decoded = decode_rgba(raw, fmt)
        if decoded is None:
            if declared_rgba is not None:
                rgba_ignored += 1
        elif declared_rgba is None:
            rgba_missing += 1
        elif declared_rgba != decoded:
            rgba_mismatches += 1

        if draw in seen_draws:
            duplicates += 1
        else:
            seen_draws.add(draw)
        samples.append(Sample(draw=draw, frame=frame, x=x, y=y, key=key, count=count,
                              format=fmt, raw=raw, rgba=decoded, order=order))

    return History(
        declared=declared,
        samples=samples,
        physical_records=physical_records,
        malformed=malformed,
        read_truncated=read_truncated,
        header_mismatch=header_mismatch,
        duplicate_draws=duplicates,
        rgba_mismatches=rgba_mismatches,
        rgba_missing=rgba_missing,
        rgba_ignored=rgba_ignored,
    )


def build_groups(samples: Iterable[Sample]) -> list[Group]:
    grouped: dict[tuple[int, int, int, int], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.frame, sample.x, sample.y, sample.format), []).append(sample)
    groups = [
        Group(frame=key[0], x=key[1], y=key[2], format=key[3],
              samples=tuple(sorted(value, key=lambda item: (item.draw, item.order))))
        for key, value in grouped.items()
    ]
    groups.sort(key=lambda group: (-len(group.samples), group.frame, group.x, group.y, group.format))
    return groups


def select_group(groups: Sequence[Group], frame: int | None,
                 pixel: tuple[int, int] | None) -> tuple[Group | None, int]:
    choices = list(groups)
    if frame is not None:
        choices = [group for group in choices if group.frame == frame]
    if pixel is not None:
        choices = [group for group in choices if group.x == pixel[0] and group.y == pixel[1]]
    if not choices:
        return None, 0
    choices.sort(key=lambda group: (-len(group.samples), group.frame, group.x, group.y, group.format))
    return choices[0], len(choices)


def find_baseline(samples: Sequence[Sample], selected: Group) -> Sample | None:
    first = selected.samples[0]
    best = None
    for sample in samples:
        if sample.x != selected.x or sample.y != selected.y or sample.format != selected.format:
            continue
        if (sample.frame, sample.draw, sample.order) >= (first.frame, first.draw, first.order):
            continue
        if best is None or (sample.draw, sample.order) > (best.draw, best.order):
            best = sample
    return best


def _read_bmp_info(path: Path) -> BmpInfo | None:
    try:
        size = path.stat().st_size
        if size < 54 or size > MAX_BMP_BYTES:
            return None
        with path.open("rb") as handle:
            header = handle.read(54)
    except OSError:
        return None
    if len(header) != 54:
        return None
    try:
        bf_type, _, _, _, bf_off_bits = struct.unpack_from("<2sIHHI", header, 0)
        bi_size, width, height, planes, bit_count, compression, _, _, _, _, _ = struct.unpack_from(
            "<IiiHHIIiiII", header, 14)
    except struct.error:
        return None
    if bf_type != b"BM" or bi_size < 40 or planes != 1 or compression != 0:
        return None
    if bit_count not in (24, 32) or width <= 0 or height == 0:
        return None
    if width > MAX_BMP_DIMENSION or abs(height) > MAX_BMP_DIMENSION:
        return None
    top_down = height < 0
    rows = abs(height)
    row_bytes = ((width * bit_count + 31) // 32) * 4
    if bf_off_bits < 54 or bf_off_bits >= size:
        return None
    if bf_off_bits + row_bytes * rows > size:
        return None
    return BmpInfo(path=path, draw=0, key=0, count=0, width=width, height=rows,
                   top_down=top_down, bits_per_pixel=bit_count, pixel_offset=bf_off_bits,
                   row_bytes=row_bytes, file_size=size)


def _bmp_pixel(info: BmpInfo, x: int, y: int) -> tuple[int, int, int] | None:
    if x < 0 or y < 0 or x >= info.width or y >= info.height:
        return None
    row = y if info.top_down else info.height - 1 - y
    offset = info.pixel_offset + row * info.row_bytes + x * (info.bits_per_pixel // 8)
    try:
        with info.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(info.bits_per_pixel // 8)
    except OSError:
        return None
    if len(data) != info.bits_per_pixel // 8:
        return None
    return data[2], data[1], data[0]


def scan_bmps(folder: Path, selected: Group | None) -> BmpSummary:
    summary = BmpSummary()
    try:
        entries = sorted(entry for entry in folder.iterdir() if entry.is_file())
    except OSError:
        return summary

    valid: list[BmpInfo] = []
    for entry in entries:
        if summary.total >= MAX_BMPS:
            break
        name_match = BMP_NAME_RE.fullmatch(entry.name)
        if name_match is None:
            continue
        summary.total += 1
        info = _read_bmp_info(entry)
        if info is None:
            summary.invalid += 1
            continue
        try:
            draw = int(name_match.group(1))
            key = int(name_match.group(2), 16)
            count = int(name_match.group(3))
        except ValueError:
            summary.invalid += 1
            continue
        if not (0 <= draw <= UINT64_MAX and 0 <= count <= UINT32_MAX):
            summary.invalid += 1
            continue
        info = BmpInfo(path=info.path, draw=draw, key=key, count=count, width=info.width,
                       height=info.height, top_down=info.top_down,
                       bits_per_pixel=info.bits_per_pixel, pixel_offset=info.pixel_offset,
                       row_bytes=info.row_bytes, file_size=info.file_size)
        valid.append(info)

    summary.valid = len(valid)
    summary.dimensions = tuple(sorted({(info.width, info.height) for info in valid}))
    if selected is None or not selected.samples:
        return summary

    index = {(info.draw, info.key, info.count): info for info in valid}
    mismatches: list[tuple[int, str, str]] = []
    for sample in selected.samples:
        info = index.get((sample.draw, sample.key, sample.count))
        if info is None:
            continue
        rgb = _bmp_pixel(info, sample.x, sample.y)
        if rgb is None:
            if not (0 <= sample.x < info.width and 0 <= sample.y < info.height):
                summary.out_of_range += 1
            else:
                summary.unreadable += 1
            continue
        if sample.rgba is None:
            summary.skipped += 1
            continue
        expected = sample.rgba[:3]
        if rgb == expected:
            summary.matched += 1
        else:
            summary.mismatched += 1
            if len(mismatches) < 3:
                mismatches.append((sample.draw, f"{expected[0]},{expected[1]},{expected[2]}",
                                   f"{rgb[0]},{rgb[1]},{rgb[2]}"))
    summary.mismatch_rows = tuple(mismatches)
    return summary


def _choose_rows(count: int, maximum: int, must: Iterable[int]) -> tuple[list[int], int]:
    if count <= maximum:
        return list(range(count)), 0
    chosen = {index for index in must if 0 <= index < count}
    left, right = 0, count - 1
    while len(chosen) < maximum and left <= right:
        chosen.add(left)
        left += 1
        if len(chosen) < maximum and left <= right:
            chosen.add(right)
            right -= 1
    ordered = sorted(chosen)
    return ordered, count - len(ordered)


def _classification(history: History, selected: Group | None,
                    baseline_known: bool, baseline_raw: int,
                    transition_count: int) -> tuple[str, str]:
    """Return (primary classification, write outcome)."""
    if selected is None or not selected.samples:
        return "no_draw", "no_draw"
    if selected.format not in RGBA8_FORMATS:
        return "target_non_rgba8", "target_non_rgba8"
    if transition_count == 0:
        if len(selected.samples) == 1 and not baseline_known:
            outcome = "value_unchanged"
        else:
            outcome = "ui_only"
    elif selected.samples[-1].raw != baseline_raw:
        outcome = "target_write"
    else:
        outcome = "value_unchanged"
    if history.cap_reached:
        return "history_truncated", outcome
    return outcome, outcome


def _render_group_table(groups: Sequence[Group], maximum: int = MAX_GROUPS) -> list[str]:
    if not groups:
        return ["_No valid history sample._", ""]
    lines = [
        "| Frame | Pixel | Format | Samples |",
        "| --- | --- | --- | --- |",
    ]
    for group in groups[:maximum]:
        lines.append(
            f"| `{group.frame}` | `{group.x},{group.y}` | {_format_cell(group.format)} "
            f"| `{len(group.samples)}` |"
        )
    if len(groups) > maximum:
        lines.append(f"_`{len(groups) - maximum}` more group(s) not shown._")
    lines.append("")
    return lines


def format_report(history: History, groups: Sequence[Group], selected: Group | None,
                  baseline_sample: Sample | None, baseline_raw: int,
                  baseline_supplied: bool, bmps: BmpSummary,
                  max_rows: int, requested_frame: int | None,
                  requested_pixel: tuple[int, int] | None) -> str:
    lines: list[str] = [
        "# Startup scene-colour write report",
        "",
        f"- Tool: `scene_color_report` v`{TOOL_VERSION}`",
        f"- Input: `X2W_DRAW_DUMP_DIR` (path omitted)",
        f"- History: `{HISTORY_FILE}` version `1`",
        "",
        "## Selection",
        "",
    ]

    if selected is None:
        if requested_frame is not None:
            lines.append(f"- Requested frame: `{requested_frame}`")
        if requested_pixel is not None:
            lines.append(f"- Requested pixel: `{requested_pixel[0]},{requested_pixel[1]}`")
        lines.extend(["- Selected sample group: _none_", ""])
    else:
        lines.extend([
            f"- Frame: `{selected.frame}`",
            f"- Pixel: `{selected.x},{selected.y}`",
            f"- Format: {_format_cell(selected.format)}",
            f"- Retained samples: `{len(selected.samples)}`",
            "",
        ])

    if selected is None:
        baseline_known = False
        transition_count = 0
        first_change = None
        last_writer = None
        changed_flags: list[bool] = []
    else:
        baseline_known = baseline_supplied or baseline_sample is not None
        first_change = None
        last_writer = None
        transition_count = 0
        changed_flags = []
        previous = baseline_raw
        for sample in selected.samples:
            changed = sample.raw != previous
            changed_flags.append(changed)
            if changed:
                transition_count += 1
                if first_change is None:
                    first_change = sample
                last_writer = sample
            previous = sample.raw

    primary, outcome = _classification(history, selected, baseline_known, baseline_raw,
                                       transition_count)
    lines.extend([
        "## Classification",
        "",
        f"- Classification: `{primary}` ({CLASS_LABELS[primary]})",
        f"- Write outcome: `{outcome}` ({CLASS_LABELS[outcome]})",
    ])
    if selected is not None:
        if baseline_supplied:
            baseline_text = f"`raw=0x{baseline_raw:08x}` (supplied)"
        elif baseline_sample is not None:
            baseline_text = (f"`draw={baseline_sample.draw}`, `frame={baseline_sample.frame}`, "
                             f"`raw=0x{baseline_sample.raw:08x}`")
        elif selected.samples:
            baseline_text = ("first retained sample; the pre-draw value is not recorded "
                             f"(`raw=0x{baseline_raw:08x}`)")
        else:
            baseline_text = "_none_"
        lines.append(f"- Baseline: {baseline_text}")

        if selected.format not in RGBA8_FORMATS:
            lines.append("- First observed change: _not evaluated (target is not RGBA8)_")
            lines.append("- Last observed writer: _not evaluated (target is not RGBA8)_")
        elif first_change is None:
            if baseline_known:
                lines.append("- First observed change: _none_")
            else:
                lines.append("- First observed change: _none among the retained samples; the first "
                             "sample is only a baseline_")
            lines.append("- Last observed writer: _none_")
        else:
            lines.append(
                f"- First observed change: `draw={first_change.draw}`, `key={first_change.key:016x}`, "
                f"`raw=0x{first_change.raw:08x}`, {_rgba_cell(first_change.rgba)}"
            )
            lines.append(
                f"- Last observed writer: `draw={last_writer.draw}`, `key={last_writer.key:016x}`, "
                f"`raw=0x{last_writer.raw:08x}`, {_rgba_cell(last_writer.rgba)}"
            )
        if history.cap_reached:
            lines.append("- History limit: the runtime cap or input bound was reached; later draws "
                         "may be missing from the conclusion above.")
    lines.append("")

    lines.extend([
        "## Retained timeline",
        "",
        "Draw ordinals are the runtime copy ordinals (`draw=`), not title draw-call numbers.",
        "",
    ])

    if selected is None or not selected.samples:
        lines.extend(["_No retained sample for the selected frame and pixel._", ""])
    else:
        must = [0, len(selected.samples) - 1]
        if first_change is not None:
            must.append(selected.samples.index(first_change))
        if last_writer is not None:
            must.append(selected.samples.index(last_writer))
        indices, omitted = _choose_rows(len(selected.samples), max_rows, must)
        lines.extend([
            "| # | Draw ordinal | Frame | Key | Count | Format | Raw | RGBA | Observation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for index in indices:
            sample = selected.samples[index]
            observation = ""
            if index == 0 and not baseline_known:
                observation = "baseline; pre-draw value unknown"
            if changed_flags[index]:
                if sample is first_change and sample is last_writer:
                    observation = "first change; last writer"
                elif sample is first_change:
                    observation = "first change"
                elif sample is last_writer:
                    observation = "last writer"
                else:
                    observation = "change"
            lines.append(
                f"| `{index + 1}` | `{sample.draw}` | `{sample.frame}` | {_key_cell(sample.key)} "
                f"| `{sample.count}` | {_format_cell(sample.format)} | {_raw_cell(sample.raw)} "
                f"| {_rgba_cell(sample.rgba)} | {observation} |"
            )
        if omitted:
            lines.append(f"_`{omitted}` timeline row(s) omitted from this bounded table._")
        lines.append("")

    lines.extend([
        "## Input accounting",
        "",
        f"- Header declares: `{history.declared}` sample(s)",
        f"- Physical records: `{history.physical_records}`",
        f"- Valid samples: `{len(history.samples)}`",
        f"- Malformed records ignored: `{history.malformed}`",
        f"- Duplicate draw ordinals: `{history.duplicate_draws}`",
        f"- RGBA/raw mismatches: `{history.rgba_mismatches}`",
        f"- RGBA omitted on a supported format (decoded from raw): `{history.rgba_missing}`",
        f"- RGBA ignored on a non-RGBA8 format: `{history.rgba_ignored}`",
    ])
    if history.read_truncated:
        lines.append("- Input read: stopped at the built-in byte or record bound")
    if history.header_mismatch:
        lines.append("- Header count and physical record count differ")
    lines.append("")

    lines.extend([
        "## BMP cross-check",
        "",
        f"- BMP files: `{bmps.total}` (valid `{bmps.valid}`, invalid `{bmps.invalid}`)",
    ])
    if bmps.dimensions:
        dims = ", ".join(f"`{width}x{height}`" for width, height in bmps.dimensions)
        lines.append(f"- Target dimensions: {dims}")
    else:
        lines.append("- Target dimensions: _not available_")
    lines.extend([
        f"- Matching selected samples: `{bmps.matched}` matched, `{bmps.mismatched}` mismatched, "
        f"`{bmps.out_of_range}` out of range, `{bmps.unreadable}` unreadable, `{bmps.skipped}` skipped",
    ])
    for draw, expected, observed in bmps.mismatch_rows:
        lines.append(f"- BMP mismatch at `draw={draw}`: history RGB `{expected}`, BMP RGB `{observed}`")
    lines.extend([
        "- BMPs carry no frame field and the runtime writes either the full dump or the texel "
        "history, not both; only a matching `draw`/`key`/`count` is compared.",
        "",
        "## Available groups",
        "",
    ])
    lines.extend(_render_group_table(groups))

    lines.extend([
        "## Scope",
        "",
        "- Reads translated draws recorded for render target 0 only.",
        "- Does not run the title, inspect shaders, or prove that a draw is a scene or UI draw.",
        "- `ui_only` means draws were retained but no retained sample changed the selected "
        "pixel; a scene draw that writes the same value cannot be excluded.",
        "- `value_unchanged` means a change was observed but the final retained value equals the "
        "baseline, including a write that restores the earlier value.",
        "- The first retained sample has no pre-draw value.  A write by that first draw is not "
        "visible unless a baseline is supplied.",
        "- The report says nothing about the presented image or about draws outside the dump window.",
        "",
    ])
    return "\n".join(lines)


def _parse_pixel(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("expected X,Y with non-negative decimal integers")
    x, y = int(parts[0]), int(parts[1])
    if x > UINT32_MAX or y > UINT32_MAX:
        raise argparse.ArgumentTypeError("pixel coordinate is out of range")
    return x, y


def _parse_raw(value: str) -> int:
    try:
        raw = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a 32-bit integer (for example 0x00000000)") from exc
    if not 0 <= raw <= UINT32_MAX:
        raise argparse.ArgumentTypeError("raw value is outside 0..0xffffffff")
    return raw


def _parse_non_negative(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("expected a non-negative decimal integer")
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scene_color_report.py",
        description="Summarize a per-draw texel history into a bounded Markdown report.",
    )
    parser.add_argument("dump_dir", type=Path, help="X2W_DRAW_DUMP_DIR containing pixel_history.txt")
    parser.add_argument("--frame", type=int, default=None, help="frame to select")
    parser.add_argument("--pixel", type=_parse_pixel, default=None, help="pixel X,Y to select")
    parser.add_argument("--baseline-raw", type=_parse_raw, default=None,
                        help="known pre-draw raw word for the selected format")
    parser.add_argument("--max-rows", type=_parse_non_negative, default=DEFAULT_MAX_ROWS,
                        help=f"maximum timeline rows (default {DEFAULT_MAX_ROWS}, max {MAX_MAX_ROWS})")
    args = parser.parse_args(argv)

    if args.max_rows < MIN_MAX_ROWS or args.max_rows > MAX_MAX_ROWS:
        print(f"scene_color_report: --max-rows must be between {MIN_MAX_ROWS} and {MAX_MAX_ROWS}",
              file=sys.stderr)
        return 2
    dump_dir: Path = args.dump_dir
    if not dump_dir.is_dir():
        print("scene_color_report: dump directory not found", file=sys.stderr)
        return 2
    history_path = dump_dir / HISTORY_FILE
    if not history_path.is_file():
        print("scene_color_report: pixel_history.txt not found in the dump directory", file=sys.stderr)
        return 2

    try:
        history = parse_history(history_path)
    except ReportError as exc:
        print(f"scene_color_report: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("scene_color_report: pixel_history.txt is not readable", file=sys.stderr)
        return 2

    groups = build_groups(history.samples)
    selected, matching_groups = select_group(groups, args.frame, args.pixel)

    if selected is not None:
        if args.baseline_raw is not None:
            baseline_raw = args.baseline_raw
            baseline_sample = None
            baseline_supplied = True
        else:
            baseline_sample = find_baseline(history.samples, selected)
            if baseline_sample is not None:
                baseline_raw = baseline_sample.raw
                baseline_supplied = False
            else:
                baseline_raw = selected.samples[0].raw
                baseline_supplied = False
    else:
        baseline_sample = None
        baseline_raw = 0
        baseline_supplied = args.baseline_raw is not None

    bmps = scan_bmps(dump_dir, selected)
    report = format_report(
        history=history,
        groups=groups,
        selected=selected,
        baseline_sample=baseline_sample,
        baseline_raw=baseline_raw,
        baseline_supplied=baseline_supplied,
        bmps=bmps,
        max_rows=args.max_rows,
        requested_frame=args.frame,
        requested_pixel=args.pixel,
    )
    if selected is not None and matching_groups > 1:
        report = report.replace(
            "## Classification",
            f"_`{matching_groups}` sample groups match the requested frame and pixel; the largest "
            "one is shown._\n\n## Classification",
            1,
        )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
