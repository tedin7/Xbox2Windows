#!/usr/bin/env python3
"""Summarize a startup log directory (and optional shader cache) into Markdown.

Reads raw logs produced by tools/startup/run.ps1 / debug_launch.py. The report is
built only from allowlisted structured values: known reason/outcome tokens, decimal
IDs and counts, hexadecimal HRESULT/format/flag codes, slot numbers and known
gfx2hlsl mnemonic categories. Any other text (free-form debug messages, shader
failure lines, module/object names, paths, addresses, links) is never copied; it is
grouped under a fixed "other" label. Raw snippets stay in the local log files.
Uses only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Iterable


LOG_EXTENSIONS = {".txt", ".log", ".out"}

DSTORAGE_OK_OUTCOMES = {
    "done",
    "empty",
    "inflated",
    "inflated (larger/short)",
    "inflated (chunked stream piece 0)",
    "inflated (slice from stream cache)",
    "bcpack decoded",
    "inflated + bcpack decoded",
}

# Allowlists: the only values that may be copied from a log into the report.
# Every refusal reason src/startup_stubs/d3d12_memory_srv.inl can log, in the order of its checks.
SLOT70_REASONS = {
    "null_destination",
    "unreadable_resource",
    "unreadable_view",
    "null_address",
    "unsupported_description",
    "destination_outside_view_heaps",
    "gpu_written_resource",
    # the acceleration-structure holds of publish_as_srv (memory_srv_as.h, as_srv_refusal_name)
    "as_unaligned",
    "as_not_placed",
    "as_wrong_state",
    "buffer_unavailable",
    "texture_unavailable",
    # the partial-tile holds of device_memory_srv
    "partial_tile_xg_rows",
    "partial_tile_beyond_xg_size",
    # the console-layout holds of device_memory_srv (#2015)
    "unsupported_desktop_format",
    "console_layout_xg_rows",
    "console_layout_without_footprints",
    "console_layout_beyond_xg_size",
    "overlapping_resource",
}
DSTORAGE_FAIL_OUTCOMES = {
    "bcpack failed",
    "destination capacity exceeded",
    "insufficient page grants",
    "missing page grant token",
    "missing page tokens",
    "page destination write failed",
    "read failed",
    "zlib failed",
}
TRAP_TAGS = {"x2w-trap", "x2w-noop", "x2w-continue", "x2w-fail"}
INDIRECT_PREFIX = "x2w-render indirect: "
# Ordered fields emitted by the slot-61 census in d3d12_xs_device.cpp.
INDIRECT_OUTCOMES = (
    "forwarded", "refused graphics-pipeline", "compute-pipeline", "placeholder-signature", "other",
)
INDIRECT_KINDS = ("other", "dispatch", "draw", "drawindexed", "dispatchmesh")
RE_INDIRECT_COUNT = re.compile(r"[0-9]{1,19}\Z")
# The budget heartbeat is produced by report_budget() in d3d12_xs_device.cpp.
BUDGET_DECIMAL = r"[0-9]{1,20}"
BUDGET_SEGMENT = rf"(?:{BUDGET_DECIMAL}/{BUDGET_DECIMAL}|n/a)"
BUDGET_CATEGORY = rf"{BUDGET_DECIMAL}/{BUDGET_DECIMAL}\+{BUDGET_DECIMAL}/{BUDGET_DECIMAL}"
RE_GPU_BUDGET = re.compile(
    rf"x2w-budget local=(?:(?P<usage>{BUDGET_DECIMAL})/(?P<budget>{BUDGET_DECIMAL})|n/a)"
    rf" nonlocal={BUDGET_SEGMENT} placed={BUDGET_CATEGORY}"
    rf" texture=(?P<live>{BUDGET_DECIMAL})/{BUDGET_DECIMAL}"
    rf"\+(?P<retained>{BUDGET_DECIMAL})/(?P<objects>{BUDGET_DECIMAL})"
    + "".join(rf" {name}={BUDGET_CATEGORY}" for name in ("buffer", "upload", "readback", "heap"))
    + rf" unmeasured={BUDGET_DECIMAL}"
    # Added with the system commit and pool counters (absent in older logs): commit=<used>/<limit> avail=<free> in MiB.
    + rf"(?: commit=(?:{BUDGET_DECIMAL}/{BUDGET_DECIMAL}|n/a) avail=(?:{BUDGET_DECIMAL}|n/a)"
    + rf" sections=(?:{BUDGET_DECIMAL}|n/a) slabs=(?:{BUDGET_DECIMAL}|n/a) slots=(?:{BUDGET_DECIMAL}/{BUDGET_DECIMAL}|n/a))?"
)
OTHER = "other (text omitted)"
RE_DEC = re.compile(r"[0-9]{1,10}\Z")
RE_HEX = re.compile(r"0x[0-9a-fA-F]{1,8}\Z")
RE_SIGN_EXTENDED_HRESULT = re.compile(r"0xffffffff([89a-f][0-9a-f]{7})\Z", re.IGNORECASE)
RE_SLOT = re.compile(r"(?:[A-Za-z0-9_]+!)?(?:([A-Za-z0-9_]+)::)?slot(\d{1,4})\b")
RE_OPENFILE_FAIL = re.compile(r"fail(?: \(err (\d{1,10})\))?\Z")
MNEMONIC_FAMILY = r"(v|s|ds|buffer|tbuffer|image|flat|global|scratch|exp|lds)_[a-z0-9_]{1,40}"
SHADER_REASONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(image_load_mip) (tfe|a16)\Z"), r"\1 \2"),
    (re.compile(r"buffer_load_format_(?:x|xy|xyz|xyzw) requires descriptor FORMAT and DST_SEL\Z"),
     "buffer_* instruction"),
    (re.compile(r"SDWA UNUSED_SEXT\Z"), "SDWA UNUSED_SEXT"),
    (re.compile(r"SDWA dst_unused\b"), "SDWA dst_unused"),
    (re.compile(r"undecodable word\b"), "undecodable word"),
    (re.compile(r"numeric use of s\d+ holding\b"), "numeric use of scalar register"),
    (re.compile(r"operand\b"), "unsupported operand"),
    (re.compile(r"modifier\b"), "unsupported modifier"),
    (re.compile(r"descriptor operand\b"), "descriptor operand"),
    (re.compile(r"sampler operand\b"), "sampler operand"),
    (re.compile(r"mask from descriptor data\Z"), "mask from descriptor data"),
    (re.compile(r"s_load\b"), "s_load form"),
    # Mnemonics are free text, so only the instruction family prefix is reported.
    (re.compile(MNEMONIC_FAMILY + r" SDWA\b"), r"\1_* SDWA form"),
    (re.compile(MNEMONIC_FAMILY + r" with\b"), r"\1_* with modifier"),
    (re.compile(MNEMONIC_FAMILY + r"\Z"), r"\1_* instruction"),
]


# Run end-point: how far a run got and how it ended. Only decimal counts, the fixed failure-cause names of the
# DirectStorage heartbeat and the 32-bit exception code with its chance are copied; the rest of those lines (module
# names, paths, addresses, exception parameters) is dropped.
FAILURE_CAUSES = (
    "read-at-or-past-eof",
    "read-crossing-eof",
    "read-other",
    "destination-capacity",
    "destination-not-writable",
    "zlib",
    "bcpack",
    "other",
)
# Omit common debugger events from the observed exception list unless they are second-chance.
FILTERED_FIRST_CHANCE = {"0x80000003", "0x80000004", "0x406d1388", "0x40010006"}
RE_HEARTBEAT_SECOND = re.compile(r"x2w-dstorage heartbeat (\d{1,10}) s:")
RE_COMMIT_PEAK = re.compile(r"x2w-dstorage heartbeat \d{1,10} s: .*?\bcommit peak=(\d{1,10}) MiB\b")
RE_FAILURE_CAUSES = re.compile(r"x2w-dstorage heartbeat (\d{1,10}) s: failed requests by cause ((?:[a-z-]+=\d{1,10}(?:\s+|$))+)")
RE_CAUSE_COUNT = re.compile(r"([a-z-]+)=(\d{1,10})")
# debug_launch.py events.txt: "<seconds> exception <code> <name> at <where> first|second-chance params=[...]".
RE_EXCEPTION = re.compile(r"^\s*\d+\.\d+\s+exception\s+(0x[0-9a-fA-F]{8})\b.*?\b(first|second)-chance\b")
RE_DEBUGGER_STOP = re.compile(r"^debugger stop (0x[0-9a-fA-F]{8}) (first|second)-chance$")
RE_STOP_SUMMARY = re.compile(
    r"^stopped at [^(]+\((0x[0-9a-fA-F]{8})\) at .+ after \d+\.\d+ s \((first|second) chance\) params=\["
)


def _keep_peak(old: int | None, new: int) -> int:
    return new if old is None else max(old, new)


def _is_later_sample(old_second: int | None, new_second: int) -> bool:
    return old_second is None or new_second >= old_second


@dataclass
class RunEndpoint:
    """Where a run got to, built only from allowlisted values."""

    peak_commit_mib: int | None = None
    last_heartbeat_s: int | None = None
    failure_second: int | None = None
    failure_counts: dict[str, int] = field(default_factory=dict)
    first_exception: tuple[str, str] | None = None
    last_exception: tuple[str, str] | None = None
    exception_events: int = 0
    confirmed_stop: tuple[str, str] | None = None

    def observe(self, line: str) -> None:
        m = RE_HEARTBEAT_SECOND.search(line)
        if m:
            second = int(m.group(1))
            if self.last_heartbeat_s is None or second > self.last_heartbeat_s:
                self.last_heartbeat_s = second
            peak = RE_COMMIT_PEAK.search(line)
            if peak:
                self.peak_commit_mib = _keep_peak(self.peak_commit_mib, int(peak.group(1)))
            causes = RE_FAILURE_CAUSES.search(line)
            if causes and _is_later_sample(self.failure_second, int(causes.group(1))):
                self.failure_second = int(causes.group(1))
                self.failure_counts = {
                    name: int(count) for name, count in RE_CAUSE_COUNT.findall(causes.group(2)) if name in FAILURE_CAUSES
                }
            return
        m = RE_DEBUGGER_STOP.match(line) or RE_STOP_SUMMARY.match(line)
        if m:
            self.confirmed_stop = (m.group(1).lower(), m.group(2))
            return
        m = RE_EXCEPTION.match(line)
        if m and (m.group(2) == "second" or m.group(1).lower() not in FILTERED_FIRST_CHANCE):
            exception = (m.group(1).lower(), m.group(2))
            self.exception_events += 1
            if self.first_exception is None:
                self.first_exception = exception
            self.last_exception = exception


def _dec(text: str | None) -> str:
    return text if text is not None and RE_DEC.match(text) else "?"


def _hexcode(text: str | None) -> str:
    return text.lower() if text is not None and RE_HEX.match(text) else "?"


def classify_shader_reason(line: str) -> str:
    """Map a .fail first line to a known category token; never copies unknown text."""
    for pattern, label in SHADER_REASONS:
        m = pattern.match(line)
        if m:
            return m.expand(label)
    return OTHER


# Log line patterns matching src/startup_stubs/*.
RE_SLOT70 = re.compile(
    r'x2w-d3d12\s+Device::slot70\b.*?\breason=(\S+)\s+res\{([^}]*)\}\s+view\{([^}]*)\}'
)
RE_KV = re.compile(r'(\w+)=([^\s}]+)')

RE_TRAP_TAG = re.compile(r'^(x2w-(?:trap|noop|continue|fail))\s+(\S+)')
RE_D3D12_PIPELINE_TRAP = re.compile(
    r'x2w-d3d12\s+Device::slot([0-9]{1,4})\((Create(?:Graphics|Compute)PipelineState)\)'
)
RE_D3D12_IID_PASS = re.compile(r'x2w-d3d12\s+Device::slot([0-9]{1,4})\s+passes an IID:')
RE_D3D12_VIEW_SKIPPED = re.compile(r'x2w-d3d12\s+Device::slot(\d+)\s+view skipped:\s+(.+)$')
RE_D3D12_FORWARDED_RESULT = re.compile(r'::slot\d+\(.*\)\s*->\s*(0x[0-9a-fA-F]+)\s*$')
# Device::slot133 (console CreatePlacedResource at a title address) logs every handled call as
# "... -> 0x<hresult> aliased|not aliased [gpuva=0x...]", so the line does not end at the HRESULT.
RE_D3D12_SLOT133 = re.compile(
    r'x2w-d3d12\s+Device::slot133\(CreatePlacedResourceX\?\)\s.*?->\s*(0x[0-9a-fA-F]+)\s+(not aliased|aliased)\b'
)
RE_D3D12_SLOT133_UNCOMMITTED = re.compile(r'x2w-d3d12\s+Device::slot133\(CreatePlacedResourceX\?\)\s+address=\S+\s+not committed\b')
SLOT133_OUTCOMES = ("aliased", "not_aliased", "failed")
RE_D3D12_SLOT_CALL = re.compile(r'x2w-d3d12\s+([A-Za-z0-9_]+::slot\d+)\b')
RE_D3D12_REMOVED = re.compile(
    r'x2w-d3d12\s+DEVICE REMOVED,\s+reason\s+(0x[0-9a-fA-F]+),\s+first noticed after\s+(.+)$'
)
RE_D3D12_DEBUG = re.compile(r'x2w-d3d12-debug\s+\[([^\]]+)\]\s+sev=(\d+)\s+id=(\d+)\s+(.+)$')

RE_DSTORAGE_OUTCOME = re.compile(r'x2w-dstorage\s+options=0x[0-9a-fA-F]+.*?->\s+(.+)$')
RE_DSTORAGE_ENTRY_FAIL = re.compile(
    r'x2w-dstorage\s+entry type\s+\d+\s+failed:\s+(.+?)(?:\s+\(options=0x[0-9a-fA-F]+\))?$'
)
RE_DSTORAGE_ZLIB_FAIL = re.compile(r'x2w-dstorage\s+zlib failure:\s+(.+)$')
RE_DSTORAGE_OPENFILE_FAIL = re.compile(r'x2w-dstorage\s+Factory::OpenFile\s+.*?->\s+(fail\b.*)$')
RE_DSTORAGE_HRESULT = re.compile(
    r'x2w-dstorage\s+StatusArray::GetHResult\s+index=\d+\s+->\s+(0x[0-9a-fA-F]+)'
)
RE_DSTORAGE_SIGNAL = re.compile(
    r'x2w-dstorage\s+Queue::EnqueueSignal\s+.*?->\s+(0x[0-9a-fA-F]+)'
)

RE_RUNTIME_CALL = re.compile(
    r'x2w-(?:runtime|api)\s+[0-9a-fA-F]+::slot([0-9]{1,4})(?:\([^)]*\))?\s+->\s+(0x[0-9a-fA-F]+)'
)
RE_RUNTIME_STORAGE = re.compile(
    r'x2w-runtime\s+persistent local storage:\s+local object instead of\s+(0x[0-9a-fA-F]+)'
)
RE_RUNTIME_GAMESAVE = re.compile(
    r'x2w-runtime\s+game save:\s+update refused\s+\(status\s+(-?\d+)\)'
)


def _slot_label(m: re.Match[str] | None) -> str:
    if m is None:
        return OTHER
    return f"{'Device' if m.group(1) == 'Device' else 'other'}::slot{m.group(2)}"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _parse_kv_block(block: str) -> dict[str, str]:
    if block.strip() == "unreadable":
        return {"_unreadable": "1"}
    return dict(RE_KV.findall(block))


def _is_failed_hresult(hex_text: str) -> tuple[bool, str]:
    extended = RE_SIGN_EXTENDED_HRESULT.fullmatch(hex_text)
    if extended:
        val = int(extended.group(1), 16)
    elif RE_HEX.fullmatch(hex_text):
        val = int(hex_text, 16)
    else:
        return False, "?"
    return (val & 0x80000000) != 0, f"0x{val:08x}"


def iter_log_files(log_target: Path) -> list[Path]:
    """Return sorted log files under log_target (or [log_target] if it is a file)."""
    if not log_target.exists():
        return []
    if log_target.is_file():
        return [log_target]
    if not log_target.is_dir():
        return []
    files = [
        p
        for p in log_target.rglob("*")
        if p.is_file() and p.suffix.lower() in LOG_EXTENSIONS
    ]
    files.sort()
    return files


def parse_indirect_snapshot(payload: str) -> dict[tuple[str, str], int] | None:
    groups = payload.split("; ")
    if len(groups) != len(INDIRECT_OUTCOMES):
        return None
    snapshot: dict[tuple[str, str], int] = {}
    for expected, group in zip(INDIRECT_OUTCOMES, groups):
        outcome, _, counts = group.partition(": ")
        if outcome != expected:
            return None
        if counts == "0":
            continue
        fields = counts.split()
        if not fields or len(fields) % 2:
            return None
        for kind, count in zip(fields[::2], fields[1::2]):
            key = (outcome, kind)
            if kind not in INDIRECT_KINDS or key in snapshot or not RE_INDIRECT_COUNT.fullmatch(count):
                return None
            snapshot[key] = int(count)
    return snapshot


@dataclass(frozen=True)
class GpuBudgetSample:
    local: tuple[int, int] | None
    live_texture_bytes: int
    retained_texture_bytes: int
    retained_textures: int


@dataclass
class GpuBudgetSummary:
    samples: int = 0
    known_local_budget: int = 0
    above_local_budget: int = 0
    local_peak: GpuBudgetSample | None = None
    texture_peak: GpuBudgetSample | None = None

    def observe(self, match: re.Match[str]) -> None:
        local = (int(match["usage"]), int(match["budget"])) if match["usage"] is not None else None
        sample = GpuBudgetSample(local, int(match["live"]), int(match["retained"]), int(match["objects"]))
        self.samples += 1
        if self.texture_peak is None or sample.retained_texture_bytes > self.texture_peak.retained_texture_bytes:
            self.texture_peak = sample
        if local is None or local[1] == 0:
            return
        usage, budget = local
        self.known_local_budget += 1
        self.above_local_budget += usage > budget
        peak = self.local_peak.local if self.local_peak else None
        if peak is None or usage * peak[1] > peak[0] * budget:
            self.local_peak = sample


def summarize_lines(lines: Iterable[str]) -> dict[str, Any]:
    """Aggregate startup telemetry counters from an iterable of log lines."""
    slot70_refusals: Counter[tuple[str, str, str, str, str, str, str]] = Counter()
    d3d12_unhandled: Counter[str] = Counter()
    slot133_outcomes: Counter[str] = Counter()
    indirect_peaks: Counter[tuple[str, str]] | None = None
    gpu_budget = GpuBudgetSummary()
    d3d12_debug: Counter[tuple[str, str]] = Counter()
    dstorage_failures: Counter[str] = Counter()
    dstorage_hresults: Counter[str] = Counter()
    runtime_failures: Counter[tuple[str, str]] = Counter()
    endpoint = RunEndpoint()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        endpoint.observe(line)  # reads only; the line is still classified below

        budget_match = RE_GPU_BUDGET.fullmatch(line)
        if budget_match:
            gpu_budget.observe(budget_match)
            continue

        if line.startswith(INDIRECT_PREFIX):
            snapshot = parse_indirect_snapshot(line.removeprefix(INDIRECT_PREFIX))
            if snapshot is not None:
                if indirect_peaks is None:
                    indirect_peaks = Counter()
                # The runtime emits cumulative snapshots. Counter union takes each bucket's maximum.
                indirect_peaks |= Counter(snapshot)
            continue

        # 1. slot70 memory SRV refusals
        m = RE_SLOT70.search(line)
        if m:
            reason, res_raw, view_raw = m.group(1), m.group(2), m.group(3)
            if reason not in SLOT70_REASONS:
                reason = OTHER
            res = _parse_kv_block(res_raw)
            view = _parse_kv_block(view_raw)
            if "_unreadable" in res:
                dim = fmt = layout = flags = "unreadable"
            else:
                dim = _dec(res.get("dim"))
                fmt = _dec(res.get("fmt"))
                layout = _hexcode(res.get("layout"))
                flags = _hexcode(res.get("flags"))
            if "_unreadable" in view:
                view_dim = view_fmt = "unreadable"
            else:
                view_dim = _dec(view.get("dim"))
                view_fmt = _dec(view.get("fmt"))
            slot70_refusals[(reason, dim, fmt, view_dim, view_fmt, layout, flags)] += 1
            continue

        # 2. x2w-trap / x2w-noop / x2w-continue / x2w-fail
        m = RE_TRAP_TAG.match(line)
        if m:
            tag = m.group(1) if m.group(1) in TRAP_TAGS else OTHER
            sm = RE_SLOT.search(m.group(2))
            d3d12_unhandled[f"{tag} {_slot_label(sm)}"] += 1
            continue

        # 3. x2w-d3d12-debug
        m = RE_D3D12_DEBUG.search(line)
        if m:
            d3d12_debug[(_dec(m.group(2)), _dec(m.group(3)))] += 1  # message text is never copied
            continue

        # 4. x2w-d3d12 unhandled/trapped/skipped slots & device removed
        if "x2w-d3d12" in line:
            m = RE_D3D12_REMOVED.search(line)
            if m:
                d3d12_unhandled[f"DEVICE REMOVED {_hexcode(m.group(1))}"] += 1
                continue
            m = RE_D3D12_PIPELINE_TRAP.search(line)
            if m:
                d3d12_unhandled[f"Device::slot{m.group(1)}({m.group(2)})"] += 1
                continue
            m = RE_D3D12_IID_PASS.search(line)
            if m:
                d3d12_unhandled[f"Device::slot{m.group(1)} (passes IID)"] += 1
                continue
            m = RE_D3D12_VIEW_SKIPPED.search(line)
            if m:
                d3d12_unhandled[f"Device::slot{_dec(m.group(1))} view skipped"] += 1
                continue
            m = RE_D3D12_SLOT133.search(line)
            if m:
                _, norm_hr = _is_failed_hresult(m.group(1))
                if norm_hr == "0x00000000":
                    slot133_outcomes[m.group(2).replace(" ", "_")] += 1
                    continue  # a handled placement; only an HRESULT other than 0 is a finding
                slot133_outcomes["failed"] += 1  # and falls through to the unhandled count below
            elif RE_D3D12_SLOT133_UNCOMMITTED.search(line):
                slot133_outcomes["failed"] += 1  # E_INVALIDARG for an uncommitted address, no HRESULT text
            m = RE_D3D12_FORWARDED_RESULT.search(line)
            if m and not _is_failed_hresult(m.group(1))[0]:
                continue  # forward_slot()/create_slot() telemetry of a call the desktop device implemented
            m = RE_D3D12_SLOT_CALL.search(line)
            if m:
                d3d12_unhandled[_slot_label(RE_SLOT.search(m.group(1)))] += 1
                continue

        # 5. x2w-dstorage failures & status HRESULTs
        if "x2w-dstorage" in line:
            m = RE_DSTORAGE_OUTCOME.search(line)
            if m:
                outcome = m.group(1).strip()
                if outcome in DSTORAGE_FAIL_OUTCOMES:
                    dstorage_failures[outcome] += 1
                elif outcome not in DSTORAGE_OK_OUTCOMES:
                    dstorage_failures[OTHER] += 1
                continue
            m = RE_DSTORAGE_ENTRY_FAIL.search(line)
            if m:
                dstorage_failures["entry failed"] += 1
                continue
            m = RE_DSTORAGE_ZLIB_FAIL.search(line)
            if m:
                dstorage_failures["zlib failure"] += 1
                continue
            m = RE_DSTORAGE_OPENFILE_FAIL.search(line)
            if m:
                fm = RE_OPENFILE_FAIL.match(m.group(1).strip())
                dstorage_failures["OpenFile: fail" + (f" (err {fm.group(1)})" if fm and fm.group(1) else "" if fm else " (other)")] += 1
                continue
            m = RE_DSTORAGE_HRESULT.search(line)
            if m:
                _, norm_hr = _is_failed_hresult(m.group(1))
                dstorage_hresults[norm_hr] += 1
                continue
            m = RE_DSTORAGE_SIGNAL.search(line)
            if m:
                failed, norm_hr = _is_failed_hresult(m.group(1))
                if failed:
                    dstorage_hresults[norm_hr] += 1
                continue

        # 6. x2w-runtime / x2w-api failing methods / HRESULTs
        if "x2w-runtime" in line or "x2w-api" in line:
            m = RE_RUNTIME_CALL.search(line)
            if m:
                failed, norm_hr = _is_failed_hresult(m.group(2))
                if failed:
                    runtime_failures[(f"slot{m.group(1)}", norm_hr)] += 1
                continue
            m = RE_RUNTIME_STORAGE.search(line)
            if m:
                _, norm_hr = _is_failed_hresult(m.group(1))
                runtime_failures[("persistent_local_storage", norm_hr)] += 1
                continue
            m = RE_RUNTIME_GAMESAVE.search(line)
            if m:
                runtime_failures[("game_save_update", f"status {_dec(m.group(1).lstrip('-')) if m.group(1).lstrip('-').isdigit() else '?'}")] += 1
                continue

    return {
        "slot70_refusals": slot70_refusals,
        "d3d12_unhandled": d3d12_unhandled,
        "slot133_outcomes": slot133_outcomes,
        "indirect_peaks": indirect_peaks,
        "gpu_budget": gpu_budget,
        "d3d12_debug": d3d12_debug,
        "dstorage_failures": dstorage_failures,
        "dstorage_hresults": dstorage_hresults,
        "runtime_failures": runtime_failures,
        "endpoint": endpoint,
    }


def summarize_shader_cache(cache_dir: Path | None) -> Counter[str]:
    """Count .fail Unsupported(...) / fxc reasons in X2W_SHADER_CACHE."""
    reasons: Counter[str] = Counter()
    if cache_dir is None or not cache_dir.is_dir():
        return reasons
    for fail_path in sorted(cache_dir.glob("*.fail")):
        try:
            text = fail_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first_line = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        if not first_line:
            first_line = "(empty .fail)"
        reason = "(empty .fail)" if first_line == "(empty .fail)" else classify_shader_reason(first_line)
        # A device slot 67 compute request (req_<key>.c67) gets its own class, so graphics counts do not move.
        if (cache_dir / ("req_" + fail_path.stem + ".c67")).is_file():
            reason = "compute: " + reason
        reasons[reason] += 1
    return reasons


def _format_table(
    headers: list[str],
    rows: list[list[object]],
    total_unique: int,
    total_events: int,
    max_items: int,
) -> list[str]:
    if not rows:
        return ["_None observed._", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows[:max_items]:
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    if total_unique > max_items:
        lines.append(
            f"_Showing top {max_items} of {total_unique} unique entries ({total_events} total events)._"
        )
    lines.append("")
    return lines


def _stop_text(stop: tuple[str, str] | None) -> str:
    return "not reported" if stop is None else f"`{stop[0]}` ({stop[1]}-chance)"


def format_endpoint(endpoint: RunEndpoint) -> list[str]:
    """Render the run end-point section."""
    peak = "not reported" if endpoint.peak_commit_mib is None else f"`{endpoint.peak_commit_mib}` MiB"
    beat = "not reported" if endpoint.last_heartbeat_s is None else f"`{endpoint.last_heartbeat_s}` s"
    out = [
        "### Run End-Point",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Peak commit (maximum of the `commit peak=` values) | {peak} |",
        f"| Last heartbeat | {beat} |",
        f"| Confirmed debugger stop | {_stop_text(endpoint.confirmed_stop)} |",
        f"| First observed exception | {_stop_text(endpoint.first_exception)} |",
        f"| Last observed exception | {_stop_text(endpoint.last_exception)} |",
        f"| Observed exceptions | `{endpoint.exception_events}` |",
        "",
    ]
    if endpoint.failure_second is None:
        out.extend(["_No failed-requests-by-cause heartbeat line._", ""])
        return out
    out.extend([f"Failed requests by cause at heartbeat second `{endpoint.failure_second}`:", ""])
    out.extend(["| Cause | Count |", "| --- | --- |"])
    for cause in FAILURE_CAUSES:
        count = endpoint.failure_counts.get(cause)
        out.append(f"| {cause} | {'?' if count is None else count} |")
    out.append("")
    return out


def format_markdown(
    summary: dict[str, Any],
    shader_failures: Counter[str],
    files_scanned: int,
    shader_cache_provided: bool,
    max_items: int,
) -> str:
    """Render aggregated counters as allowlisted Markdown."""
    out: list[str] = [
        "## Startup Run Triage Summary",
        "",
        f"- Log files scanned: `{files_scanned}`",
        f"- `slot70` memory SRV refusals: `{sum(summary['slot70_refusals'].values())}` "
        f"({len(summary['slot70_refusals'])} unique)",
        f"- `x2w-d3d12` / trap slot events: `{sum(summary['d3d12_unhandled'].values())}` "
        f"({len(summary['d3d12_unhandled'])} unique)",
        f"- `x2w-d3d12-debug` messages: `{sum(summary['d3d12_debug'].values())}` "
        f"({len(summary['d3d12_debug'])} unique)",
        f"- `x2w-dstorage` failures: `{sum(summary['dstorage_failures'].values())}` "
        f"({len(summary['dstorage_failures'])} unique)",
        f"- `x2w-runtime` failing calls: `{sum(summary['runtime_failures'].values())}` "
        f"({len(summary['runtime_failures'])} unique)",
    ]
    if shader_cache_provided:
        out.append(
            f"- `X2W_SHADER_CACHE` `.fail` shaders: `{sum(shader_failures.values())}` "
            f"({len(shader_failures)} unique reasons)"
        )
    out.append("")

    # 1. slot70 refusals
    out.extend(["### `Device::slot70` Memory SRV Refusals", ""])
    s70 = summary["slot70_refusals"]
    s70_rows = [
        [count, reason, dim, fmt, vdim, vfmt, layout, flags]
        for (reason, dim, fmt, vdim, vfmt, layout, flags), count in s70.most_common()
    ]
    out.extend(
        _format_table(
            ["Count", "Reason", "Res Dim", "Res Fmt", "View Dim", "View Fmt", "Layout", "Flags"],
            s70_rows,
            len(s70),
            sum(s70.values()),
            max_items,
        )
    )

    # 2. x2w-d3d12 unhandled/stubbed calls & traps
    out.extend(["### Unhandled / Stubbed `x2w-d3d12` & Trap Calls", ""])
    d3d = summary["d3d12_unhandled"]
    d3d_rows = [[count, f"`{call}`"] for call, count in d3d.most_common()]
    out.extend(
        _format_table(
            ["Count", "Call / Slot"],
            d3d_rows,
            len(d3d),
            sum(d3d.values()),
            max_items,
        )
    )

    # 2b. Device::slot133 placed resources, by outcome
    s133 = summary["slot133_outcomes"]
    if s133:
        out.extend(["### `Device::slot133` Placed Resources", ""])
        s133_rows = [[s133[name], name] for name in SLOT133_OUTCOMES if s133[name]]
        out.extend(_format_table(["Count", "Outcome"], s133_rows, len(s133_rows), sum(s133.values()), max_items))

    indirect = summary["indirect_peaks"]
    if indirect is not None:
        out.extend([
            "### `ExecuteIndirect` Outcomes",
            "",
            "Peak sampled cumulative count per cell. Rows and columns can peak in different samples.",
            "",
        ])
        rows = [[outcome, *(indirect[outcome, kind] for kind in INDIRECT_KINDS)] for outcome in INDIRECT_OUTCOMES]
        out.extend(_format_table(["Outcome", *INDIRECT_KINDS], rows, len(rows), 0, len(rows)))

    gpu_budget = summary["gpu_budget"]
    if gpu_budget.samples:
        out.extend([
            "### GPU Memory Budget",
            "",
            f"- Complete samples: `{gpu_budget.samples}`",
            f"- Samples above local budget: `{gpu_budget.above_local_budget}` of `{gpu_budget.known_local_budget}` "
            "with a positive reported local budget",
            "",
        ])
        rows = []
        for label, sample in (
            ("Highest local usage-to-budget ratio", gpu_budget.local_peak),
            ("Most retained texture bytes", gpu_budget.texture_peak),
        ):
            if sample is not None:
                rows.append([
                    label, *(sample.local if sample.local is not None else ("n/a", "n/a")),
                    sample.live_texture_bytes, sample.retained_texture_bytes, sample.retained_textures,
                ])
        headers = ["Sample", "Local usage bytes", "Local budget bytes", "Live texture bytes",
                   "Retained texture bytes", "Retained textures"]
        out.extend(_format_table(headers, rows, len(rows), 0, len(rows)))

    # 3. x2w-d3d12-debug
    dbg = summary["d3d12_debug"]
    if dbg:
        out.extend(["### `x2w-d3d12-debug` Messages", ""])
        dbg_rows = [[count, sev, msg_id] for (sev, msg_id), count in dbg.most_common()]
        out.extend(
            _format_table(
                ["Count", "Sev", "ID"],
                dbg_rows,
                len(dbg),
                sum(dbg.values()),
                max_items,
            )
        )

    # 4. x2w-dstorage failures & HRESULTs
    out.extend(["### `x2w-dstorage` Failures & Status HRESULTs", ""])
    ds_fail = summary["dstorage_failures"]
    ds_rows = [[count, reason] for reason, count in ds_fail.most_common()]
    out.extend(
        _format_table(
            ["Count", "Failure Reason"],
            ds_rows,
            len(ds_fail),
            sum(ds_fail.values()),
            max_items,
        )
    )
    ds_hr = summary["dstorage_hresults"]
    if ds_hr:
        hr_rows = [[count, f"`{hr}`"] for hr, count in ds_hr.most_common()]
        out.extend(
            _format_table(
                ["Count", "Status HRESULT"],
                hr_rows,
                len(ds_hr),
                sum(ds_hr.values()),
                max_items,
            )
        )

    # 5. x2w-runtime failing methods / HRESULTs
    out.extend(["### `x2w-runtime` Failing Methods", ""])
    rt = summary["runtime_failures"]
    rt_rows = [[count, f"`{method}`", f"`{hr}`"] for (method, hr), count in rt.most_common()]
    out.extend(
        _format_table(
            ["Count", "Method", "HRESULT / Status"],
            rt_rows,
            len(rt),
            sum(rt.values()),
            max_items,
        )
    )

    # 6. Run end-point
    out.extend(format_endpoint(summary["endpoint"]))

    # 7. X2W_SHADER_CACHE .fail reasons
    if shader_cache_provided:
        out.extend(["### `X2W_SHADER_CACHE` `.fail` Reasons", ""])
        sh_rows = [[count, f"`{reason}`"] for reason, count in shader_failures.most_common()]
        out.extend(
            _format_table(
                ["Count", "Reason"],
                sh_rows,
                len(shader_failures),
                sum(shader_failures.values()),
                max_items,
            )
        )

    return "\n".join(out)


def triage_run(
    logdir: Path,
    shader_cache: Path | None = None,
    max_items: int = 16,
) -> str:
    """Scan logdir (and optional shader_cache) and return an allowlisted Markdown summary."""
    log_files = iter_log_files(logdir)

    # run.ps1 / debug_launch.py capture stderr.txt while X2W_TRAP_LOG appends the same lines to trap.log:
    # each trap.log line already present in the sibling stderr.txt is skipped once, so repeats still count.
    mirrored: dict[Path, Counter[str]] = {}
    for path in log_files:
        if path.name.lower() == "trap.log" and (path.parent / "stderr.txt") in log_files:
            seen: Counter[str] = Counter()
            try:
                with (path.parent / "stderr.txt").open("r", encoding="utf-8", errors="replace") as handle:
                    seen.update(line.rstrip("\r\n") for line in handle)
            except OSError:
                pass
            mirrored[path] = seen

    def _line_stream() -> Iterable[str]:
        for path in log_files:
            budget = mirrored.get(path)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if budget is not None and budget[line.rstrip("\r\n")] > 0:
                            budget[line.rstrip("\r\n")] -= 1
                            continue
                        yield line
            except OSError:
                continue

    summary = summarize_lines(_line_stream())
    shader_failures = summarize_shader_cache(shader_cache)
    return format_markdown(
        summary=summary,
        shader_failures=shader_failures,
        files_scanned=len(log_files),
        shader_cache_provided=shader_cache is not None,
        max_items=max(1, max_items),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.logdir.exists():
        raise FileNotFoundError(f"log path not found: {args.logdir}")
    if not (args.logdir.is_file() or args.logdir.is_dir()):
        raise ValueError(f"log path must be a regular file or directory: {args.logdir}")
    if args.shader_cache is not None:
        if not args.shader_cache.exists():
            raise FileNotFoundError(f"shader cache directory not found: {args.shader_cache}")
        if not args.shader_cache.is_dir():
            raise NotADirectoryError(f"shader cache path is not a directory: {args.shader_cache}")
    if args.max_items < 1:
        raise ValueError(f"--max-items must be at least 1, got {args.max_items}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize startup logs and optional shader cache into allowlisted Markdown."
    )
    parser.add_argument("logdir", type=Path, help="Startup log directory or single log file")
    parser.add_argument(
        "--shader-cache",
        type=Path,
        default=None,
        help="Optional X2W_SHADER_CACHE directory containing *.fail files",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=16,
        help="Maximum rows per summary table (default: 16)",
    )
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        report = triage_run(args.logdir, shader_cache=args.shader_cache, max_items=args.max_items)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
