"""Inspect a candidate as the manifest's Scarlett executable, using only file reads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import pefile

from dependency_paths import trace
from import_routing import annotate

from pe_inventory import changes, import_tables, scan
from verify_build import read_layout

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_routes(image_bytes, caller):
    """Return candidate route rows for every normal and delay import."""
    with pefile.PE(data=image_bytes, fast_load=True) as pe:
        imports = import_tables(pe, caller)
    return [annotate({"image": caller, "kind": kind, "dll": dll, "symbol": symbol}, "dll")
            for kind, dll, symbol in imports]


def routing_failures(rows):
    """Return annotated imports whose module has no candidate route."""
    return [row for row in rows if row.get("route") == "unsupported"]


def inside(root, name):
    """Resolve a package-relative path, normalizing root so unnormalized '..' in root does not reject valid names."""
    base = root.resolve()
    path = (base / name).resolve()
    if (not path.is_relative_to(base) or path == base or
            Path(name).is_absolute() or str(name).startswith(("/", "\\"))):
        raise ValueError(f"Invalid package path: {name}")
    return path


def describe_startup(pe, locations, read_rva):
    """Look up x64 runtime-function intervals for the recorded startup addresses."""
    directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY
    table = directory[3] if len(directory) > 3 else None
    by_rva = sorted({item["rva"] for item in locations})
    matches = {}
    count = 0
    status = "absent"
    if table and (table.VirtualAddress or table.Size):
        try:
            if not table.VirtualAddress or not table.Size or table.Size % 12:
                raise ValueError("Invalid x64 exception directory extent")
            records = read_rva(table.VirtualAddress, table.Size)
            cursor = previous_end = 0
            for begin, end, unwind in struct.iter_unpack("<III", records):
                if (begin < previous_end or not begin < end <= pe.OPTIONAL_HEADER.SizeOfImage
                        or not 0 < unwind < pe.OPTIONAL_HEADER.SizeOfImage):
                    raise ValueError("Invalid or overlapping x64 function interval")
                previous_end = end
                count += 1
                while cursor < len(by_rva) and by_rva[cursor] < end:
                    target = by_rva[cursor]
                    if begin <= target:
                        matches[target] = dict(begin_rva=begin, end_rva=end, unwind_data_rva=unwind)
                    cursor += 1
            status = "indexed"
        except ValueError as exc:
            # Keep independent entry and TLS evidence for nonstandard tables.
            return dict(exception_table="unusable", detail=str(exc), locations=locations)
    for item in locations:
        item["runtime_function"] = matches.get(item["rva"])
    return dict(exception_table=status, runtime_functions=count, locations=locations)


def inspect_executable(path):
    size = path.stat().st_size
    with pefile.PE(str(path), fast_load=True) as pe:
        hdr = pe.OPTIONAL_HEADER
        if (pe.FILE_HEADER.Machine != 0x8664 or hdr.Magic != 0x20B or
                not pe.FILE_HEADER.Characteristics & 2 or pe.FILE_HEADER.Characteristics & 0x2000):
            raise ValueError("Expected an x64 PE32+ executable")
        if (not pe.sections or len(pe.sections) != pe.FILE_HEADER.NumberOfSections or
                hdr.SizeOfHeaders < pe.sections[-1].get_file_offset() + 40 or hdr.SizeOfHeaders > size):
            raise ValueError("Incomplete PE headers or section table")
        for section in pe.sections:
            if section.SizeOfRawData and section.PointerToRawData + section.SizeOfRawData > size:
                raise ValueError("Section extends beyond end of file")

        def section_span(section):
            """Bytes the loader maps for a section: VirtualSize, else raw size, inside the image."""
            span = section.Misc_VirtualSize or section.SizeOfRawData
            return min(span, max(0, hdr.SizeOfImage - section.VirtualAddress))

        def section_for(rva, length, *, executable=False):
            for s in pe.sections:
                offset = rva - s.VirtualAddress
                if (0 <= offset and offset + length <= section_span(s) and
                        rva + length <= hdr.SizeOfImage and
                        (not executable or s.Characteristics & 0x20000000)):
                    return s
            raise ValueError(f"Unmapped or unsuitable RVA {rva:#x}")

        def read_rva(rva, length):
            section = section_for(rva, length)
            offset = rva - section.VirtualAddress
            # A zero raw pointer leaves the entire section zero-filled, as in pe_map.
            raw_size = section.SizeOfRawData if section.PointerToRawData else 0
            file_backed = min(length, max(0, raw_size - offset))
            data = section.get_data(rva, file_backed) if file_backed else b""
            if len(data) != file_backed:
                raise ValueError(f"Truncated data at RVA {rva:#x}")
            return data + bytes(length - file_backed)

        entry = hdr.AddressOfEntryPoint
        entry_section = section_for(entry, 1, executable=True) if entry else None
        tls = None
        if len(hdr.DATA_DIRECTORY) > 9:
            d = hdr.DATA_DIRECTORY[9]
            if d.VirtualAddress:
                start, end, index, callbacks, zero_fill, flags = struct.unpack(
                    "<4Q2I", read_rva(d.VirtualAddress, 40))
                callback_rvas = []
                if callbacks:
                    table = callbacks - hdr.ImageBase
                    section = section_for(table, 8)
                    slots = (section.VirtualAddress + section_span(section) - table) // 8
                    for i in range(min(slots, 65536)):
                        va, = struct.unpack("<Q", read_rva(table + i * 8, 8))
                        if not va:
                            break
                        rva = va - hdr.ImageBase
                        section_for(rva, 1, executable=True)
                        callback_rvas.append(rva)
                    else:
                        raise ValueError("Unterminated TLS callback table or inspection limit reached")
                tls = dict(directory_rva=d.VirtualAddress, template_start_va=start, template_end_va=end,
                           index_va=index, callback_table_va=callbacks, callbacks_rva=callback_rvas,
                           zero_fill=zero_fill, characteristics=flags)
        def location(role, rva):
            section = section_for(rva, 1, executable=True)
            return dict(role=role, rva=rva, section=section.Name.rstrip(b"\0").decode("ascii", "replace"),
                        file_offset=section.PointerToRawData + rva - section.VirtualAddress)

        locations = []
        if tls:
            locations.extend(location("tls_callback", rva) for rva in tls["callbacks_rva"])
        if entry:
            locations.append(location("entry", entry))
        return dict(machine="AMD64", format="PE32+", image_base=hdr.ImageBase, entry_rva=entry,
                    entry_section=entry_section.Name.rstrip(b"\0").decode("ascii", "replace")
                    if entry_section else None, tls=tls, startup=describe_startup(pe, locations, read_rva))


def collect(root, candidate, baseline, integrity, report):
    config = ET.parse(root / "MicrosoftGame.Config")
    slots = [e.attrib["Name"].replace("\\", "/") for e in config.iter("Executable")
             if e.get("TargetDeviceFamily", "").casefold() == "scarlett"]
    if len(slots) != 1:
        raise ValueError("Expected one Scarlett executable in MicrosoftGame.Config")
    slot, = slots
    installed = inside(root, slot)
    previous = json.loads(integrity.read_text(encoding="utf-8"))
    if previous["directory"] != root.name or not previous["torrent_payload_passed"]:
        raise ValueError("No passing historical integrity report for this build")
    torrent = inside(root, previous["torrent"])
    if digest(torrent) != previous["torrent_sha256"]:
        raise ValueError("Torrent hash differs from historical verification")
    _, targets = read_layout(inside(root, previous["layout"]["source"]))
    if slot not in targets or targets[slot] != previous["layout"]["absent_from_torrent"].get(slot):
        raise ValueError("Manifest slot and verified missing torrent entry disagree")
    old = json.loads(baseline.read_text(encoding="utf-8"))
    if old.get("schema") != 1 or any(p.casefold() == slot.casefold() for p in old["images"]):
        raise ValueError("Expected pre-EXE inventory")
    report["package"] = dict(build=root.name, slot=slot, chunks=targets[slot],
                             torrent_sha256=previous["torrent_sha256"], old_pieces=previous["pieces"],
                             verified_utc=previous.get("finished_utc"), candidate_covered_by_torrent=False,
                             baseline_sha256=digest(baseline), baseline_images=len(old["images"]))
    path = candidate if candidate is not None else installed
    report["candidate"] = {"path": str(path)}
    if not path.exists() and candidate is None:
        report["status"] = "awaiting_executable"
        return 3
    report["candidate"].update(size=path.stat().st_size, sha256=digest(path))
    report["pe"] = inspect_executable(path)
    current = scan(root, extra_image=(slot, path))
    candidate_key = next(k for k in current["images"] if k.casefold() == slot.casefold())
    if current["images"][candidate_key]["sha256"] != report["candidate"]["sha256"]:
        raise ValueError("Candidate changed during intake")
    for row in current["imports"]:
        annotate(row, "dll")
    for row in current["unresolved_edges"]:
        annotate(row, "dll")
    report["diff"] = changes(old, current)
    report["dependency_paths"] = trace(current, candidate_key)
    previous_missing = {(r["dll"], r["symbol"]) for r in old["unresolved_edges"]}
    report["entry_import_gaps"] = [
        annotate({"dll": r["dll"], "symbol": r["symbol"], "kind": r["kind"],
                  "seen_unresolved_in_baseline": (r["dll"], r["symbol"]) in previous_missing}, "dll")
        for r in current["unresolved_edges"] if r["image"] == candidate_key
    ]
    report["inventory"] = current
    report["status"] = "inspected"
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, nargs="?", help="Separate file, or omit to use the manifest path")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--baseline", type=Path, default=PROJECT / "out/dependencies/pre-exe.json")
    parser.add_argument("--integrity", type=Path)
    parser.add_argument("--reports", type=Path, default=PROJECT / "out/intake")
    parser.add_argument("--require-routing", action="store_true",
                        help="return 1 and name each reported import without a candidate module route")
    args = parser.parse_args(argv)
    if args.build is None:
        builds = list((PROJECT / "game-files").glob("*/MicrosoftGame.Config"))
        if len(builds) != 1:
            parser.error("Use --build to select a package")
        args.build = builds[0].parent
    try:
        root = args.build.resolve()
        reports = args.reports.resolve()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if reports.is_relative_to(root) or reports.is_relative_to(PROJECT / "game-files"):
        parser.error("Choose a report directory outside game files")
    integrity = args.integrity or PROJECT / "out/integrity" / (root.name + ".json")
    report = {"schema": 1, "status": "error"}
    try:
        code = collect(root, args.candidate.resolve() if args.candidate else None,
                       args.baseline, integrity, report)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ET.ParseError, pefile.PEFormatError, struct.error) as exc:
        report["error"] = str(exc)
        code = 2
    if code == 0 and args.require_routing:
        failures = routing_failures(report.get("inventory", {}).get("imports", []))
        for row in failures:
            print(f"unsupported: {row['image']} -> {row['dll']}!{row['symbol']}", file=sys.stderr)
        if failures:
            code = 1
    tmp_path = None
    try:
        reports.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", suffix=".tmp", dir=reports, encoding="utf-8", delete=False) as f:
            tmp_path = Path(f.name)
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        dest = tmp_path.with_suffix("." + report["status"] + ".json")
        os.replace(tmp_path, dest)
        tmp_path = None
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: failed to write report in {reports}: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    print(f"{report['status']} | {dest}")
    if "error" in report:
        print(report["error"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
