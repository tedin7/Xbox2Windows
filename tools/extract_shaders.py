#!/usr/bin/env python3
"""
Extract shaders from Xbox Scarlett shaders.rpf archive.

Extracts DXBC containers and raw AMD GFX1013 (RDNA 2) machine code binaries
organized by shader stage (pixel, vertex, compute, mesh, geometry).
"""

import argparse
from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from resource_inventory import parts
except ImportError:
    parts = None


STAGE_NAMES = {
    0: "pixel",
    1: "vertex",
    2: "geometry",
    3: "hull",
    4: "domain",
    5: "compute",
    6: "library",
    13: "mesh",
    14: "amplification",
}


MAX_STEM_BYTES = 128


def sanitize_filename(name: str) -> str:
    """Sanitize shader record name for valid Windows/Linux filesystem usage."""
    # Replace characters not allowed in filenames
    clean = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Keep room for a collision suffix and extension under the 255-byte limit of common filesystems.
    clean = clean.encode("utf-8")[:MAX_STEM_BYTES].decode("utf-8", "ignore")
    clean = clean.strip(". ")
    return clean if clean else "unnamed_shader"


def extract_native_rdna2_code(program: bytes) -> bytes | None:
    """Extract raw AMD GFX1013 / RDNA 2 machine code from DXIL XDAT/XBCH/FSSM sections."""
    try:
        table_idx = program.index(b"XDAT", 24)
    except ValueError:
        return None

    if table_idx + 12 > len(program):
        return None
    count, = struct.unpack_from("<I", program, table_idx + 4)
    if table_idx + 12 + count * 12 > len(program):
        return None

    entries = struct.iter_unpack("<3I", program[table_idx + 12:table_idx + 12 + count * 12])
    native_entries = [(o, n) for tag, o, n in entries if tag == 13]
    if not native_entries:
        return None

    offset, length = native_entries[0]
    table_end = table_idx + 12 + count * 12
    if offset < table_end or offset + length > len(program) or length < 24 or program[offset:offset + 4] != b"XBCH":
        return None

    size, = struct.unpack_from("<I", program, offset + 16)
    if not 196 <= size <= 16 << 20:
        return None
    decoder = zlib.decompressobj(15)
    try:
        image = decoder.decompress(program[offset + 24:offset + length], size + 1)
    except zlib.error:
        return None

    if len(image) != size or not decoder.eof or image[:4] != b"FSSM":
        return None

    start = struct.unpack_from("<I", image, 192)[0] + 40
    if start < 196 or start % 4 or start + 4 > len(image):
        return None

    # Locate s_endpgm (0xbf810000 in little-endian: \0\0\x81\xbf)
    ends = [i + 4 for i in range(start, len(image) - 3, 4) if image[i:i + 4] == b"\0\0\x81\xbf"]
    if not ends:
        return None

    return image[start:ends[-1]]


def _write_atomic(path: Path, content: str) -> None:
    """Write content to a sibling temp file and atomically replace path, leaving no temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def extract_shaders(
    archive_path: Path,
    output_dir: Path,
    target_stages: set[str] | None = None,
    limit: int | None = None,
    extract_format: str = "native",
    *,
    length_prefixed_names: bool = False,
):
    """
    Scan shaders.rpf and extract shaders into organized folders.
    """
    output_dir = output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "xbox2windows-shaders"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[!] Warning: Cannot write to '{output_dir}'. Falling back to '{fallback}'.", file=sys.stderr)
        output_dir = fallback

    def ensure_separate_output(path: Path) -> None:
        try:
            collision = path.samefile(archive_path)
        except FileNotFoundError:
            return
        if collision:
            raise ValueError(f"Output file would overwrite the input archive: {path}")

    manifest_path = output_dir / "manifest.json"
    ensure_separate_output(manifest_path)
    archive_size = archive_path.stat().st_size
    print(f"[*] Opening archive: {archive_path.name} ({archive_size / (1024*1024):.1f} MB)...")
    start_time = time.time()

    names_pattern = re.compile(rb"[!-~]{4,}")
    stage_counters = Counter()
    used_stems = set()
    extracted_records = []
    total_found = 0
    extracted_count = 0

    if archive_size > 0:
        with archive_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            data_len = len(data)
            cursor = 0

            while match := names_pattern.search(data, cursor):
                scan_start = cursor
                cursor = match.end()
                if cursor + 18 > data_len or data[cursor] != 0:
                    continue

                stored, _, repeated, decoded = struct.unpack_from("<4I", data, cursor + 2)
                if not (16 < stored <= 8 << 20 and 32 <= decoded <= 16 << 20 and stored == repeated and cursor + 18 + stored <= data_len):
                    continue

                start_payload = cursor + 18

                # Decompress raw-DEFLATE container within the declared decoded length
                stream = zlib.decompressobj(-15)
                try:
                    blob = stream.decompress(data[start_payload:start_payload + stored], decoded + 1)
                except zlib.error:
                    continue

                if len(blob) != decoded or not stream.eof:
                    continue

                # Parse DXBC sections
                if parts is None:
                    continue
                try:
                    sections = parts(blob)
                except ValueError:
                    continue

                program = sections.get(b"DXIL", b"")
                if len(program) < 24:
                    continue

                name = match.group()
                # The length includes the terminator. A printable length byte can join the name's run.
                if length_prefixed_names and name[0] == len(name):
                    name_start = match.start()
                    if name_start == scan_start or data[name_start - 1] != len(name) + 1:
                        name = name[1:]
                record_name = name.decode("ascii", "replace")
                cursor = start_payload + (stored - len(stream.unused_data))
                total_found += 1

                kind = struct.unpack_from("<H", program, 2)[0]
                stage = STAGE_NAMES.get(kind, f"stage_{kind}")
                stage_counters[stage] += 1

                # Filter by stage if requested
                if target_stages and stage not in target_stages and "all" not in target_stages:
                    continue

                # Extract native RDNA 2 code
                native_code = extract_native_rdna2_code(program)

                # Stage folder
                stage_dir = output_dir / stage
                stage_dir.mkdir(parents=True, exist_ok=True)

                safe_name = sanitize_filename(record_name)
                file_stem = safe_name
                suffix = 1
                while (stage, file_stem.casefold()) in used_stems:
                    file_stem = f"{safe_name}_{suffix}"
                    suffix += 1
                used_stems.add((stage, file_stem.casefold()))

                record_info = {
                    "name": record_name,
                    "stage": stage,
                    "dxbc_size": len(blob),
                    "native_size": len(native_code) if native_code else 0,
                }

                # Save formats
                if extract_format in ("dxbc", "both"):
                    dxbc_path = stage_dir / f"{file_stem}.dxbc"
                    ensure_separate_output(dxbc_path)
                    with dxbc_path.open("wb") as out_f:
                        out_f.write(blob)
                    record_info["dxbc_file"] = f"{stage}/{file_stem}.dxbc"

                if extract_format in ("native", "both"):
                    if native_code:
                        bin_path = stage_dir / f"{file_stem}.bin"
                        ensure_separate_output(bin_path)
                        with bin_path.open("wb") as out_f:
                            out_f.write(native_code)
                        record_info["native_file"] = f"{stage}/{file_stem}.bin"
                        record_info["sha256"] = hashlib.sha256(native_code).hexdigest()
                    elif extract_format == "native":
                        # Fallback to dxbc if native code not unpackable
                        dxbc_path = stage_dir / f"{file_stem}.dxbc"
                        ensure_separate_output(dxbc_path)
                        with dxbc_path.open("wb") as out_f:
                            out_f.write(blob)
                        record_info["dxbc_file"] = f"{stage}/{file_stem}.dxbc"

                extracted_records.append(record_info)
                extracted_count += 1

                if extracted_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"    [+] Extracted {extracted_count} shaders ({elapsed:.1f}s)...")

                if limit is not None and extracted_count >= limit:
                    print(f"[*] Reached limit of {limit} shaders.")
                    break

    # Save manifest.json
    manifest_data = {
        "archive": archive_path.name,
        "total_extracted": extracted_count,
        "format": extract_format,
        "stage_distribution": dict(stage_counters),
        "shaders": extracted_records,
    }
    _write_atomic(manifest_path, json.dumps(manifest_data, indent=2))

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(" EXTRACTION COMPLETED ")
    print("=" * 60)
    print(f"Total Shaders Found:     {total_found}")
    print(f"Total Shaders Extracted: {extracted_count}")
    print(f"Time Elapsed:            {elapsed:.2f} seconds")
    print("\nExtracted by Stage:")
    for stg, cnt in stage_counters.most_common():
        print(f"  * {stg:<15}: {cnt:>5}")
    print(f"\n[+] Shaders directory: {output_dir}")
    print(f"[+] Manifest file:     {manifest_path}")
    print("=" * 60)

    return output_dir, manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to shaders.rpf file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shaders"),
        help="Target folder for extracted shaders (default: shaders)",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", *STAGE_NAMES.values()],
        help="Filter by shader stage (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of shaders to extract (e.g. 50 or 500)",
    )
    parser.add_argument(
        "--format",
        choices=["native", "dxbc", "both"],
        default="native",
        help="Extraction format: 'native' (raw AMD RDNA 2 .bin, default), 'dxbc' (DXBC container), or 'both'",
    )
    parser.add_argument(
        "--length-prefixed-names",
        action="store_true",
        help="Interpret a matching printable byte as a record name's length prefix",
    )

    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        archive_path = args.archive.resolve()
        output_dir = args.output_dir.resolve()
        if not args.archive.exists():
            raise FileNotFoundError(f"File not found: {args.archive}")
        if not args.archive.is_file():
            raise ValueError(f"Archive path is not a file: {args.archive}")
        if output_dir == archive_path:
            raise ValueError(f"Output directory must not match archive path: {args.output_dir}")
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {args.output_dir}")
        extract_shaders(
            archive_path=args.archive,
            output_dir=args.output_dir,
            target_stages=set(args.stage),
            limit=args.limit,
            extract_format=args.format,
            length_prefixed_names=args.length_prefixed_names,
        )
    except (OSError, ValueError, struct.error, zlib.error) as exc:
        parser.exit(1, f"[-] Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
