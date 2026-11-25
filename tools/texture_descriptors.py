"""Inspect ETPF texture descriptors in the first raw range of an RDS5 archive."""
import argparse
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import pairwise
import json
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO, NamedTuple
import zlib

from paged_resource import PagedResource, Segment


class Descriptor(NamedTuple):
    width: int
    height: int
    dxgi_format: int
    mip_marker: int


class Boundary(NamedTuple):
    stored: int
    logical: int
    codec: int
    trailing_bytes: int


class Tables(NamedTuple):
    archive: PagedResource
    metadata: PagedResource
    boundaries: int
    records: int
    logical_size: int
    version: int


class PageMap(NamedTuple):
    logical_offset: int
    file_offset: int


@dataclass(frozen=True)
class ZlibSegment:
    segment_logical_offset: int
    page_offset_in_segment: int
    segment: Segment


def boundary(data: bytes, offset: int = 0) -> Boundary:
    low, high = struct.unpack_from("<II", data, offset)
    return Boundary((low & 0x03ffffff) * 256,
                    ((high & 0xfffff) * 4 + (low >> 30)) * 4096,
                    (low >> 26) & 15, high >> 20)


def read_tables(stream: BinaryIO) -> Tables:
    size = stream.seek(0, 2)
    if size < 32:
        raise ValueError("Truncated RDS5 header")
    archive = PagedResource(stream, [Segment(0, size, size, "raw")])
    header = archive.read(0, 32)
    magic, count, size_field = struct.unpack_from("<4sIQ", header)
    logical_size = size_field & ((1 << 40) - 1)
    if magic != b"RDS5" or count < 2:
        raise ValueError("Expected an RDS5 header and at least two boundaries")
    first = boundary(header, 16)
    if first.logical or first.codec:
        raise ValueError("Expected raw metadata at logical offset zero")
    table_end = 16 + count * 8
    if not table_end <= first.stored < size:
        raise ValueError("RDS5 boundary table exceeds the header area")
    last = boundary(archive.read(table_end - 8, 8))
    if last.stored > size or last.logical != logical_size:
        raise ValueError("RDS5 terminal boundary disagrees with archive extents")
    second = boundary(header, 24)
    if not first.stored < second.stored <= last.stored or not 0 < second.logical <= logical_size:
        raise ValueError("Invalid first RDS5 range")

    # The first stored range can include alignment padding beyond its logical end.
    extent = min(second.stored - first.stored, second.logical)
    metadata = PagedResource(stream, [Segment(first.stored, extent, extent, "raw")])
    tag, version, records = struct.unpack("<4sII4x", metadata.read(0, 16))
    if tag != b"ETPF":
        raise ValueError("Expected ETPF at the first RDS5 boundary")
    table_bytes = records * 16
    if table_bytes > min(1 << 20, extent - 16):
        raise ValueError("ETPF descriptor table exceeds its first range or 1 MiB budget")
    return Tables(archive, metadata, count, records, logical_size, version)


def descriptor_indices(stream: BinaryIO, key: int) -> list[int]:
    if not 0 <= key < 1 << 64:
        raise ValueError("Texture key must be an unsigned 64-bit integer")
    tables = read_tables(stream)
    if tables.version != 0x27a:
        raise ValueError("Unsupported ETPF key-table version")
    keys_start = 16 + tables.records * 16
    keys_size = tables.records * 8
    if keys_start + keys_size > tables.metadata.size:
        raise ValueError("ETPF key table exceeds its first range")
    matches: list[int] = []
    for offset in range(0, keys_size, 4096):
        batch = tables.metadata.read(keys_start + offset, min(4096, keys_size - offset))
        for index, (candidate,) in enumerate(struct.iter_unpack("<Q", batch), start=offset // 8):
            if candidate == key:
                matches.append(index)
    return matches


def page_address(location: int, logical_size: int) -> int:
    address = (location & 0x1fffff) * 4096
    if address >= logical_size:
        raise ValueError("ETPF descriptor page outside the declared logical archive")
    return address


def descriptor_fields(shape: int, flags: int) -> Descriptor:
    return Descriptor((shape & 0x3fff) + 1, ((shape >> 14) & 0x3fff) + 1,
                      (flags >> 3) & 127, flags >> 24)


def descriptor_offset(tables: Tables, index: int) -> int:
    if not 0 <= index < tables.records:
        raise ValueError("Descriptor index outside the ETPF table")
    return 16 + index * 16


def describe_descriptor(stream: BinaryIO, index: int) -> dict[str, int]:
    tables = read_tables(stream)
    offset = descriptor_offset(tables, index)
    shape, flags, location = struct.unpack("<III4x", tables.metadata.read(offset, 16))
    return {
        "descriptor_index": index,
        **descriptor_fields(shape, flags)._asdict(),
        "logical_offset": page_address(location, tables.logical_size),
    }


def inventory(stream: BinaryIO) -> Counter[Descriptor]:
    tables = read_tables(stream)
    table_bytes = tables.records * 16
    counts: Counter[Descriptor] = Counter()
    for offset in range(0, table_bytes, 4096):
        batch = tables.metadata.read(16 + offset, min(4096, table_bytes - offset))
        for shape, flags, location in struct.iter_unpack("<III4x", batch):
            page_address(location, tables.logical_size)
            counts[descriptor_fields(shape, flags)] += 1
    return counts


def iter_boundaries(tables: Tables) -> Iterator[Boundary]:
    table_bytes = tables.boundaries * 8
    if table_bytes > 1 << 20:
        raise ValueError("RDS5 boundary table exceeds 1 MiB mapping budget")
    for offset in range(0, table_bytes, 4096):
        batch = tables.archive.read(16 + offset, min(4096, table_bytes - offset))
        for position in range(0, len(batch), 8):
            yield boundary(batch, position)


def descriptor_interval(tables: Tables, index: int) -> tuple[int, Boundary, Boundary]:
    offset = descriptor_offset(tables, index)
    location, = struct.unpack("<I", tables.metadata.read(offset + 8, 4))
    address = page_address(location, tables.logical_size)
    selected: tuple[Boundary, Boundary] | None = None
    for left, right in pairwise(iter_boundaries(tables)):
        if right.stored <= left.stored or right.logical <= left.logical:
            raise ValueError("RDS5 boundaries must increase in both address spaces")
        if left.logical <= address < right.logical:
            selected = left, right
    if selected is None:
        raise ValueError("Descriptor page has no RDS5 range")
    return address, *selected


def raw_page_map(stream: BinaryIO, index: int) -> PageMap:
    tables = read_tables(stream)
    address, left, right = descriptor_interval(tables, index)
    if left.codec:
        raise ValueError("Descriptor page belongs to a coded RDS5 range")
    payload_end = left.stored + (right.logical - left.logical)
    if right.stored != payload_end:
        if right.stored - payload_end != left.trailing_bytes:
            raise ValueError("Raw interval size disagrees with its declared padding")
        if any(tables.archive.read(payload_end, left.trailing_bytes)):
            raise ValueError("Nonzero bytes in declared raw interval padding")
    return PageMap(address, left.stored + address - left.logical)


def zlib_segment(stream: BinaryIO, index: int) -> ZlibSegment:
    tables = read_tables(stream)
    address, left, right = descriptor_interval(tables, index)
    if left.codec != 8:
        raise ValueError("Descriptor page requires a zlib RDS5 range")
    stored_span = right.stored - left.stored
    decoded_size = right.logical - left.logical
    if max(stored_span, decoded_size) > 1 << 20:
        raise ValueError("Zlib interval exceeds 1 MiB stored or decoded budget")
    data = tables.archive.read(left.stored, stored_span)
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(data, decoded_size + 1)
    except zlib.error as exc:
        raise ValueError("Invalid zlib stream or checksum") from exc
    if not decoder.eof or len(decoded) != decoded_size:
        raise ValueError("Incomplete zlib stream or mismatched logical range size")
    if any(decoder.unused_data):
        raise ValueError("Nonzero bytes follow the zlib stream")
    stored = stored_span - len(decoder.unused_data)
    segment = Segment(left.stored, stored, decoded_size, "zlib")
    return ZlibSegment(left.logical, address - left.logical, segment)


def main(argv: list[str] | None = None) -> int:
    """Run the texture descriptor CLI and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", metavar="archive", type=Path, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--descriptor", type=int, metavar="INDEX",
                      help="Report the metadata and logical page offset of a zero-based descriptor")
    mode.add_argument("--raw-page-map", type=int, metavar="INDEX",
                      help="Report offsets of a zero-based descriptor's first 4 KiB page in a verified raw range")
    mode.add_argument("--zlib-segment", type=int, metavar="INDEX",
                      help="Verify the zlib interval containing the descriptor's first page and report its extents")
    mode.add_argument("--lookup-key", metavar="KEY",
                      help="Find all descriptor indices for a decimal or 0x-prefixed 64-bit texture key")
    args = parser.parse_args(argv)
    archive = args.archives[0]
    try:
        for index in (args.descriptor, args.raw_page_map, args.zlib_segment):
            if index is not None and index < 0:
                raise ValueError("Descriptor index outside the ETPF table (must be nonnegative)")
        if args.lookup_key is not None:
            key = int(args.lookup_key, 16 if args.lookup_key.lower().startswith("0x") else 10)
            seen: set[tuple[int, int]] = set()
            matches: list[dict[str, str | int]] = []
            for archive in args.archives:
                with archive.open("rb") as file:
                    stat = os.fstat(file.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    if identity in seen:
                        raise ValueError("Duplicate archive input")
                    seen.add(identity)
                    matches.extend({"archive": str(archive), "descriptor_index": index}
                                   for index in descriptor_indices(file, key))
            json.dump({"matches": matches}, sys.stdout, indent=2)
            print()
        else:
            if len(args.archives) != 1:
                parser.error("Multiple archives require --lookup-key")
            with archive.open("rb") as file:
                if args.descriptor is not None:
                    print(json.dumps(describe_descriptor(file, args.descriptor), indent=2))
                elif args.zlib_segment is not None:
                    print(json.dumps(asdict(zlib_segment(file, args.zlib_segment)), indent=2))
                elif args.raw_page_map is None:
                    counts = inventory(file)
                    print(json.dumps({"descriptor_count": sum(counts.values()), "groups": [
                        {**item._asdict(), "count": counts[item]} for item in sorted(counts)
                    ]}, indent=2))
                else:
                    print(json.dumps(raw_page_map(file, args.raw_page_map)._asdict(), indent=2))
    except (OSError, ValueError, struct.error, zlib.error) as exc:
        parser.exit(1, f"{archive}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
