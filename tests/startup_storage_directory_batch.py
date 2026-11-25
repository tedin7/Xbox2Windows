"""Submit absolute RDS5 file offsets to the startup DirectStorage queue on Windows."""
import itertools
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from texture_descriptors import iter_boundaries, read_tables

PAGE = 4096
ALIGNMENT = 256
SHIFT = 0xe0000
DISTANT = SHIFT // ALIGNMENT
GUARD = b"\xa5" * 16


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def write_fixture(path, records, labels):
    # A raw ETPF interval, DISTANT + 1 individually compressed pages, and an end marker.
    streams = [zlib.compress(bytes([tag]) * PAGE, 9) for tag in labels]
    assert {len(stream) for stream in streams} == {28}
    write_compressed_fixture(path, records, streams)


def write_compressed_fixture(path, records, streams):
    assert len(streams) == 3 and all(0 < len(stream) <= ALIGNMENT for stream in streams)
    spans = [(streams[0], 1)] * (DISTANT + 1)
    spans[1] = streams[1], 1
    spans[-1] = streams[2], 1
    write_intervals(path, records, spans)


def write_intervals(path, records, spans):
    count = len(spans) + 2
    metadata_start = round_up(16 + count * 8, ALIGNMENT)
    metadata_size = round_up(16 + records * (16 + 8 + 32), PAGE)
    offset = metadata_start + metadata_size
    logical = metadata_size
    extent = offset + sum(round_up(len(stream), ALIGNMENT) for stream, _ in spans)
    logical_size = logical + sum(pages * PAGE for _, pages in spans)
    data = bytearray(extent)
    struct.pack_into("<4sIQ", data, 0, b"RDS5", count, logical_size)
    struct.pack_into("<4sIII", data, metadata_start, b"ETPF", 0x27a, records, records * 24)
    descriptor = struct.pack("<4I", 0, 0, metadata_size // PAGE, 0)
    data[metadata_start + 16:metadata_start + 16 + records * 16] = descriptor * records

    def boundary(index, stored, logical_position, codec=0, padding=0):
        entry = (stored // ALIGNMENT) | (codec << 26) | ((logical_position // PAGE) << 30) | (padding << 52)
        struct.pack_into("<Q", data, 16 + 8 * index, entry)

    boundary(0, metadata_start, 0)
    for index, (stream, pages) in enumerate(spans, start=1):
        aligned = round_up(len(stream), ALIGNMENT)
        boundary(index, offset, logical, 8, aligned - len(stream))
        data[offset:offset + len(stream)] = stream
        offset += aligned
        logical += pages * PAGE
    boundary(count - 1, offset, logical)
    path.write_bytes(data)


def ranges(path):
    with path.open("rb") as stream:
        table = read_tables(stream)
        entries = list(iter_boundaries(table))
    result = []
    for left, right in itertools.pairwise(entries[1:]):
        assert left.codec == 8 and right.logical - left.logical == PAGE
        result.append((left.stored, right.stored - left.stored - left.trailing_bytes))
    return result


def check(consumer, folder):
    cases = []
    labels = ((0x41, 0x43, 0x42), (0x51, 0x53, 0x52))
    for file, (name, count) in enumerate((("small", 1), ("large", 16384))):
        path = folder / f"{name}.ttpf"
        write_fixture(path, count, labels[file])
        segments = ranges(path)
        (first, length), (second, next_length) = segments[:2]
        assert first == (0x8100, 0xe8100)[file] and second == first + ALIGNMENT
        assert segments[DISTANT] == (first + SHIFT, length)
        assert second + SHIFT == path.stat().st_size and length == next_length == 28
        a, c, b = labels[file]
        for batch, offset, size, tag in (
                (file, first, length, a), (file, second, next_length, c),
                (2 + file, first + SHIFT, length, b),
                (2 + file, second + SHIFT, next_length, None),
                (2 + file, second, next_length, c)):
            cases.append(((batch, file, offset, size), tag))

    # Submit both files' ordinary batches back-to-back, followed by both shifted controls.
    cases.sort(key=lambda case: case[0][0])
    rows = "".join("%d %d %d %d\n" % spec for spec, _ in cases)
    env = dict(os.environ, X2W_XMEM_PERMISSIVE="1")
    run = subprocess.run([consumer, "dstorage-directory"], input=rows, text=True, cwd=folder, env=env,
                         capture_output=True, timeout=15)
    assert run.returncode == 0, run.stderr
    statuses = [int(line, 16) for line in run.stdout.splitlines()]
    data = (folder / "readback.bin").read_bytes()
    stride = PAGE + 2 * len(GUARD)
    assert len(statuses) == len(cases) and len(data) == len(cases) * stride
    for i, (spec, tag) in enumerate(cases):
        batch, file, offset, _ = spec
        page = bytes([0xa5 if tag is None else tag]) * PAGE
        actual = data[i * stride:(i + 1) * stride]
        assert statuses[i] == (0x80070026 if tag is None else 0), (spec, hex(statuses[i]))
        assert actual == GUARD + page + GUARD, (spec, "changed guard or wrong bytes")
        if batch >= 2 and offset == (0x8100, 0xe8100)[file] + SHIFT:
            assert actual[len(GUARD):-len(GUARD)] != bytes([labels[file][0]]) * PAGE
    print("2 files, 4 submissions, 10 guarded readbacks: 4 ordinary, 2 wrong valid streams, "
          "2 rejected at EOF, 2 correct reads after rejection")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        check(str(Path(sys.argv[1]).resolve()), Path(temporary))
