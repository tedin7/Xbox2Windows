#!/usr/bin/env python3
"""Check tools/texture_descriptors.py: boundary(), read_tables(), describe_descriptor(), inventory(),
raw_page_map(), zlib_segment(), descriptor_indices(), JSON CLI modes, and error exits without tracebacks."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "texture_descriptors.py"
sys.path.insert(0, str(ROOT / "tools"))
from paged_resource import PagedResource
import texture_descriptors as td


def _pack_boundary(stored: int, logical: int, codec: int = 0, trailing_bytes: int = 0) -> bytes:
    assert stored % 256 == 0 and logical % 4096 == 0
    stored_units = stored // 256
    logical_pages = logical // 4096
    low = (stored_units & 0x03ffffff) | ((codec & 15) << 26) | ((logical_pages & 3) << 30)
    high = ((logical_pages >> 2) & 0xfffff) | ((trailing_bytes & 0xfff) << 20)
    return struct.pack("<II", low, high)


def _shape(width: int, height: int) -> int:
    return ((height - 1) << 14) | (width - 1)


def _flags(dxgi_format: int, mip_marker: int) -> int:
    return (mip_marker << 24) | (dxgi_format << 3)


def _build_mixed_archive(*, key_version: int = 0x27a, keys: tuple[int, ...] = (0x1122334455667788, 0x99aabbccddeeff00),
                         raw_padding: int = 512) -> tuple[bytearray, bytes, bytes]:
    """Build a synthetic RDS5/ETPF archive with one raw range (padded) and one zlib range."""
    raw_payload = bytes((i * 11 + 3) % 256 for i in range(4096))
    zlib_plain = b"ZLIB_PAGE_0_" + b"A" * 4084 + b"ZLIB_PAGE_1_" + b"B" * 4084
    zlib_packed = zlib.compress(zlib_plain)

    # Range 0: metadata [256, 4352) -> logical [0, 4096)
    # Range 1: raw [4352, 4352 + 4096 + raw_padding) -> logical [4096, 8192)
    # Range 2: zlib [raw_end, raw_end + zlib_slot) -> logical [8192, 16384)
    raw_start = 4352
    raw_end = raw_start + 4096 + raw_padding
    zlib_slot = ((len(zlib_packed) + 255) // 256) * 256
    total_size = raw_end + zlib_slot

    archive = bytearray(total_size)
    struct.pack_into("<4sIQ", archive, 0, b"RDS5", 4, 16384)
    archive[16:24] = _pack_boundary(256, 0, codec=0, trailing_bytes=0)
    archive[24:32] = _pack_boundary(raw_start, 4096, codec=0, trailing_bytes=raw_padding)
    archive[32:40] = _pack_boundary(raw_end, 8192, codec=8, trailing_bytes=0)
    archive[40:48] = _pack_boundary(total_size, 16384, codec=0, trailing_bytes=0)

    # Two descriptors: index 0 in padded raw page (4096), index 1 in second page of zlib segment (12288).
    records = len(keys)
    struct.pack_into("<4sIII", archive, 256, b"ETPF", key_version, records, 0)
    struct.pack_into("<4I", archive, 272, _shape(64, 32), _flags(71, 3), 1, 0)
    struct.pack_into("<4I", archive, 288, _shape(128, 16), _flags(98, 1), 3, 0)
    struct.pack_into(f"<{records}Q", archive, 272 + records * 16, *keys)

    archive[raw_start:raw_start + 4096] = raw_payload
    archive[raw_end:raw_end + len(zlib_packed)] = zlib_packed
    return archive, raw_payload, zlib_plain


def _expect_raises(exc_type, fn, fragment=""):
    try:
        fn()
    except exc_type as exc:
        if fragment:
            assert fragment in str(exc), str(exc)
        return exc
    raise AssertionError(f"expected {exc_type.__name__} containing {fragment!r}")


def _expect_exit(argv, expected_code=1, fragment=""):
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            td.main(argv)
    except SystemExit as exc:
        assert exc.code == expected_code, (exc.code, stderr.getvalue())
    else:
        raise AssertionError(f"main({argv!r}) did not raise SystemExit")
    err = stderr.getvalue()
    assert err.strip(), f"expected error message on stderr for {argv!r}"
    assert "Traceback" not in err, err
    if fragment:
        assert fragment in err, err
    return err


def check_positive_api():
    """Verify boundary(), read_tables(), describe_descriptor(), inventory(), raw_page_map(), zlib_segment(), descriptor_indices()."""
    packed_b = _pack_boundary(4352, 12288, codec=8, trailing_bytes=512)
    b = td.boundary(b"\xaa\xbb" + packed_b, offset=2)
    assert b == td.Boundary(stored=4352, logical=12288, codec=8, trailing_bytes=512), b

    archive, raw_payload, zlib_plain = _build_mixed_archive()
    stream = io.BytesIO(archive)

    tables = td.read_tables(stream)
    assert (tables.boundaries, tables.records, tables.logical_size, tables.version) == (4, 2, 16384, 0x27a)

    desc0 = td.describe_descriptor(stream, 0)
    assert desc0 == {"descriptor_index": 0, "width": 64, "height": 32, "dxgi_format": 71,
                     "mip_marker": 3, "logical_offset": 4096}, desc0
    desc1 = td.describe_descriptor(stream, 1)
    assert desc1 == {"descriptor_index": 1, "width": 128, "height": 16, "dxgi_format": 98,
                     "mip_marker": 1, "logical_offset": 12288}, desc1

    counts = td.inventory(stream)
    assert dict(counts) == {
        td.Descriptor(64, 32, 71, 3): 1,
        td.Descriptor(128, 16, 98, 1): 1,
    }, counts

    rpm = td.raw_page_map(stream, 0)
    assert rpm == td.PageMap(logical_offset=4096, file_offset=4352), rpm
    assert archive[rpm.file_offset:rpm.file_offset + 4096] == raw_payload

    zseg = td.zlib_segment(stream, 1)
    assert zseg.segment_logical_offset == 8192
    assert zseg.page_offset_in_segment == 4096
    assert zseg.segment.codec == "zlib" and zseg.segment.size == 8192
    decoded_page = PagedResource(stream, [zseg.segment]).read(zseg.page_offset_in_segment, 4096)
    assert decoded_page == zlib_plain[4096:]

    assert td.descriptor_indices(stream, 0x99aabbccddeeff00) == [1]
    assert td.descriptor_indices(stream, 0xdeadbeef) == []
    return archive, desc0, rpm, zseg


def check_negative_api(archive: bytearray):
    """Pin exact ValueError/struct.error rejections on malformed headers, padding, boundaries, zlib, and keys."""
    # Truncated / empty RDS5 header and short boundary unpack.
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(b"")), "Truncated RDS5 header")
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(b"RDS5" + bytes(12))), "Truncated RDS5 header")
    _expect_raises(struct.error, lambda: td.boundary(b"\x00" * 4))

    # Bad RDS5 / ETPF magic.
    bad_rds = archive.copy()
    bad_rds[:4] = b"NOPE"
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(bad_rds)), "RDS5 header")

    bad_etpf = archive.copy()
    bad_etpf[256:260] = b"NOPE"
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(bad_etpf)), "Expected ETPF")

    # Non-raw metadata at logical offset zero (codec != 0 or logical != 0).
    coded_meta = archive.copy()
    coded_meta[16:24] = _pack_boundary(256, 0, codec=8, trailing_bytes=0)
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(coded_meta)), "raw metadata")
    shifted_meta = archive.copy()
    shifted_meta[16:24] = _pack_boundary(256, 4096, codec=0, trailing_bytes=0)
    _expect_raises(ValueError, lambda: td.read_tables(io.BytesIO(shifted_meta)), "raw metadata")

    # Nonzero raw padding and wrong declared padding size.
    nonzero_pad = archive.copy()
    nonzero_pad[4352 + 4096 + 17] = 0x5A
    _expect_raises(ValueError, lambda: td.raw_page_map(io.BytesIO(nonzero_pad), 0), "Nonzero bytes")

    wrong_pad = archive.copy()
    wrong_pad[24:32] = _pack_boundary(4352, 4096, codec=0, trailing_bytes=256)
    _expect_raises(ValueError, lambda: td.raw_page_map(io.BytesIO(wrong_pad), 0), "disagrees with its declared padding")

    # Non-increasing boundaries in stored or logical space.
    non_inc = archive.copy()
    non_inc[32:40] = _pack_boundary(4352, 8192, codec=8, trailing_bytes=0)
    _expect_raises(ValueError, lambda: td.raw_page_map(io.BytesIO(non_inc), 0), "increase in both address spaces")

    # Corrupt zlib stream.
    corrupt_zlib = archive.copy()
    raw_end = 4352 + 4096 + 512
    corrupt_zlib[raw_end] ^= 0xff
    _expect_raises(ValueError, lambda: td.zlib_segment(io.BytesIO(corrupt_zlib), 1), "Invalid zlib stream")

    # Invalid lookup key and unsupported ETPF key-table version.
    _expect_raises(ValueError, lambda: td.descriptor_indices(io.BytesIO(archive), -1), "unsigned 64-bit")
    _expect_raises(ValueError, lambda: td.descriptor_indices(io.BytesIO(archive), 1 << 64), "unsigned 64-bit")
    bad_ver, _, _ = _build_mixed_archive(key_version=0x100)
    _expect_raises(ValueError, lambda: td.descriptor_indices(io.BytesIO(bad_ver), 0), "Unsupported ETPF key-table version")


def check_cli(archive: bytearray, desc0, rpm, zseg):
    """Verify JSON CLI modes and clean SystemExit error handling without raw tracebacks."""
    archive2, _, _ = _build_mixed_archive(keys=(0x1122334455667788, 0xaaaaaaaaaaaaaaaa))
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        arch1 = work / "one.ttpf"
        arch2 = work / "two.ttpf"
        arch1.write_bytes(archive)
        arch2.write_bytes(archive2)

        # Inventory mode.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = td.main([str(arch1)])
        assert rc == 0, rc
        inv_json = json.loads(stdout.getvalue())
        assert inv_json["descriptor_count"] == 2
        assert inv_json["groups"] == [
            {"width": 64, "height": 32, "dxgi_format": 71, "mip_marker": 3, "count": 1},
            {"width": 128, "height": 16, "dxgi_format": 98, "mip_marker": 1, "count": 1},
        ]

        # --descriptor mode.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = td.main([str(arch1), "--descriptor", "0"])
        assert rc == 0 and json.loads(stdout.getvalue()) == desc0

        # --raw-page-map mode.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = td.main([str(arch1), "--raw-page-map", "0"])
        assert rc == 0 and json.loads(stdout.getvalue()) == rpm._asdict()

        # --zlib-segment mode.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = td.main([str(arch1), "--zlib-segment", "1"])
        assert rc == 0 and json.loads(stdout.getvalue()) == asdict(zseg)

        # --lookup-key across multiple archives.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = td.main([str(arch1), str(arch2), "--lookup-key", "0x1122334455667788"])
        assert rc == 0
        assert json.loads(stdout.getvalue()) == {
            "matches": [
                {"archive": str(arch1), "descriptor_index": 0},
                {"archive": str(arch2), "descriptor_index": 0},
            ]
        }

        # End-to-end subprocess positive run.
        done = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(arch1), "--descriptor", "0"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stderr == "", done
        assert json.loads(done.stdout) == desc0

        # Negative CLI controls: missing / directory / truncated / empty archive.
        missing = work / "missing.ttpf"
        _expect_exit([str(missing)])
        dir_arch = work / "dir_arch"
        dir_arch.mkdir()
        _expect_exit([str(dir_arch)])

        empty_arch = work / "empty.ttpf"
        empty_arch.write_bytes(b"")
        _expect_exit([str(empty_arch)], fragment="Truncated RDS5 header")

        short_arch = work / "short.ttpf"
        short_arch.write_bytes(b"RDS5" + bytes(8))
        _expect_exit([str(short_arch)], fragment="Truncated RDS5 header")

        # Negative indices validated before file open (kills main() negative-index guard).
        for flag in ("--descriptor", "--raw-page-map", "--zlib-segment"):
            _expect_exit([str(arch1), flag, "-1"], fragment="nonnegative")
            _expect_exit([str(missing), flag, "-1"], fragment="nonnegative")

        # Invalid --lookup-key and duplicate archive input.
        for bad_key in ("not_a_key", "-1", str(1 << 64)):
            _expect_exit([str(arch1), "--lookup-key", bad_key])
        _expect_exit([str(arch1), str(arch1), "--lookup-key", "0"], fragment="Duplicate archive input")

        # Multiple archives without --lookup-key -> argparse exit(2).
        _expect_exit([str(arch1), str(arch2)], expected_code=2, fragment="--lookup-key")

        # Catch struct.error and zlib.error cleanly in main() without raw tracebacks.
        orig_describe = td.describe_descriptor
        orig_zseg = td.zlib_segment
        try:
            def fail_struct(*_args, **_kwargs):
                raise struct.error("simulated struct failure")

            def fail_zlib(*_args, **_kwargs):
                raise zlib.error("simulated zlib failure")

            td.describe_descriptor = fail_struct
            _expect_exit([str(arch1), "--descriptor", "0"], fragment="simulated struct failure")

            td.zlib_segment = fail_zlib
            _expect_exit([str(arch1), "--zlib-segment", "1"], fragment="simulated zlib failure")
        finally:
            td.describe_descriptor = orig_describe
            td.zlib_segment = orig_zseg

        # Subprocess negative check (truncated header -> exit 1, no traceback).
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(short_arch)],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1
        assert "Truncated RDS5 header" in sub_err.stderr and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    archive, desc0, rpm, zseg = check_positive_api()
    check_negative_api(archive)
    check_cli(archive, desc0, rpm, zseg)
    print("texture_descriptors: synthetic RDS5/ETPF helpers, JSON CLI modes, and error exits passed")


if __name__ == "__main__":
    main()
