#!/usr/bin/env python3
"""Check tools/bcpack_probe.py: lane_record(), scan_lanes(), probe() for lane strides 6 and 8,
main(argv) JSON output, and bounded/truncated/missing CLI error handling without tracebacks."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bcpack_probe.py"
sys.path.insert(0, str(ROOT / "tools"))
import bcpack_probe as bp


def _expect_value_error(fn, fragment=""):
    try:
        fn()
    except ValueError as exc:
        if fragment:
            assert fragment in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"expected ValueError containing {fragment!r}")


def _expect_exit(argv, expected_code=1, fragment=""):
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            bp.main(argv)
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


def _interleaved_two_refill_prefix():
    prefix = bytearray(48)
    prefix[3::8] = bytes.fromhex("dd 05 03 0b 87 c1")
    return bytes(prefix)


def _contiguous_constant_prefix():
    prefix = bytearray(48)
    prefix[24:30] = bytes.fromhex("de 03 01 00 40 02")
    return bytes(prefix)


def check_positive_api():
    """Verify lane_record(), scan_lanes(), and probe() for lane_stride=8 and lane_stride=6."""
    interleaved = _interleaved_two_refill_prefix()
    assert bp.lane_record(interleaved, 3, 8) == bytes.fromhex("dd 05 03 0b 87 c1")
    assert bp.lane_record(interleaved, 0, 8) == bytes(6)

    contiguous = _contiguous_constant_prefix()
    assert bp.lane_record(contiguous, 4, 6) == bytes.fromhex("de 03 01 00 40 02")
    assert bp.lane_record(contiguous, 0, 6) == bytes(6)

    scan8 = bp.scan_lanes(interleaved, lane_stride=8)
    assert len(scan8) == 8
    assert scan8[3] == {
        "lane": 3,
        "mode": 0xDD,
        "budget": 5,
        "state": "030b87c1",
        "known_constant_mode": True,
    }
    assert scan8[0]["known_constant_mode"] is False

    scan6 = bp.scan_lanes(contiguous, lane_stride=6)
    assert len(scan6) == 8
    assert scan6[4] == {
        "lane": 4,
        "mode": 0xDE,
        "budget": 3,
        "state": "01004002",
        "known_constant_mode": True,
    }

    matches8 = bp.probe(
        interleaved,
        100000,
        prefetch_mask=0x09,
        prefetch=b"\xaa\x93",
        next_mask=0x18,
        next_bytes=b"\xe3\xbb",
        lane_stride=8,
    )
    assert matches8 == [{
        "lane": 3,
        "mode": 0xDD,
        "refill_bytes": 2,
        "status": "verified_constant_lane_with_supplied_refills",
        "first_refill_offset": 49,
        "first_refill_matches": True,
        "second_refill_offset": 50,
        "second_refill_matches": True,
        "refill_symbol_indices": [9129, 54564],
    }], matches8

    matches6 = bp.probe(contiguous, 2, lane_stride=6)
    assert matches6 == [{
        "lane": 4,
        "mode": 0xDE,
        "refill_bytes": 0,
        "status": "verified_constant_lane",
    }], matches6

    # Upper block-count boundary (1 << 20) is accepted.
    assert bp.probe(bytes(48), 1 << 20) == []
    return interleaved, contiguous, scan8, scan6, matches8, matches6


def check_negative_api():
    """Pin exact ValueError checks for bounds, lane stride, and mismatched lengths."""
    _expect_value_error(lambda: bp.scan_lanes(bytes(47)), "48-byte prefix")
    _expect_value_error(lambda: bp.scan_lanes(bytes(49)), "48-byte prefix")
    _expect_value_error(lambda: bp.scan_lanes(bytes(48), lane_stride=7), "lane stride 6 or 8")

    _expect_value_error(lambda: bp.probe(bytes(47), 2), "48-byte prefix")
    _expect_value_error(lambda: bp.probe(bytes(49), 2), "48-byte prefix")
    _expect_value_error(lambda: bp.probe(bytes(48), 0), "positive bounded block count")
    _expect_value_error(lambda: bp.probe(bytes(48), -1), "positive bounded block count")
    _expect_value_error(lambda: bp.probe(bytes(48), (1 << 20) + 1), "positive bounded block count")
    _expect_value_error(lambda: bp.probe(bytes(48), 2, lane_stride=7), "lane stride 6 or 8")
    _expect_value_error(
        lambda: bp.probe(bytes(48), 2, prefetch_mask=0x03, prefetch=b"\x00"),
        "first-round byte",
    )
    _expect_value_error(
        lambda: bp.probe(bytes(48), 2, next_mask=0x03, next_bytes=b"\x00"),
        "second-round byte",
    )
    _expect_value_error(
        lambda: bp.probe(bytes(48), 2, prefetch_mask=0x100, prefetch=b"\x00"),
        "first-round byte",
    )
    _expect_value_error(
        lambda: bp.probe(bytes(48), 2, next_mask=0x100, next_bytes=b"\x00"),
        "second-round byte",
    )


def check_cli(interleaved, contiguous, scan8, scan6, matches8, matches6):
    """Verify main(argv) JSON stdout positive runs and clean SystemExit negative controls."""
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        pkt8 = work / "interleaved.bin"
        pkt8.write_bytes(interleaved + b"\xaa\x93" + b"\xe3\xbb")
        pkt6 = work / "contiguous.bin"
        pkt6.write_bytes(contiguous)

        # main(argv) --scan-lanes (stride 8 and stride 6)
        for pkt, stride, expected_scan in ((pkt8, 8, scan8), (pkt6, 6, scan6)):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = bp.main([str(pkt), "--scan-lanes", "--lane-stride", str(stride)])
            assert rc == 0, rc
            assert json.loads(stdout.getvalue()) == {"lane_stride": stride, "lanes": expected_scan}

        # main(argv) --blocks with --prefetch-mask and --next-mask (stride 8)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = bp.main([
                str(pkt8),
                "--blocks", "100000",
                "--prefetch-mask", "0x09",
                "--next-mask", "0x18",
            ])
        assert rc == 0, rc
        assert json.loads(stdout.getvalue()) == matches8

        # main(argv) --blocks with --lane-stride 6
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = bp.main([str(pkt6), "--blocks", "2", "--lane-stride", "6"])
        assert rc == 0, rc
        assert json.loads(stdout.getvalue()) == matches6

        # End-to-end subprocess positive check.
        done = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(pkt8), "--blocks", "100000",
             "--prefetch-mask", "0x09", "--next-mask", "0x18"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stderr == "", done
        assert json.loads(done.stdout) == matches8

        # Missing packet -> OSError / SystemExit(1).
        missing = work / "missing.bin"
        try:
            missing.open("rb")
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for missing packet")
        _expect_exit([str(missing), "--blocks", "4"], fragment=missing.name)

        # Directory packet -> OSError / SystemExit(1).
        dir_pkt = work / "dir_packet"
        dir_pkt.mkdir()
        try:
            dir_pkt.open("rb")
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for directory packet")
        _expect_exit([str(dir_pkt), "--blocks", "4"], fragment=dir_pkt.name)

        # Truncated prefix (< 48 bytes) -> SystemExit(1) for both --blocks and --scan-lanes.
        short = work / "short.bin"
        short.write_bytes(bytes(10))
        _expect_exit([str(short), "--blocks", "4"], fragment="got 10 bytes")
        _expect_exit([str(short), "--scan-lanes"], fragment="got 10 bytes")

        # Truncated prefetch and truncated next-round bytes -> SystemExit(1).
        header_only = work / "header_only.bin"
        header_only.write_bytes(bytes(48))
        _expect_exit([str(header_only), "--blocks", "2", "--prefetch-mask", "0x01"],
                     fragment="Expected 1 prefetch byte(s), got 0")
        _expect_exit([str(header_only), "--scan-lanes", "--prefetch-mask", "0x01"],
                     fragment="Expected 1 prefetch byte(s), got 0")

        one_prefetch = work / "one_prefetch.bin"
        one_prefetch.write_bytes(bytes(49))
        _expect_exit([str(one_prefetch), "--blocks", "2", "--prefetch-mask", "0x01", "--next-mask", "0x03"],
                     fragment="Expected 2 next-round byte(s), got 0")

        # Missing --blocks -> argparse usage exit(2).
        _expect_exit([str(header_only)], expected_code=2, fragment="--blocks is required")

        # --blocks <= 0 or > 1<<20 -> SystemExit(1), checked before opening packet.
        for bad_blocks in ("0", "-1", str((1 << 20) + 1)):
            _expect_exit([str(header_only), "--blocks", bad_blocks], fragment="Expected --blocks in 1..")
            _expect_exit([str(missing), "--blocks", bad_blocks], fragment="Expected --blocks in 1..")

        # Unsupported --lane-stride and out-of-range masks -> argparse usage exit(2).
        _expect_exit([str(header_only), "--scan-lanes", "--lane-stride", "7"], expected_code=2)
        _expect_exit([str(header_only), "--blocks", "2", "--prefetch-mask", "0x100"], expected_code=2)
        _expect_exit([str(header_only), "--blocks", "2", "--next-mask", "0x100"], expected_code=2)

        # Subprocess negative check (no traceback on stderr).
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(short), "--blocks", "4"],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1
        assert short.name in sub_err.stderr and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    interleaved, contiguous, scan8, scan6, matches8, matches6 = check_positive_api()
    check_negative_api()
    check_cli(interleaved, contiguous, scan8, scan6, matches8, matches6)
    print("bcpack_probe_cli: lane records, scan_lanes, probe strides 6/8, JSON CLI, and bounds checks passed")


if __name__ == "__main__":
    main()
