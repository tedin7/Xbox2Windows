"""Pure-Python CLI and chunk validation unit suite for tools/startup/capture.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup" / "capture.py"
sys.path.insert(0, str(ROOT / "tools" / "startup"))

import capture as C  # noqa: E402


RES_W = 0x7FF600123456
HEAP_W = 0x7FF600222222
CONSOLE = (3, 0x10000, 64, 32, 1, 4, 28, 1, 0, 9, 0x2)
DESKTOP = (3, 0x10000, 128, 64, 1, 6, 87, 1, 0, 0, 0x5)


def _chunk(tag: bytes, header: bytes = b"", body: bytes = b"") -> bytes:
    return struct.pack("<4sIQ", tag, len(header), len(body)) + header + body


def _desc(dim, align, width, height, depth, mips, fmt, count, quality, layout, flags) -> bytes:
    return struct.pack("<I4xQQIHHIIIII4x", dim, align, width, height, depth, mips, fmt, count, quality, layout, flags)


def _res(wrapper: int = RES_W) -> bytes:
    return _chunk(b"RES ", struct.pack("<4Q", wrapper, 0xAA0000, 4096, 0xBB0000) + _desc(*CONSOLE) + _desc(*DESKTOP))


def _view(vid: int, slot: int, resource: int, desc_bytes: bytes, *, exact_header: bool = False) -> bytes:
    header = struct.pack("<IIQQQQII", vid, slot, resource, 0xC0, 0xD0, 0xE0, len(desc_bytes), 0)
    payload = desc_bytes if exact_header else desc_bytes.ljust(64, b"\x00")
    return _chunk(b"VIEW", header + payload)


def _table(seq: int, slot: int, parameter: int, entries: int, gpu: int, heap: int, index: int, ids=()) -> bytes:
    return _chunk(
        b"TABL",
        struct.pack("<IIIIQQQ", seq, slot, parameter, entries, gpu, heap, index),
        struct.pack(f"<{len(ids)}I", *ids),
    )


def _call(seq: int = 5) -> bytes:
    return _chunk(b"CALL", struct.pack("<IIIIQ7QQ", 1, seq, 77, 36, 0x1000ABC, RES_W, 2, 3, 4, 5, 6, 7, 0x99))


class StartupCaptureCliPositiveTests(unittest.TestCase):
    def _assert_main_matches_describe(self, blob: bytes) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.bin"
            path.write_bytes(blob)
            expected = io.StringIO()
            C.describe(C.Capture(path), expected)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = C.main([str(path)])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(stdout.getvalue(), expected.getvalue())
            return stdout.getvalue()

    def test_main_matches_describe_on_empty_capture(self):
        text = self._assert_main_matches_describe(b"")
        self.assertEqual(
            text,
            "calls 0, memory blocks 0 (0 bytes), pipelines 0, roots 0, resources 0, heaps 0, views 0\nslot histogram: \n",
        )

    def test_main_matches_describe_with_shadow_only_table_and_exact_view_boundary(self):
        blob = b"".join(
            [
                _chunk(b"FRAM", struct.pack("<II", 1, 0)),
                _call(seq=5),
                _res(),
                # Exact boundary: len(h) == 48 + desc_bytes (48 + 4 == 52).
                _view(0, 18, RES_W, b"\xAA\xBB\xCC\xDD", exact_header=True),
                _view(1, 17, 0, b"", exact_header=True),
                # Shadow-only TABL: entries == 2 and len(ids) == 2 (no token half).
                _table(5, 36, 2, 2, 0x3000, HEAP_W, 1, ids=(1, 2)),
                _chunk(b"PRES", struct.pack("<IIQ", 1, 0, RES_W)),
            ]
        )
        text = self._assert_main_matches_describe(blob)
        self.assertIn("calls 1, memory blocks 0 (0 bytes), pipelines 0, roots 0, resources 1, heaps 0, views 2", text)
        self.assertIn("table param 2 heap index 1: 2 of 2 entries known", text)
        self.assertIn("SRV view 0 res=123456", text)
        self.assertIn("desc=aabbccdd", text)
        self.assertIn("CBV view 1 res=0", text)
        self.assertNotIn("shadow", text)
        self.assertIn("present frame 1 surface Resource@123456", text)

        # Subprocess smoke test on valid synthetic capture.bin.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.bin"
            path.write_bytes(blob)
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(proc.stdout.replace("\r\n", "\n"), text)


class StartupCaptureCliNegativeTests(unittest.TestCase):
    def _assert_main_fails(self, argv: list[str], *, expected_code: int, expected_text: str = "") -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                C.main(argv)
        self.assertEqual(exc.exception.code, expected_code)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip())
        if expected_text:
            self.assertIn(expected_text, stderr.getvalue())
        return stderr.getvalue()

    def _assert_capture_and_cli_reject(self, blob: bytes, expected_text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_capture.bin"
            path.write_bytes(blob)
            with self.assertRaises(ValueError) as exc:
                C.Capture(path)
            self.assertIn(expected_text, str(exc.exception))
            self._assert_main_fails([str(path)], expected_code=1, expected_text=expected_text)

    def test_truncated_view_descriptor_rejected(self):
        # 48 + desc_bytes (48 + 8 = 56) > len(h) (55 -> 1 byte short).
        header_short_by_1 = struct.pack("<IIQQQQII", 0, 18, RES_W, 0, 0, 0, 8, 0) + b"\x11" * 7
        self._assert_capture_and_cli_reject(_chunk(b"VIEW", header_short_by_1), "truncated VIEW descriptor")

        # Header shorter than 48 even when desc_bytes == 0.
        header_44 = struct.pack("<IIQQQQI", 0, 18, RES_W, 0, 0, 0, 0)
        self._assert_capture_and_cli_reject(_chunk(b"VIEW", header_44), "truncated VIEW descriptor")

    def test_malformed_tabl_rejected(self):
        # Non-4-multiple body (6 bytes) with entries == 1.
        tabl_header = struct.pack("<IIIIQQQ", 5, 36, 2, 1, 0x3000, HEAP_W, 1)
        self._assert_capture_and_cli_reject(_chunk(b"TABL", tabl_header, b"\x01\x00\x00\x00\xAA\xBB"), "multiple of 4")

        # entries > len(ids) (declares 3 entries, only 2 uint32 ids supplied).
        tabl_header_3 = struct.pack("<IIIIQQQ", 5, 36, 2, 3, 0x3000, HEAP_W, 1)
        self._assert_capture_and_cli_reject(_chunk(b"TABL", tabl_header_3, struct.pack("<2I", 1, 2)), "declares 3 entries")

    def test_malformed_capture_containers_rejected(self):
        cases = (
            ("truncated_chunk_header", b"FRAM\x08\x00\x00\x00", "truncated chunk header"),
            ("truncated_chunk_payload", _chunk(b"FRAM", struct.pack("<II", 1, 0))[:-2], "truncated b'FRAM' chunk"),
            ("unknown_chunk_tag", _chunk(b"ZZZZ", b"\x00" * 8), "unknown chunk tag b'ZZZZ'"),
            ("bad_fram_struct", _chunk(b"FRAM", b"\x00" * 4), "bad b'FRAM' chunk"),
            ("bad_call_struct", _chunk(b"CALL", b"\x00" * 20), "bad b'CALL' chunk"),
        )
        for label, blob, expected_text in cases:
            with self.subTest(case=label):
                self._assert_capture_and_cli_reject(blob, expected_text)

    def test_cli_missing_arg_and_oserror_paths_exit_without_traceback(self):
        self._assert_main_fails([], expected_code=2)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "does_not_exist.bin"
            self._assert_main_fails([str(missing)], expected_code=1)

            folder = root / "a_directory"
            folder.mkdir()
            self._assert_main_fails([str(folder)], expected_code=1)

            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(missing)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())


if __name__ == "__main__":
    unittest.main()
