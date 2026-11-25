"""Pure-Python unit and CLI validation suite for tools/disasm_rdna2.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "disasm_rdna2.py"
sys.path.insert(0, str(ROOT / "tools"))

import disasm_rdna2  # noqa: E402


S_NOP_0 = bytes.fromhex("000080bf")
S_NOP_1 = bytes.fromhex("010080bf")
S_MOV_LIT = bytes.fromhex("ff0380be78563412")
S_ENDPGM = bytes.fromhex("000081bf")


class DisassembleBinaryUnitTests(unittest.TestCase):
    def test_empty_stream_returns_empty_list(self):
        self.assertIs(disasm_rdna2.disasm_rdna2, disasm_rdna2.disassemble_binary)
        self.assertEqual(disasm_rdna2.disassemble_binary(b""), [])

    def test_offsets_lengths_opcodes_after_endpgm_and_trailing_partial_words(self):
        for tail in (b"\xaa", b"\xaa\xbb", b"\xaa\xbb\xcc"):
            with self.subTest(tail_len=len(tail)):
                stream = S_NOP_0 + S_MOV_LIT + S_ENDPGM + S_NOP_1 + tail
                records = disasm_rdna2.disassemble_binary(stream, "synthetic")
                self.assertEqual(
                    [(r["offset"], r["length"], r["opcode"], r["after_endpgm"]) for r in records],
                    [
                        (0, 4, "s_nop", False),
                        (4, 8, "s_mov_b32", False),
                        (12, 4, "s_endpgm", False),
                        (16, 4, "s_nop", True),
                        (20, len(tail), "data", True),
                    ],
                )
                self.assertEqual(records[-1]["hex"], tail.hex())
                self.assertEqual(records[-1]["assembly"], f".byte {tail.hex()}")

    def test_truncated_literal_instruction_followed_by_trailing_bytes(self):
        truncated = S_MOV_LIT[:6]
        records = disasm_rdna2.disassemble_binary(truncated)
        self.assertEqual(
            [(r["offset"], r["length"], r["opcode"], r["after_endpgm"]) for r in records],
            [
                (0, 4, "sop1_truncated", False),
                (4, 2, "data", False),
            ],
        )


class DisasmRdna2CliPositiveTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = disasm_rdna2.main(argv)
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_hex_mode_with_and_without_spaces_and_output_listing(self):
        compact = (S_NOP_0 + S_ENDPGM).hex()
        spaced = " 00 00 80 bf  00 00 81 bf "

        rc_compact, out_compact, err_compact = self._run_main(["--hex", compact])
        self.assertEqual(rc_compact, 0)
        self.assertEqual(err_compact, "")
        self.assertIn("RDNA 2 DISASSEMBLY: hex_input", out_compact)
        self.assertIn("s_nop 0", out_compact)
        self.assertIn("s_endpgm", out_compact)
        self.assertIn("Summary: 2 instructions | 8 bytes", out_compact)

        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "hex_listing.asm"
            rc_spaced, out_spaced, err_spaced = self._run_main(["--hex", spaced, "--output", str(listing)])
            self.assertEqual(rc_spaced, 0)
            self.assertEqual(err_spaced, "")
            self.assertIn(f"[+] Disassembly saved to: {listing}", out_spaced)
            text = listing.read_text(encoding="utf-8")
            self.assertIn("; Disassembly: hex_input", text)
            self.assertIn("s_nop 0", text)
            self.assertIn("s_endpgm", text)

    def test_single_bin_and_directory_of_bins_with_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single.bin"
            single.write_bytes(S_MOV_LIT + S_ENDPGM)

            out_single = root / "single_out.txt"
            rc, out, err = self._run_main([str(single), "--output", str(out_single)])
            self.assertEqual(rc, 0)
            self.assertEqual(err, "")
            self.assertIn("RDNA 2 DISASSEMBLY: single.bin", out)
            self.assertIn("s_mov_b32 s0, 0x12345678", out)
            self.assertIn("; Disassembly: single.bin", out_single.read_text(encoding="utf-8"))

            shader_dir = root / "shaders"
            shader_dir.mkdir()
            (shader_dir / "a_first.bin").write_bytes(S_NOP_0 + S_ENDPGM)
            (shader_dir / "b_second.bin").write_bytes(S_NOP_1 + S_ENDPGM)
            (shader_dir / "ignore.txt").write_text("not a binary", encoding="utf-8")
            (shader_dir / "subdir.bin").mkdir()

            out_dir_listing = root / "dir_out.txt"
            rc_dir, out_dir, err_dir = self._run_main([str(shader_dir), "--output", str(out_dir_listing)])
            self.assertEqual(rc_dir, 0)
            self.assertEqual(err_dir, "")
            self.assertIn("Found 2 shader binary files in shaders", out_dir)
            self.assertIn("RDNA 2 DISASSEMBLY: a_first.bin", out_dir)
            self.assertIn("RDNA 2 DISASSEMBLY: b_second.bin", out_dir)
            listing_text = out_dir_listing.read_text(encoding="utf-8")
            self.assertIn("; Disassembly: a_first.bin", listing_text)
            self.assertIn("; Disassembly: b_second.bin", listing_text)
            self.assertNotIn("ignore.txt", listing_text)

    def test_subprocess_positive_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shader = root / "probe.bin"
            shader.write_bytes(S_NOP_0 + S_ENDPGM)
            listing = root / "probe.txt"
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(shader), "--output", str(listing)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("RDNA 2 DISASSEMBLY: probe.bin", proc.stdout)
            self.assertTrue(listing.is_file())


class DisasmRdna2CliNegativeTests(unittest.TestCase):
    def _assert_main_returns_1(self, argv: list[str], expected_fragment: str = "") -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = disasm_rdna2.main(argv)
        err = stderr.getvalue()
        self.assertEqual(rc, 1, (argv, stdout.getvalue(), err))
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr error for {argv!r}")
        if expected_fragment:
            self.assertIn(expected_fragment, err)
        return err

    def _assert_main_exits_2(self, argv: list[str], expected_fragment: str = "") -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                disasm_rdna2.main(argv)
        err = stderr.getvalue()
        self.assertEqual(exc.exception.code, 2, (argv, stdout.getvalue(), err))
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr error for {argv!r}")
        if expected_fragment:
            self.assertIn(expected_fragment, err)
        return err

    def test_missing_targets_and_no_hex_exits_2_without_traceback(self):
        self._assert_main_exits_2([], "Specify targets or provide hex via --hex")

    def test_invalid_odd_and_empty_hex_exit_2_without_traceback(self):
        for bad_hex in ("", "   ", "\t\n", "000", "zzzz", "00 0g"):
            with self.subTest(bad_hex=bad_hex):
                self._assert_main_exits_2(["--hex", bad_hex], "--hex")

        proc = subprocess.run(
            [sys.executable, "-B", str(TOOL), "--hex", ""],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("--hex", proc.stderr)

    def test_missing_and_unreadable_targets_return_1_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.bin"
            out_file = root / "should_not_exist.txt"
            self._assert_main_returns_1([str(missing), "--output", str(out_file)], "not found")
            self.assertFalse(out_file.exists())

            # Mixed valid + missing target must fail without writing partial --output.
            valid = root / "valid.bin"
            valid.write_bytes(S_NOP_0 + S_ENDPGM)
            self._assert_main_returns_1([str(valid), str(missing), "--output", str(out_file)], "not found")
            self.assertFalse(out_file.exists())

            unreadable = root / "unreadable.bin"
            unreadable.write_bytes(S_ENDPGM)
            with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("access denied")):
                self._assert_main_returns_1([str(unreadable), "--output", str(out_file)], "access denied")
            self.assertFalse(out_file.exists())

            shader_dir = root / "dir"
            shader_dir.mkdir()
            with mock.patch.object(Path, "glob", side_effect=PermissionError("dir denied")):
                self._assert_main_returns_1([str(shader_dir), "--output", str(out_file)], "dir denied")
            self.assertFalse(out_file.exists())

            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(missing)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("not found", proc.stderr)

    def test_output_collision_with_input_bin_is_rejected_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            original = S_NOP_0 + S_ENDPGM
            shader = root / "shader.bin"
            shader.write_bytes(original)

            # Direct path collision and resolved ".." path collision.
            self._assert_main_returns_1([str(shader), "--output", str(shader)], "collides")
            self.assertEqual(shader.read_bytes(), original)

            dotdot_out = sub / ".." / "shader.bin"
            self._assert_main_returns_1([str(shader), "--output", str(dotdot_out)], "collides")
            self.assertEqual(shader.read_bytes(), original)

            # Directory target containing shader.bin colliding with --output.
            self._assert_main_returns_1([str(root), "--output", str(shader)], "collides")
            self.assertEqual(shader.read_bytes(), original)

            # Hardlink / samefile collision (when supported by filesystem) and mocked samefile fallback.
            alias = root / "alias.txt"
            try:
                os.link(shader, alias)
            except OSError:
                alias.write_bytes(b"distinct")
                with mock.patch.object(Path, "samefile", return_value=True):
                    self._assert_main_returns_1([str(shader), "--output", str(alias)], "collides")
                self.assertEqual(alias.read_bytes(), b"distinct")
            else:
                self._assert_main_returns_1([str(shader), "--output", str(alias)], "collides")
                self.assertEqual(shader.read_bytes(), original)

    def test_output_directory_and_unwritable_paths_return_1_and_clean_up_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shader = root / "shader.bin"
            shader.write_bytes(S_NOP_0 + S_ENDPGM)

            out_dir = root / "existing_dir"
            out_dir.mkdir()
            self._assert_main_returns_1([str(shader), "--output", str(out_dir)], "directory")
            self._assert_main_returns_1(["--hex", S_ENDPGM.hex(), "--output", str(out_dir)], "directory")
            self._assert_main_returns_1(["--hex", S_ENDPGM.hex(), "--output", ""], "directory")

            # Missing parent directory -> OSError caught cleanly.
            missing_parent = root / "no_such_dir" / "out.txt"
            self._assert_main_returns_1([str(shader), "--output", str(missing_parent)])

            # File used as parent directory -> OSError caught cleanly.
            not_a_dir = shader / "out.txt"
            self._assert_main_returns_1([str(shader), "--output", str(not_a_dir)])

            # Replace failure cleans up temporary file and preserves pre-existing destination.
            existing = out_dir / "existing.txt"
            existing.write_text("original-listing", encoding="utf-8")
            with mock.patch.object(Path, "replace", side_effect=PermissionError("read-only target")):
                self._assert_main_returns_1([str(shader), "--output", str(existing)], "read-only target")
            self.assertEqual(existing.read_text(encoding="utf-8"), "original-listing")
            self.assertEqual(list(out_dir.iterdir()), [existing])


if __name__ == "__main__":
    unittest.main()
