"""Unit and CLI suite for tools/range_manifest.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "range_manifest.py"
sys.path.insert(0, str(ROOT / "tools"))
import range_manifest  # noqa: E402


SYNTHETIC_MAP = """\
 sample_image

 Timestamp is 00000000 (Thu Jan 01 00:00:00 1970)

 Preferred load address is 0000000140000000

 Start         Length     Name                   Class
 0001:00000000 00001000H .text                   CODE

  Address         Publics by Value              Rva+Base               Lib:Object

 0001:00000100       fn_alpha                   0000000140001100 f   fixture.obj
 0001:00000140       fn_alpha_end               0000000140001140 f   fixture.obj
 0001:00000140       fn_alpha                   0000000140001999 f   duplicate.obj
 0001:00000200       fn_beta                    0000000140001200 f   fixture.obj
 0001:00000280       fn_beta_end                0000000140001280 f   fixture.obj
 0001:00000300       fn_empty                   0000000140001300 f   fixture.obj
 0001:00000300       fn_empty_end               0000000140001300 f   fixture.obj
 0001:00000420       fn_inverted                0000000140001420 f   fixture.obj
 0001:00000400       fn_inverted_end            0000000140001400 f   fixture.obj
 0001:00000500       fn_orphan                  0000000140001500 f   fixture.obj
"""


class RangeManifestPositiveTests(unittest.TestCase):
    def test_map_symbols_rva_subtraction_and_first_symbol_precedence(self):
        symbols = range_manifest.map_symbols(SYNTHETIC_MAP)
        self.assertEqual(symbols["fn_alpha"], 0x1100)
        self.assertEqual(symbols["fn_alpha_end"], 0x1140)
        self.assertEqual(symbols["fn_beta"], 0x1200)
        self.assertEqual(symbols["fn_beta_end"], 0x1280)

    def test_manifest_schema_and_sha256(self):
        image_bytes = b"MZ\x90\x00synthetic-range-image"
        expected_sha = hashlib.sha256(image_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.exe"
            map_file = root / "sample.map"
            image.write_bytes(image_bytes)
            map_file.write_text(SYNTHETIC_MAP, encoding="utf-8")

            result = range_manifest.manifest(image, map_file, ["fn_alpha", "fn_beta"])
            self.assertEqual(
                result,
                {
                    "image_sha256": expected_sha,
                    "functions": [
                        {"name": "fn_alpha", "start": "0x1100", "end": "0x1140"},
                        {"name": "fn_beta", "start": "0x1200", "end": "0x1280"},
                    ],
                },
            )

    def test_cli_main_and_subprocess_write_json_manifest(self):
        image_bytes = b"synthetic-pe-payload-for-cli"
        expected_sha = hashlib.sha256(image_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "app.exe"
            map_file = root / "app.map"
            out_dir = root / "out"
            out_file = out_dir / "ranges.json"
            image.write_bytes(image_bytes)
            map_file.write_text(SYNTHETIC_MAP, encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = range_manifest.main([str(image), str(map_file), str(out_file), "fn_alpha", "fn_beta"])

            self.assertEqual(rc, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["ranges.json"])
            parsed = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(parsed["image_sha256"], expected_sha)
            self.assertEqual(
                parsed["functions"],
                [
                    {"name": "fn_alpha", "start": "0x1100", "end": "0x1140"},
                    {"name": "fn_beta", "start": "0x1200", "end": "0x1280"},
                ],
            )

            sub_out = out_dir / "sub.json"
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(image), str(map_file), str(sub_out), "fn_beta"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            sub_parsed = json.loads(sub_out.read_text(encoding="utf-8"))
            self.assertEqual(sub_parsed["image_sha256"], expected_sha)
            self.assertEqual(sub_parsed["functions"], [{"name": "fn_beta", "start": "0x1200", "end": "0x1280"}])


class RangeManifestNegativeTests(unittest.TestCase):
    def test_map_symbols_requires_preferred_load_address(self):
        with self.assertRaisesRegex(ValueError, "preferred load address"):
            range_manifest.map_symbols(" 0001:00000100 fn_alpha 0000000140001100 f obj\n")

    def test_manifest_rejects_missing_empty_inverted_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "app.exe"
            map_file = root / "app.map"
            image.write_bytes(b"img")
            map_file.write_text(SYNTHETIC_MAP, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least one function name"):
                range_manifest.manifest(image, map_file, [])

            for bad_name in ("", "   "):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    range_manifest.manifest(image, map_file, [bad_name])

            with self.assertRaisesRegex(ValueError, "duplicate function name: fn_alpha"):
                range_manifest.manifest(image, map_file, ["fn_alpha", "fn_alpha"])

            for missing in ("fn_missing", "fn_orphan"):
                with self.assertRaisesRegex(ValueError, f"linker map has no {missing}"):
                    range_manifest.manifest(image, map_file, [missing])

            for bad_interval in ("fn_empty", "fn_inverted"):
                with self.assertRaisesRegex(ValueError, f"invalid interval for {bad_interval}"):
                    range_manifest.manifest(image, map_file, [bad_interval])

            with self.assertRaises(OSError):
                range_manifest.manifest(root / "missing.exe", map_file, ["fn_alpha"])

            with self.assertRaises(OSError):
                range_manifest.manifest(image, root / "missing.map", ["fn_alpha"])

    def test_cli_failures_exit_1_without_traceback_or_temp_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "app.exe"
            map_file = root / "app.map"
            out_dir = root / "out"
            out_dir.mkdir()
            image.write_bytes(b"img")
            map_file.write_text(SYNTHETIC_MAP, encoding="utf-8")

            cases = (
                ([str(root / "missing.exe"), str(map_file), str(out_dir / "a.json"), "fn_alpha"], "missing.exe"),
                ([str(image), str(root / "missing.map"), str(out_dir / "b.json"), "fn_alpha"], "missing.map"),
                ([str(image), str(map_file), str(out_dir / "c.json"), "fn_empty"], "invalid interval for fn_empty"),
                ([str(image), str(map_file), str(out_dir / "d.json"), "fn_inverted"], "invalid interval for fn_inverted"),
                ([str(image), str(map_file), str(out_dir / "e.json"), "fn_alpha", "fn_alpha"], "duplicate function name"),
                ([str(image), str(map_file), str(out_dir / "f.json"), ""], "must not be empty"),
                ([str(image), str(map_file), str(out_dir / "g.json"), "fn_orphan"], "linker map has no fn_orphan"),
            )
            for argv, expected_msg in cases:
                with self.subTest(expected=expected_msg):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as exc:
                            range_manifest.main(argv)
                    self.assertEqual(exc.exception.code, 1)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertIn(expected_msg, stderr.getvalue())
                    self.assertEqual(list(out_dir.iterdir()), [])

            # Output write/replace failure cleans up temp file and exits 1 without traceback.
            target = out_dir / "preserved.json"
            target.write_text('{"preserved": true}\n', encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(range_manifest.os, "replace", side_effect=OSError("disk full")),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exc:
                    range_manifest.main([str(image), str(map_file), str(target), "fn_alpha"])
            self.assertEqual(exc.exception.code, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn("disk full", stderr.getvalue())
            self.assertEqual([p.name for p in out_dir.iterdir()], ["preserved.json"])
            self.assertEqual(target.read_text(encoding="utf-8"), '{"preserved": true}\n')

            # Subprocess negative smoke test.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(image), str(map_file), str(out_dir / "sub_fail.json"), "fn_inverted"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("invalid interval for fn_inverted", proc.stderr)
            self.assertEqual([p.name for p in out_dir.iterdir()], ["preserved.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
