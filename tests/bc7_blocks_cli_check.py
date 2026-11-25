"""CLI, atomic output, and mutation control suite for tools/bc7_blocks.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bc7_blocks.py"
sys.path.insert(0, str(ROOT / "tools"))

import bc7_blocks  # noqa: E402


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = bc7_blocks.main(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class Bc7BlocksCliPositiveTests(unittest.TestCase):
    def test_main_creates_nested_outputs_byte_exact_case_and_fields_json(self):
        expected_fields, expected_blocks, expected_rgba = bc7_blocks.fixture(2, 3)
        expected_case = struct.pack("<4sII", b"BC7C", 8, 12) + expected_blocks + expected_rgba

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "nested" / "case_dir" / "blocks.bc7c"
            fields_path = root / "nested" / "json_dir" / "fields.json"
            self.assertFalse(case_path.parent.exists())
            self.assertFalse(fields_path.parent.exists())

            seen_replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path = Path(src)
                dst_path = Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(bc7_blocks.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main(
                    [str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)]
                )

            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(len(seen_replaces), 2)
            self.assertEqual([p.name for p in case_path.parent.iterdir()], [case_path.name])
            self.assertEqual([p.name for p in fields_path.parent.iterdir()], [fields_path.name])
            self.assertEqual(case_path.read_bytes(), expected_case)
            loaded = json.loads(fields_path.read_text(encoding="ascii"))
            self.assertIsInstance(loaded, list)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(loaded, expected_fields)

    def test_subprocess_same_parent_without_and_with_fields_atomically_replaces(self):
        expected_fields, expected_blocks, expected_rgba = bc7_blocks.fixture(2, 3)
        expected_case = struct.pack("<4sII", b"BC7C", 8, 12) + expected_blocks + expected_rgba

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "shared"
            out_dir.mkdir()
            case_path = out_dir / "blocks.bc7c"
            fields_path = out_dir / "fields.json"
            case_path.write_bytes(b"stale-case")
            fields_path.write_text("stale-json", encoding="ascii")

            proc_no_fields = _run_cli_subprocess(str(case_path), "--columns", "2", "--rows", "3")
            self.assertEqual(proc_no_fields.returncode, 0, proc_no_fields.stderr)
            self.assertEqual(proc_no_fields.stderr, "")
            self.assertEqual(case_path.read_bytes(), expected_case)
            self.assertEqual(fields_path.read_text(encoding="ascii"), "stale-json")

            proc_with_fields = _run_cli_subprocess(
                str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)
            )
            self.assertEqual(proc_with_fields.returncode, 0, proc_with_fields.stderr)
            self.assertEqual(proc_with_fields.stderr, "")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["blocks.bc7c", "fields.json"],
            )
            self.assertEqual(case_path.read_bytes(), expected_case)
            loaded = json.loads(fields_path.read_text(encoding="ascii"))
            self.assertEqual(len(loaded), 6)
            self.assertEqual(loaded, expected_fields)


class Bc7BlocksCliNegativeAndMutationTests(unittest.TestCase):
    def test_non_positive_columns_or_rows_rejected_upfront_without_traceback(self):
        for bad_args in (
            ["--columns", "0", "--rows", "3"],
            ["--columns", "2", "--rows", "-2"],
            ["--columns", "-1", "--rows", "0"],
        ):
            with self.subTest(bad_args=bad_args), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case_path = root / "nested" / "case.bc7c"
                fields_path = root / "nested" / "fields.json"
                argv = [str(case_path), *bad_args, "--fields", str(fields_path)]

                with mock.patch.object(bc7_blocks, "fixture") as fixture_mock:
                    code, out, err = _run_main(argv)
                fixture_mock.assert_not_called()
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertNotIn("Traceback", err)
                self.assertIn("positive block dimensions", err)
                self.assertFalse(case_path.parent.exists())

                proc = _run_cli_subprocess(*argv)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("positive block dimensions", proc.stderr)
                self.assertFalse(case_path.parent.exists())

    def test_fields_collision_with_case_resolve_and_samefile_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent paths colliding via parent traversal (kills resolve() mutation).
            case_ghost = root / "out" / "case.bc7c"
            fields_alias = root / "out" / ".." / "out" / "case.bc7c"
            with mock.patch.object(bc7_blocks, "fixture") as fixture_mock:
                code, out, err = _run_main(
                    [str(case_ghost), "--columns", "2", "--rows", "3", "--fields", str(fields_alias)]
                )
            fixture_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("collides with case", err)
            self.assertFalse(case_ghost.parent.exists())

            # 2) Hardlink alias where resolve() differs (kills samefile() mutation).
            case_real = root / "case.bc7c"
            fields_hardlink = sub / "fields_link.json"
            case_real.write_bytes(b"keep-case")
            os.link(case_real, fields_hardlink)
            self.assertNotEqual(case_real.resolve(), fields_hardlink.resolve())
            self.assertTrue(case_real.samefile(fields_hardlink))

            with mock.patch.object(bc7_blocks, "fixture") as fixture_mock:
                code, out, err = _run_main(
                    [str(case_real), "--columns", "2", "--rows", "3", "--fields", str(fields_hardlink)]
                )
            fixture_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("collides with case", err)
            self.assertEqual(case_real.read_bytes(), b"keep-case")

    def test_nested_destination_paths_refused_before_creating_anything(self):
        for fields_inside_case in (True, False):
            with self.subTest(fields_inside_case=fields_inside_case), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "new"
                child = parent / "child"
                case, fields = (parent, child) if fields_inside_case else (child, parent)
                code, _, err = _run_main([str(case), "--fields", str(fields)])
                self.assertEqual(code, 1)
                self.assertIn("cannot contain one another", err)
                self.assertFalse(parent.exists())

    def test_second_replace_failure_reports_partial_publish_and_retry_repairs_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case.bc7c"
            fields = root / "fields.json"
            case.write_bytes(b"old-case")
            fields.write_text("old-json", encoding="ascii")
            argv = [str(case), "--columns", "1", "--rows", "1", "--fields", str(fields)]
            replace = os.replace

            def fail_fields(src, dst):
                if dst == fields:
                    raise PermissionError("fields locked")
                replace(src, dst)

            with mock.patch.object(bc7_blocks.os, "replace", side_effect=fail_fields):
                code, _, err = _run_main(argv)
            self.assertEqual(code, 1)
            self.assertIn("Retry with both output paths", err)
            self.assertIn("fields locked", err)
            self.assertTrue(case.read_bytes().startswith(b"BC7C"))
            self.assertEqual(fields.read_text(encoding="ascii"), "old-json")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["case.bc7c", "fields.json"])
            code, _, err = _run_main(argv)
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(fields.read_text(encoding="ascii")), bc7_blocks.fixture(1, 1)[0])

    def test_existing_directory_outputs_rejected_upfront(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_dir = root / "already_dir"
            existing_dir.mkdir()
            case_file = root / "nested" / "case.bc7c"
            fields_file = root / "nested" / "fields.json"

            # case is an existing directory
            with mock.patch.object(bc7_blocks, "fixture") as fixture_mock:
                code, _, err = _run_main(
                    [str(existing_dir), "--columns", "2", "--rows", "3", "--fields", str(fields_file)]
                )
            fixture_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("existing directory", err)
            self.assertFalse(fields_file.parent.exists())

            # --fields is an existing directory
            with mock.patch.object(bc7_blocks, "fixture") as fixture_mock:
                code, _, err = _run_main(
                    [str(case_file), "--columns", "2", "--rows", "3", "--fields", str(existing_dir)]
                )
            fixture_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("existing directory", err)
            self.assertFalse(case_file.parent.exists())

    def test_struct_error_and_value_error_caught_cleanly_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "case.bc7c"
            fields_path = root / "fields.json"

            with mock.patch.object(bc7_blocks.struct, "pack", side_effect=struct.error("bad pack dimensions")):
                code, _, err = _run_main(
                    [str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("bad pack dimensions", err)
            self.assertEqual(list(root.iterdir()), [])

            with mock.patch.object(bc7_blocks, "fixture", side_effect=ValueError("simulated value failure")):
                code, _, err = _run_main(
                    [str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated value failure", err)
            self.assertEqual(list(root.iterdir()), [])

    def test_os_replace_and_staging_failures_leave_zero_temp_dirs_or_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_out"
            fields_dir = root / "fields_out"
            case_path = case_dir / "blocks.bc7c"
            fields_path = fields_dir / "fields.json"

            # 1) OSError during os.replace -> zero output files and zero temp staging dirs.
            replace_calls = []

            def failing_replace(src, dst):
                src_path = Path(src)
                replace_calls.append((src_path, Path(dst)))
                self.assertTrue(src_path.is_file())
                raise OSError("simulated replace failure")

            with mock.patch.object(bc7_blocks.os, "replace", side_effect=failing_replace):
                code, _, err = _run_main(
                    [str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated replace failure", err)
            self.assertEqual(len(replace_calls), 1)
            self.assertEqual(list(case_dir.iterdir()), [])
            self.assertEqual(list(fields_dir.iterdir()), [])

            # 2) Failure while staging --fields -> os.replace never called, both dirs empty.
            with (
                mock.patch.object(bc7_blocks.os, "replace") as replace_mock,
                mock.patch.object(bc7_blocks.json, "dumps", side_effect=ValueError("simulated json failure")),
            ):
                code, _, err = _run_main(
                    [str(case_path), "--columns", "2", "--rows", "3", "--fields", str(fields_path)]
                )
            replace_mock.assert_not_called()
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated json failure", err)
            self.assertEqual(list(case_dir.iterdir()), [])
            self.assertEqual(list(fields_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
