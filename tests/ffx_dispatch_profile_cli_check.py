"""CLI, atomic output, and mutation control suite for tools/ffx_dispatch_profile.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ffx_dispatch_profile.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import ffx_dispatch_profile  # noqa: E402
from ffx_dispatch_profile_check import build_synthetic_pe  # noqa: E402


EXPECTED_PROGRAMS = ("pass24", "pass25", "producer", "rcas", "initializer")


def _write_synthetic_image(directory: Path, name: str = "upscaler.dll"):
    buf, meta = build_synthetic_pe()
    raw = bytes(buf)
    image = directory / name
    image.write_bytes(raw)
    return image, raw, meta, hashlib.sha256(raw).hexdigest()


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = ffx_dispatch_profile.main(argv)
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


class FfxDispatchProfileCliPositiveTests(unittest.TestCase):
    def test_main_creates_nested_output_dir_with_exact_bins_sha_and_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, meta, digest = _write_synthetic_image(root)
            out_dir = root / "nested" / "deep" / "profiles"
            self.assertFalse(out_dir.exists())

            seen_replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                seen_replaces.append((Path(src), Path(dst)))
                return real_replace(src, dst)

            with (
                mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest),
                mock.patch.object(ffx_dispatch_profile.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main([str(image), str(out_dir)])

            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                sorted(f"{name}.bin" for name in EXPECTED_PROGRAMS),
            )
            self.assertEqual(len(seen_replaces), len(EXPECTED_PROGRAMS))
            for src, dst in seen_replaces:
                self.assertEqual(src.parent.parent.resolve(), out_dir.resolve())
                self.assertEqual(dst.parent.resolve(), out_dir.resolve())

            for name in EXPECTED_PROGRAMS:
                expected = meta["payloads"][name]
                actual = (out_dir / f"{name}.bin").read_bytes()
                self.assertEqual(actual, expected)
                self.assertEqual(
                    hashlib.sha256(actual).hexdigest(),
                    hashlib.sha256(expected).hexdigest(),
                )

            expected_threads = {"pass24": 64, "pass25": 32, "producer": 256, "rcas": 64}
            lines = out.splitlines()
            self.assertEqual(len(lines), 4)
            for line, name in zip(lines, ("pass24", "pass25", "producer", "rcas")):
                self.assertRegex(
                    line,
                    rf"^{name} container=0x[0-9a-f]+ threads={expected_threads[name]} "
                    rf"code_bytes={len(meta['payloads'][name])}$",
                )

    def test_main_atomically_replaces_existing_bins_without_leftover_temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, meta, digest = _write_synthetic_image(root)
            out_dir = root / "out"
            out_dir.mkdir()
            for name in EXPECTED_PROGRAMS:
                (out_dir / f"{name}.bin").write_bytes(b"stale-data")

            with mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest):
                code, _, err = _run_main([str(image), str(out_dir)])

            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                sorted(f"{name}.bin" for name in EXPECTED_PROGRAMS),
            )
            for name in EXPECTED_PROGRAMS:
                self.assertEqual((out_dir / f"{name}.bin").read_bytes(), meta["payloads"][name])


class FfxDispatchProfileCliNegativeAndMutationTests(unittest.TestCase):
    def test_missing_image_exits_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.dll"
            out_dir = root / "out_missing"

            code, out, err = _run_main([str(missing), str(out_dir)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertTrue(err.strip())
            self.assertFalse(out_dir.exists())

            proc = _run_cli_subprocess(str(missing), str(out_dir))
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())
            self.assertFalse(out_dir.exists())

    def test_mismatching_upscaler_image_exits_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, _, _ = _write_synthetic_image(root)
            out_dir = root / "out_mismatch"

            code, out, err = _run_main([str(image), str(out_dir)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("version-query profile", err)
            self.assertFalse(out_dir.exists())

            proc = _run_cli_subprocess(str(image), str(out_dir))
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("version-query profile", proc.stderr)
            self.assertFalse(out_dir.exists())

    def test_invalid_pe_format_error_exits_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_image = root / "not_pe.dll"
            payload = b"not a portable executable"
            bad_image.write_bytes(payload)
            out_dir = root / "out_bad_pe"

            with mock.patch.object(
                ffx_dispatch_profile,
                "UPSCALER",
                hashlib.sha256(payload).hexdigest(),
            ):
                code, out, err = _run_main([str(bad_image), str(out_dir)])

            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertTrue(err.strip())
            self.assertFalse(out_dir.exists())

    def test_struct_error_from_truncated_data_exits_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, _, digest = _write_synthetic_image(root)
            out_dir = root / "out_struct"

            with (
                mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest),
                mock.patch.object(
                    ffx_dispatch_profile,
                    "inspect",
                    side_effect=struct.error("unpack requires a buffer of 4 bytes"),
                ),
            ):
                code, out, err = _run_main([str(image), str(out_dir)])

            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("unpack requires a buffer of 4 bytes", err)
            self.assertFalse(out_dir.exists())

    def test_output_file_collision_preserves_image_and_other_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, original, _, digest = _write_synthetic_image(root, "initializer.bin")
            with mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest):
                code, _, err = _run_main([str(image), str(root)])

            self.assertEqual(code, 1)
            self.assertIn("Output file collides with image", err)
            self.assertEqual(image.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [image])

    def test_output_directory_collision_with_image_resolve_and_samefile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, raw, _, digest = _write_synthetic_image(root)
            sub = root / "sub"
            sub.mkdir()

            # 1) Relative/parent-traversal alias resolving to existing image.
            alias = sub / ".." / "upscaler.dll"
            with mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest):
                code, out, err = _run_main([str(image), str(alias)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("collides with image", err)
            self.assertEqual(image.read_bytes(), raw)

            proc = _run_cli_subprocess(str(image), str(alias))
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("collides with image", proc.stderr)
            self.assertEqual(image.read_bytes(), raw)

            # 2) Non-existent image path collision (kills mutating resolve() == resolve()).
            ghost = root / "ghost.dll"
            ghost_alias = sub / ".." / "ghost.dll"
            code, out, err = _run_main([str(ghost), str(ghost_alias)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("collides with image", err)

            # 3) Hardlink alias where resolve() differs (kills mutating samefile()).
            hardlink = root / "upscaler_hardlink.dll"
            os.link(image, hardlink)
            self.assertNotEqual(image.resolve(), hardlink.resolve())
            self.assertTrue(image.samefile(hardlink))
            with mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest):
                code, out, err = _run_main([str(image), str(hardlink)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("collides with image", err)
            self.assertEqual(hardlink.read_bytes(), raw)

    def test_output_directory_existing_regular_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, _, digest = _write_synthetic_image(root)
            existing_file = root / "existing_output.bin"
            existing_file.write_bytes(b"preserve-me")

            with mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest):
                code, out, err = _run_main([str(image), str(existing_file)])
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertIn("is not a directory", err)
            self.assertEqual(existing_file.read_bytes(), b"preserve-me")

            proc = _run_cli_subprocess(str(image), str(existing_file))
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("is not a directory", proc.stderr)
            self.assertEqual(existing_file.read_bytes(), b"preserve-me")

    def test_os_replace_failure_leaves_zero_partial_bins_or_temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, _, _, digest = _write_synthetic_image(root)
            out_dir = root / "atomic_fail"
            replace_calls = []

            def failing_replace(src, dst):
                src_path = Path(src)
                replace_calls.append((src_path, Path(dst)))
                # Prove staging file actually resides in a TemporaryDirectory under out_dir.
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), out_dir.resolve())
                raise OSError("simulated replace failure")

            with (
                mock.patch.object(ffx_dispatch_profile, "UPSCALER", digest),
                mock.patch.object(ffx_dispatch_profile.os, "replace", side_effect=failing_replace),
            ):
                code, _, err = _run_main([str(image), str(out_dir)])

            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("simulated replace failure", err)
            self.assertEqual(len(replace_calls), 1)
            self.assertTrue(out_dir.is_dir())
            self.assertEqual(list(out_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
