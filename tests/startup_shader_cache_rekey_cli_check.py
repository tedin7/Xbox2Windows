#!/usr/bin/env python3
"""Pure-Python CLI, atomic-publish, and error-guard suite for tools/startup/shader_cache_rekey.py."""

from contextlib import redirect_stderr, redirect_stdout
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
TOOLS_STARTUP = ROOT / "tools" / "startup"
TOOL = TOOLS_STARTUP / "shader_cache_rekey.py"
sys.path.insert(0, str(TOOLS_STARTUP))
import gfx2hlsl  # noqa: E402
import shader_cache_rekey as rekey_tool  # noqa: E402


def _write_synthetic_cache(src: Path) -> dict[str, int]:
    """Populate src with valid graphics, compute, RTV, and mesh request fixtures."""
    src.mkdir(parents=True, exist_ok=True)
    vs = struct.pack("<4I", 0x7E000280, 0x01020304, 1, 0xBF810000)
    ps = struct.pack("<3I", 0x7E020281, 7, 0xBF810000)
    root = b"RTS0 synthetic root"
    ia = b"synthetic layout"
    g_key = gfx2hlsl.pipeline_key(vs, ps, root, ia)
    for ext, blob in ((".a", vs), (".b", ps), (".root", root), (".ia", ia)):
        (src / f"req_{g_key:016x}{ext}").write_bytes(blob)

    c_parts = {
        ".desc": struct.pack("<12Q", *range(12)),
        ".sa": struct.pack("<7I", *range(7)),
        ".sb": b"\x05" * 20,
        ".data": struct.pack("<2Q", 0x1122334455667788, 0x99),
        ".root": b"RTS0 compute root",
        ".c67": struct.pack("<2I", 0x7E020280, 0xBF810000),
    }
    c_key = gfx2hlsl.compute_key(
        c_parts[".c67"],
        c_parts[".data"],
        c_parts[".root"],
        c_parts[".desc"],
        c_parts[".sa"],
        c_parts[".sb"],
    )
    for ext, blob in c_parts.items():
        (src / f"req_{c_key:016x}{ext}").write_bytes(blob)

    rtv_body = f"rtv1 {g_key} 28 87".encode("ascii")
    rtv_key = gfx2hlsl.fnv(rtv_body)
    (src / f"req_{rtv_key:016x}.rtv").write_bytes(rtv_body)
    (src / "req_0123456789abcdef.m86").write_bytes(b"\x01" * 8)
    return {"g": g_key, "c": c_key, "rtv": rtv_key}


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = rekey_tool.main(argv)
    return rc, stdout.getvalue(), stderr.getvalue()


class ShaderCacheRekeyCliTests(unittest.TestCase):
    def test_positive_main_argv_none_and_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "src_cache"
            dst_argv = base / "dst_argv"
            dst_none = base / "dst_none"
            dst_sub = base / "dst_sub"
            keys = _write_synthetic_cache(src)

            # Explicit argv -> 0 with X2W_TRANSLATE_JOBS=1 to keep in-process test fast.
            with mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}):
                rc, stdout, stderr = _run_main([str(src), str(dst_argv)])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn("not re-keyed (own key strings): m86 1", stdout)
            expected_names = sorted([
                f"req_{keys['g']:016x}.a",
                f"req_{keys['g']:016x}.b",
                f"req_{keys['g']:016x}.root",
                f"req_{keys['g']:016x}.ia",
                f"req_{keys['c']:016x}.desc",
                f"req_{keys['c']:016x}.sa",
                f"req_{keys['c']:016x}.sb",
                f"req_{keys['c']:016x}.data",
                f"req_{keys['c']:016x}.root",
                f"req_{keys['c']:016x}.c67",
                f"req_{keys['rtv']:016x}.rtv",
                gfx2hlsl.KEY_SEED_NAME,
            ])
            self.assertEqual(sorted(p.name for p in dst_argv.iterdir()), expected_names)

            # main(None) uses sys.argv[1:].
            with (
                mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}),
                mock.patch.object(sys, "argv", ["shader_cache_rekey.py", str(src), str(dst_none)]),
            ):
                rc, stdout, stderr = _run_main(None)
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertEqual(sorted(p.name for p in dst_none.iterdir()), expected_names)

            # Subprocess positive smoke test.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(src), str(dst_sub)],
                env={**os.environ, "X2W_TRANSLATE_JOBS": "1"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("not re-keyed (own key strings): m86 1", proc.stdout)
            self.assertEqual(sorted(p.name for p in dst_sub.iterdir()), expected_names)

    def test_usage_and_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "src"
            _write_synthetic_cache(src)
            reg_file = base / "regular.txt"
            reg_file.write_text("keep-me", encoding="utf-8")

            # Usage errors -> exit 2 without traceback.
            for argv in ([], ["only_one"], ["--help", str(base)]):
                with self.subTest(argv=argv):
                    rc, stdout, stderr = _run_main(argv)
                    self.assertEqual(rc, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn("usage:", stderr)
                    self.assertNotIn("Traceback", stderr)

            # Invalid src (missing / regular file), samefile, and regular-file dst -> exit 2.
            for label, bad_src, bad_dst in (
                ("missing_src", base / "missing_src", base / "out1"),
                ("file_src", reg_file, base / "out2"),
                ("same_dir", src, src),
                ("file_dst", src, reg_file),
            ):
                with self.subTest(case=label):
                    rc, stdout, stderr = _run_main([str(bad_src), str(bad_dst)])
                    self.assertEqual(rc, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn("error:", stderr)
                    self.assertNotIn("Traceback", stderr)
            self.assertEqual(reg_file.read_text(encoding="utf-8"), "keep-me")

    def test_io_error_guards_in_rekey(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "src"
            dst = base / "dst"
            _write_synthetic_cache(src)

            # os.listdir(src) OSError -> exit 1 without traceback.
            with mock.patch.object(rekey_tool.os, "listdir", side_effect=OSError("listdir denied")):
                rc, _, stderr = _run_main([str(src), str(dst)])
            self.assertEqual(rc, 1)
            self.assertIn("listdir denied", stderr)
            self.assertNotIn("Traceback", stderr)

            # os.makedirs(dst) OSError -> exit 1 without traceback.
            with (
                mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}),
                mock.patch.object(rekey_tool.os, "makedirs", side_effect=OSError("mkdir denied")),
            ):
                rc, _, stderr = _run_main([str(src), str(dst)])
            self.assertEqual(rc, 1)
            self.assertIn("mkdir denied", stderr)
            self.assertNotIn("Traceback", stderr)

            # Second-pass read_parts() OSError -> exit 1 without traceback.
            real_read_parts = rekey_tool.read_parts
            calls = {"count": 0}

            def fail_second_pass(directory, kind, key):
                calls["count"] += 1
                if calls["count"] > 2:
                    raise OSError("second-pass read failed")
                return real_read_parts(directory, kind, key)

            with (
                mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}),
                mock.patch.object(rekey_tool, "read_parts", side_effect=fail_second_pass),
            ):
                rc, _, stderr = _run_main([str(src), str(dst)])
            self.assertEqual(rc, 1)
            self.assertIn("second-pass read failed", stderr)
            self.assertNotIn("Traceback", stderr)

            # publish() OSError -> exit 1 without traceback.
            with (
                mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}),
                mock.patch.object(rekey_tool, "publish", side_effect=OSError("publish io failed")),
            ):
                rc, _, stderr = _run_main([str(src), str(dst)])
            self.assertEqual(rc, 1)
            self.assertIn("publish io failed", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_atomic_publish_cleanup_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out_dir = base / "out"
            out_dir.mkdir()
            existing = out_dir / "existing.bin"
            existing.write_bytes(b"original-bytes")
            fresh = out_dir / "fresh.bin"

            stats = {"written": 0, "kept": 0}
            # Existing target is kept without creating temp artifacts.
            rekey_tool.publish(str(existing), b"new-bytes", stats)
            self.assertEqual(stats, {"written": 0, "kept": 1})
            self.assertEqual(existing.read_bytes(), b"original-bytes")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["existing.bin"])

            # Fresh target succeeds with zero temp artifacts left behind.
            rekey_tool.publish(str(fresh), b"payload", stats)
            self.assertEqual(stats, {"written": 1, "kept": 1})
            self.assertEqual(fresh.read_bytes(), b"payload")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["existing.bin", "fresh.bin"])

            # os.replace failure cleans up TemporaryDirectory and leaves zero temp artifacts.
            failing = out_dir / "failing.bin"
            with mock.patch.object(rekey_tool.os, "replace", side_effect=OSError("replace boom")):
                with self.assertRaisesRegex(OSError, "replace boom"):
                    rekey_tool.publish(str(failing), b"never", stats)
            self.assertFalse(failing.exists())
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["existing.bin", "fresh.bin"])
            self.assertEqual(existing.read_bytes(), b"original-bytes")

            # Same failure through main() exits 1 cleanly with zero temp artifacts.
            src = base / "src"
            _write_synthetic_cache(src)
            dst_rekey = base / "dst_rekey"
            dst_rekey.mkdir()
            sentinel = dst_rekey / "sentinel.txt"
            sentinel.write_text("preserved", encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {"X2W_TRANSLATE_JOBS": "1"}),
                mock.patch.object(rekey_tool.os, "replace", side_effect=OSError("disk full")),
            ):
                rc, _, stderr = _run_main([str(src), str(dst_rekey)])
            self.assertEqual(rc, 1)
            self.assertIn("disk full", stderr)
            self.assertNotIn("Traceback", stderr)
            self.assertEqual(sorted(p.name for p in dst_rekey.iterdir()), ["sentinel.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")

            # Subprocess negative smoke test (regular-file dst -> exit 2).
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(src), str(sentinel)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("not a directory", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
