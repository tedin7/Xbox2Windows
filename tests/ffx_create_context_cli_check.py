#!/usr/bin/env python3
"""Pure-Python CLI, atomic-write, path-guard, and mutation suite for tools/ffx_create_context.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_DIR / "ffx_create_context.py"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))
import ffx_create_context  # noqa: E402
from ffx_create_context_check import build_synthetic_loader  # noqa: E402


SYNTHETIC_CPP = (
    '#include <cstdint>\n'
    'using word = uint64_t;\n'
    'extern "C" __declspec(dllexport) uint32_t create_context(void** output, const void* desc, '
    'const void* allocator, const void* host) { return 0; }\n'
)


def _run_main(module, argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class FfxCreateContextCliPositiveTests(unittest.TestCase):
    def test_atomic_cpp_output_in_nested_dirs_and_overwrite_with_zero_tmp(self) -> None:
        raw = build_synthetic_loader()
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "synthetic_loader.dll"
            loader.write_bytes(raw)
            nested_out = base / "nested" / "sub" / "create_context.cpp"
            self.assertFalse(nested_out.parent.exists())

            real_replace = os.replace
            replaced_pairs: list[tuple[Path, Path]] = []

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                replaced_pairs.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                self.assertEqual(src_p.suffix, ".tmp")
                return real_replace(src, dst)

            # 1) Create missing parent directories and write real lowered C++ atomically with 0 .tmp files.
            with (
                mock.patch.object(ffx_create_context, "LOADER", digest),
                mock.patch.object(ffx_create_context.os, "replace", side_effect=tracking_replace),
            ):
                expected_cpp = ffx_create_context.generate(loader)
                code, out, err = _run_main(ffx_create_context, [str(loader), str(nested_out)])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), expected_cpp)
            self.assertIn('extern "C" __declspec(dllexport) uint32_t create_context', expected_cpp)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["create_context.cpp"])
            self.assertEqual(len(replaced_pairs), 1)
            self.assertEqual(replaced_pairs[0][1].resolve(), nested_out.resolve())

            # 2) Atomic overwrite replaces stale content with 0 .tmp files.
            nested_out.write_text("stale content", encoding="ascii")
            updated_cpp = SYNTHETIC_CPP + "// updated\n"
            with mock.patch.object(ffx_create_context, "generate", return_value=updated_cpp):
                code_ow, out_ow, err_ow = _run_main(ffx_create_context, [str(loader), str(nested_out)])
            self.assertEqual(code_ow, 0, err_ow)
            self.assertEqual(out_ow, "")
            self.assertEqual(err_ow, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), updated_cpp)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["create_context.cpp"])

            # 3) argv=None forwards sys.argv cleanly.
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(loader), str(nested_out)]),
                mock.patch.object(ffx_create_context, "generate", return_value=SYNTHETIC_CPP),
            ):
                code_none, out_none, err_none = _run_main(ffx_create_context, None)
            self.assertEqual(code_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), SYNTHETIC_CPP)

            # 4) __main__ entrypoint propagates SystemExit(1) cleanly on unmatched loader.
            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(loader), str(base / "unused.cpp")]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout_m.getvalue(), "")
            self.assertNotIn("Traceback", stderr_m.getvalue())
            self.assertIn("version-query profile", stderr_m.getvalue())


class FfxCreateContextCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(ffx_create_context, argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_missing_dir_corrupt_and_hash_mismatch_loader_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dir_loader = base / "dir_loader"
            dir_loader.mkdir()
            absent_out = base / "absent_dir" / "create_context.cpp"

            with (
                mock.patch.object(
                    ffx_create_context,
                    "generate",
                    side_effect=AssertionError("generate called for invalid loader path"),
                ),
                mock.patch.object(
                    ffx_create_context,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid loader path"),
                ),
            ):
                self._assert_clean_failure(
                    [str(base / "missing.dll"), str(absent_out)],
                    expected_code=1,
                    fragments=("loader file not found",),
                )
                self._assert_clean_failure(
                    [str(dir_loader), str(absent_out)],
                    expected_code=1,
                    fragments=("loader path is a directory",),
                )

            self.assertFalse(absent_out.parent.exists())

            # Unmatched SHA-256 hash fails cleanly with 0 temps and 0 created dirs.
            raw = build_synthetic_loader()
            unmatched = base / "unmatched.dll"
            unmatched.write_bytes(raw)
            self._assert_clean_failure(
                [str(unmatched), str(absent_out)],
                expected_code=1,
                fragments=("version-query profile",),
            )
            self.assertFalse(absent_out.parent.exists())

            # Corrupt PE bytes (with matching SHA-256) fail cleanly without creating output parent.
            corrupt_payload = b"not a PE32+ image"
            corrupt_digest = hashlib.sha256(corrupt_payload).hexdigest()
            corrupt = base / "corrupt.dll"
            corrupt.write_bytes(corrupt_payload)
            with mock.patch.object(ffx_create_context, "LOADER", corrupt_digest):
                self._assert_clean_failure(
                    [str(corrupt), str(absent_out)],
                    expected_code=1,
                    fragments=("dos header",),
                    check_subprocess=False,
                )
            self.assertFalse(absent_out.parent.exists())
            self.assertEqual([p.name for p in base.iterdir() if p.suffix == ".tmp"], [])

    def test_output_guards_dir_file_parent_samefile_and_non_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "loader.dll"
            loader.write_bytes(b"synthetic-loader-bytes")
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(b"synthetic-cpp-bytes")
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(loader)
            out_dir = base / "existing_dir.cpp"
            out_dir.mkdir()
            sub_dir = base / "sub"
            sub_dir.mkdir()
            file_parent = base / "regular_parent.txt"
            file_parent.write_text("keep-me", encoding="ascii")

            with (
                mock.patch.object(
                    ffx_create_context,
                    "generate",
                    side_effect=AssertionError("generate called for invalid output path"),
                ),
                mock.patch.object(
                    ffx_create_context,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid output path"),
                ),
            ):
                for label, argv, code, frags in (
                    ("non_cpp_suffix", [str(loader), str(base / "out.txt")], 2, ("separate .cpp output",)),
                    ("empty_output", [str(loader), ""], 2, ("separate .cpp output",)),
                    ("same_path", [str(same_cpp), str(same_cpp)], 2, ("separate .cpp output",)),
                    ("resolve_alias", [str(same_cpp), str(sub_dir / ".." / "same.cpp")], 2, ("separate .cpp output",)),
                    ("hardlink_samefile", [str(loader), str(hardlink_cpp)], 2, ("separate .cpp output",)),
                    ("output_is_dir", [str(loader), str(out_dir)], 1, ("output path is a directory",)),
                    ("parent_is_file", [str(loader), str(file_parent / "create_context.cpp")], 1, ("not a directory",)),
                    (
                        "ancestor_is_file",
                        [str(loader), str(file_parent / "sub" / "create_context.cpp")],
                        1,
                        ("not a directory",),
                    ),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=code, fragments=frags)

            self.assertEqual(same_cpp.read_bytes(), b"synthetic-cpp-bytes")
            self.assertEqual(loader.read_bytes(), b"synthetic-loader-bytes")
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertEqual(file_parent.read_text(encoding="ascii"), "keep-me")
            self.assertFalse((base / "out.txt").exists())

    def test_output_symlink_rejected_before_generate_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "loader.dll"
            loader.write_bytes(b"synthetic-loader-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="ascii")

            # Unconditional unit check for output.is_symlink() guard even if OS lacks symlink privilege.
            with (
                mock.patch.object(Path, "is_symlink", return_value=True),
                mock.patch.object(
                    ffx_create_context,
                    "generate",
                    side_effect=AssertionError("generate called for symlink output"),
                ),
                mock.patch.object(
                    ffx_create_context,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                self._assert_clean_failure(
                    [str(loader), str(out_dir / "mock_link.cpp")],
                    expected_code=1,
                    fragments=("output path must not be a symlink",),
                    check_subprocess=False,
                )

            live_link = out_dir / "live_link.cpp"
            dangling_link = out_dir / "dangling_link.cpp"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.cpp")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(
                    ffx_create_context,
                    "generate",
                    side_effect=AssertionError("generate called for symlink output"),
                ),
                mock.patch.object(
                    ffx_create_context,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(loader), str(link)],
                            expected_code=1,
                            fragments=("output path must not be a symlink",),
                        )

            self.assertEqual(real_target.read_text(encoding="ascii"), "untouched")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.cpp", "live_link.cpp", "real.cpp"],
            )

    def test_os_replace_and_write_failures_preserve_output_and_leave_zero_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "loader.dll"
            loader.write_bytes(b"synthetic-loader-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "create_context.cpp"
            out_cpp.write_text("original", encoding="ascii")

            with (
                mock.patch.object(ffx_create_context, "generate", return_value=SYNTHETIC_CPP),
                mock.patch.object(ffx_create_context.os, "replace", side_effect=OSError("simulated replace failure")),
            ):
                self._assert_clean_failure(
                    [str(loader), str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )

            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["create_context.cpp"])

            # Non-ASCII payload fails during temp write; temp file is still unlinked and original preserved.
            with mock.patch.object(ffx_create_context, "generate", return_value="non-ascii-\u00e9"):
                self._assert_clean_failure(
                    [str(loader), str(out_cpp)],
                    expected_code=1,
                    fragments=("ascii",),
                    check_subprocess=False,
                )

            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["create_context.cpp"])

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "loader.dll"
            loader.write_bytes(b"synthetic")

            for argv, frags in (
                ([], ("required",)),
                ([str(loader)], ("required",)),
                ([str(loader), str(base / "create_context.cpp"), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "loader.dll"
            loader.write_bytes(b"synthetic-loader-bytes")
            dir_loader = base / "dir_loader"
            dir_loader.mkdir()
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(loader)
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep", encoding="ascii")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "create_context.cpp"
            out_cpp.write_text("original", encoding="ascii")

            def check_symlink_mutation(mod):
                with mock.patch.object(Path, "is_symlink", return_value=True):
                    _, _, err = _run_main(mod, [str(loader), str(out_dir / "sym.cpp")])
                self.assertNotIn("output path must not be a symlink", err.lower())

            # 1) Dropping loader.is_dir() guard misses "loader path is a directory".
            # 2) Dropping output.is_symlink() guard misses "output path must not be a symlink".
            # 3) Dropping parent.is_file() guard misses "output parent is not a directory".
            # 4) Dropping samefile() guard allows hardlink collision past _validate_paths.
            # 5) Dropping tmp_path.unlink() leaks a .tmp file on os.replace failure.
            for label, target, replacement, check_fn in (
                (
                    "drop_loader_is_dir_guard",
                    '    if loader.is_dir():\n        raise IsADirectoryError(f"loader path is a directory: {loader}")\n',
                    "    pass\n",
                    lambda mod: self.assertNotIn(
                        "loader path is a directory",
                        _run_main(mod, [str(dir_loader), str(out_dir / "a.cpp")])[2].lower(),
                    ),
                ),
                (
                    "drop_output_is_symlink_guard",
                    '    if output.is_symlink():\n        raise ValueError(f"output path must not be a symlink: {output}")\n',
                    "    pass\n",
                    check_symlink_mutation,
                ),
                (
                    "drop_parent_is_file_guard",
                    '        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):\n'
                    '            raise NotADirectoryError(f"output parent is not a directory: {parent}")\n',
                    "        pass\n",
                    lambda mod: self.assertNotIn(
                        "output parent is not a directory",
                        _run_main(mod, [str(loader), str(file_parent / "create_context.cpp")])[2].lower(),
                    ),
                ),
                (
                    "drop_samefile_guard",
                    "    if output.resolve() == loader.resolve() or (output.exists() and loader.samefile(output)):\n",
                    "    if output.resolve() == loader.resolve():\n",
                    lambda mod: self.assertNotIn(
                        "separate .cpp output",
                        _run_main(mod, [str(loader), str(hardlink_cpp)])[2].lower(),
                    ),
                ),
                (
                    "drop_tmp_unlink",
                    "        if tmp_path is not None:\n            tmp_path.unlink(missing_ok=True)\n",
                    "        pass\n",
                    lambda mod: self._verify_tmp_leaked_on_replace_fail(mod, loader, out_cpp, out_dir),
                ),
            ):
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mod = types.ModuleType(f"ffx_create_context_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                        with mock.patch.object(mod, "generate", return_value=SYNTHETIC_CPP):
                            check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)

    def _verify_tmp_leaked_on_replace_fail(self, mod, loader: Path, out_cpp: Path, out_dir: Path) -> None:
        with mock.patch.object(mod.os, "replace", side_effect=OSError("boom")):
            code, _, _ = _run_main(mod, [str(loader), str(out_cpp)])
        self.assertEqual(code, 1)
        leaked = [p for p in out_dir.iterdir() if p.suffix == ".tmp"]
        self.assertTrue(leaked, "expected unlinked .tmp file under mutation")
        for p in leaked:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
