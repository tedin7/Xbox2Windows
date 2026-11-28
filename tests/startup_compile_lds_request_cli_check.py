#!/usr/bin/env python3
"""Offline unit, CLI, warm-reuse, path-guard, and mutation suite for tools/startup/compile_lds_request.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import warnings

ROOT = Path(__file__).resolve().parents[1]
TOOLS_STARTUP = ROOT / "tools" / "startup"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_STARTUP / "compile_lds_request.py"
sys.path.insert(0, str(TOOLS_STARTUP))
sys.path.insert(0, str(TESTS_DIR))
import compile_lds_request  # noqa: E402
from startup_compute_translate_check import state_a, write_request  # noqa: E402
from startup_gfx2hlsl_compute_check import CASES, USER7, TGID_X  # noqa: E402

COMPANION_EXTS = (".root", ".data", ".desc", ".sa", ".sb")
_REAL_SUBPROCESS_RUN = subprocess.run


def _run_main(module, argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with warnings.catch_warnings(), redirect_stdout(stdout), redirect_stderr(stderr):
        warnings.simplefilter("ignore", ResourceWarning)
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
    return _REAL_SUBPROCESS_RUN(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _setup_env(base: Path) -> tuple[Path, Path, Path, Path]:
    compiler = base / "fake_compile_dxil.exe"
    compiler.write_bytes(b"FAKE_DXIL_COMPILER")
    sdk = base / "sdk"
    sdk.mkdir()
    output = base / "output"
    cache = base / "cache"
    return compiler, sdk, output, cache


def _write_case_f(work: Path) -> tuple[int, Path]:
    _, code, rsrc2, group, _ = CASES["F"]
    key = write_request(work, code, state_a(group, rsrc2))
    return key, work / f"req_{key:016x}.c67"


def _fake_compile_dxil(calls: list[list[object]] | None = None, *, validate_paths: bool = True):
    def _runner(cmd, check=False, **_kwargs):
        assert check is True
        if calls is not None:
            calls.append(list(cmd))
        compiler, sdk, cache, profile, stage, bytecode = cmd
        if validate_paths:
            assert Path(compiler).is_file()
            assert Path(sdk).is_dir()
        assert profile == "cs_6_6"
        assert Path(stage).is_file()
        assert Path(cache).parent.exists() or not Path(cache).exists()
        Path(bytecode).write_bytes(b"DXBC_SYNTHETIC_CSO")
        return subprocess.CompletedProcess(cmd, 0)

    return _runner


class CompileLdsRequestPositiveTests(unittest.TestCase):
    def test_synthetic_c67_compile_warm_reuse_and_damaged_bundle_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, output, cache = _setup_env(base)
            key, marker = _write_case_f(req_dir)
            calls: list[list[object]] = []

            argv = [
                str(marker),
                str(output),
                "--compiler",
                str(compiler),
                "--sdk",
                str(sdk),
                "--cache",
                str(cache),
                "--rw-image",
                "0",
                "0",
                "--wave-size",
                "32",
            ]
            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil(calls)):
                code, out, err = _run_main(compile_lds_request, argv)

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(calls), 1)
            dest = Path(out.strip())
            self.assertTrue(dest.is_dir())
            self.assertEqual(list(output.iterdir()), [dest])

            hlsl = (dest / f"{key:016x}.cs.hlsl").read_text(encoding="utf-8")
            self.assertIn("[WaveSize(32)]\n[numthreads(", hlsl)
            self.assertIn("#pragma x2w_lds ", hlsl)
            self.assertIn("#pragma x2w_vop2 ", hlsl)
            self.assertIn("GroupMemoryBarrierWithGroupSync();", hlsl)
            self.assertEqual((dest / f"{key:016x}.cs.cso").read_bytes(), b"DXBC_SYNTHETIC_CSO")
            self.assertIn("group 64 1 1", (dest / f"{key:016x}.txt").read_text(encoding="utf-8").splitlines())

            replay_raw = (dest / "replay.json").read_bytes()
            replay = json.loads(replay_raw.decode("utf-8"))
            self.assertEqual(replay["schema"], 1)
            self.assertEqual(replay["profile"], "cs_6_6")
            self.assertEqual(replay["rw_image"], [0, 0])
            self.assertEqual(replay["wave_size"], 32)
            for rel, expected_sha in replay["sha256"].items():
                self.assertEqual(hashlib.sha256((dest / rel).read_bytes()).hexdigest(), expected_sha)

            # Warm reuse: identical inputs collide on dest, verify all sha256 entries, and succeed idempotently.
            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil(calls)):
                code_warm, out_warm, err_warm = _run_main(compile_lds_request, argv)
            self.assertEqual(code_warm, 0, err_warm)
            self.assertEqual(err_warm, "")
            self.assertEqual(Path(out_warm.strip()), dest)
            self.assertEqual(list(output.iterdir()), [dest])

            # Damaged artifact sha256 detection: corrupting .cs.cso fails cleanly with no temp leftovers.
            cso_path = dest / f"{key:016x}.cs.cso"
            saved_cso = cso_path.read_bytes()
            cso_path.write_bytes(b"CORRUPT_CSO")
            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()):
                code_dmg, out_dmg, err_dmg = _run_main(compile_lds_request, argv)
            self.assertEqual(code_dmg, 1)
            self.assertEqual(out_dmg, "")
            self.assertNotIn("Traceback", err_dmg)
            self.assertIn("damaged replay artifact", err_dmg)
            self.assertEqual(list(output.iterdir()), [dest])
            cso_path.write_bytes(saved_cso)

            # Corrupting replay.json on disk also triggers refusal cleanly.
            (dest / "replay.json").write_text("{corrupt}\n", encoding="utf-8")
            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()):
                code_rep, out_rep, err_rep = _run_main(compile_lds_request, argv)
            self.assertEqual(code_rep, 1)
            self.assertEqual(out_rep, "")
            self.assertNotIn("Traceback", err_rep)
            self.assertIn("damaged replay artifact replay.json", err_rep)
            self.assertEqual(list(output.iterdir()), [dest])
            (dest / "replay.json").write_bytes(replay_raw)

            # Wave size 64 and argv=None sys.argv forwarding.
            argv64 = argv[:-1] + ["64"]
            with (
                mock.patch.object(sys, "argv", [str(TOOL), *argv64]),
                mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()),
            ):
                code64, out64, err64 = _run_main(compile_lds_request, None)
            self.assertEqual(code64, 0, err64)
            self.assertEqual(err64, "")
            dest64 = Path(out64.strip())
            self.assertNotEqual(dest64, dest)
            self.assertIn("[WaveSize(64)]", (dest64 / f"{key:016x}.cs.hlsl").read_text(encoding="utf-8"))
            self.assertEqual(sorted(output.iterdir()), sorted([dest, dest64]))


class CompileLdsRequestNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(compile_lds_request, argv)
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

    def test_missing_companion_and_hostile_path_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, _, cache = _setup_env(base)
            key, marker = _write_case_f(req_dir)
            absent_out = base / "absent_parent" / "output"
            file_target = base / "regular_file"
            file_target.write_text("KEEP", encoding="utf-8")
            dir_path = base / "some_dir"
            dir_path.mkdir()

            def make_argv(
                *,
                m: Path = marker,
                o: Path = absent_out,
                comp: Path = compiler,
                s: Path = sdk,
                ca: Path = cache,
            ) -> list[str]:
                return [
                    str(m),
                    str(o),
                    "--compiler",
                    str(comp),
                    "--sdk",
                    str(s),
                    "--cache",
                    str(ca),
                    "--rw-image",
                    "0",
                    "0",
                    "--wave-size",
                    "32",
                ]

            with (
                mock.patch.object(
                    compile_lds_request.gfx,
                    "load_compute_request",
                    side_effect=AssertionError("load_compute_request called on invalid path"),
                ),
                mock.patch.object(
                    compile_lds_request.subprocess,
                    "run",
                    side_effect=AssertionError("subprocess.run called on invalid path"),
                ),
            ):
                # Marker missing / dir / non-regular.
                self._assert_clean_failure(
                    make_argv(m=req_dir / "req_0123456789abcdef.c67"),
                    fragments=("request marker", "not found"),
                )
                self._assert_clean_failure(
                    make_argv(m=dir_path / "req_0123456789abcdef.c67"),
                    fragments=("request marker", "not found"),
                )
                dir_marker = base / "req_0123456789abcdef.c67"
                dir_marker.mkdir()
                self._assert_clean_failure(make_argv(m=dir_marker), fragments=("request marker", "directory"))

                # Missing or directory companion files.
                for ext in COMPANION_EXTS:
                    comp_file = marker.with_suffix(ext)
                    saved = comp_file.read_bytes()
                    comp_file.unlink()
                    self._assert_clean_failure(make_argv(), fragments=(f"companion {ext}", "not found"))
                    comp_file.mkdir()
                    self._assert_clean_failure(make_argv(), fragments=(f"companion {ext}", "directory"))
                    comp_file.rmdir()
                    comp_file.write_bytes(saved)

                # Compiler missing / dir / non-regular.
                self._assert_clean_failure(make_argv(comp=base / "missing.exe"), fragments=("compiler", "not found"))
                self._assert_clean_failure(make_argv(comp=dir_path), fragments=("compiler", "directory"))

                # SDK missing / file.
                self._assert_clean_failure(make_argv(s=base / "missing_sdk"), fragments=("sdk", "not found"))
                self._assert_clean_failure(make_argv(s=file_target), fragments=("sdk", "not a directory"))

                # Output / cache existing regular file or file-backed parent.
                self._assert_clean_failure(make_argv(o=file_target), fragments=("output", "not a directory"))
                self._assert_clean_failure(
                    make_argv(o=file_target / "sub"),
                    fragments=("output parent", "not a directory"),
                )
                self._assert_clean_failure(make_argv(ca=file_target), fragments=("cache", "not a directory"))
                self._assert_clean_failure(
                    make_argv(ca=file_target / "sub"),
                    fragments=("cache parent", "not a directory"),
                )

                # Non-regular file input (mocked unconditionally).
                with mock.patch.object(Path, "is_file", return_value=False):
                    self._assert_clean_failure(
                        make_argv(),
                        fragments=("not a regular file",),
                        check_subprocess=False,
                    )

            self.assertFalse(absent_out.parent.exists())
            self.assertEqual(file_target.read_text(encoding="utf-8"), "KEEP")

            # Symlinked output/cache are the caller's chosen location and now
            # succeed: the bundle is written through the link into its target.
            link_out = base / "link_out"
            try:
                link_out.symlink_to(dir_path)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
            else:
                with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()):
                    code_link, out_link, err_link = _run_main(compile_lds_request, make_argv(o=link_out))
                self.assertEqual(code_link, 0, err_link)
                self.assertEqual(err_link, "")
                dest_link = Path(out_link.strip())
                self.assertEqual(dest_link.resolve().parent, dir_path)
                self.assertTrue((dest_link / f"{key:016x}.cs.hlsl").is_file())
                self.assertEqual(list(link_out.iterdir()), [dest_link])
                # The same bundle through a symlinked cache warm-reuses the
                # bundle written above instead of refusing the link.
                with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()):
                    code_ca, out_ca, err_ca = _run_main(compile_lds_request, make_argv(o=dir_path, ca=link_out))
                self.assertEqual(code_ca, 0, err_ca)
                self.assertEqual(err_ca, "")
                self.assertEqual(Path(out_ca.strip()), dest_link.resolve())

    def test_fifo_output_and_dangling_symlink_parent_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, output, cache = _setup_env(base)
            _, marker = _write_case_f(req_dir)

            def make_argv(*, o: Path = output, ca: Path = cache) -> list[str]:
                return [
                    str(marker),
                    str(o),
                    "--compiler",
                    str(compiler),
                    "--sdk",
                    str(sdk),
                    "--cache",
                    str(ca),
                    "--rw-image",
                    "0",
                    "0",
                    "--wave-size",
                    "32",
                ]

            ran = False
            # A FIFO is neither a file nor a directory: refused, not written through.
            if hasattr(os, "mkfifo"):
                fifo_out = base / "fifo_out"
                os.mkfifo(fifo_out)
                self._assert_clean_failure(make_argv(o=fifo_out), fragments=("output", "not a directory"))
                ran = True

            # A dangling symlink as a parent is not a directory: refused, not followed.
            dangling = base / "dangling"
            try:
                dangling.symlink_to(base / "no_such_target")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
            else:
                self._assert_clean_failure(
                    make_argv(o=dangling / "output"),
                    fragments=("output parent", "not a directory"),
                )
                self._assert_clean_failure(
                    make_argv(ca=dangling / "cache"),
                    fragments=("cache parent", "not a directory"),
                )
                ran = True

            if not ran:
                self.skipTest("neither FIFOs nor symlinks are available on this platform")

    def test_translator_bug_keeps_traceback_and_out_of_range_branch_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, output, cache = _setup_env(base)

            def make_argv(m: Path) -> list[str]:
                return [
                    str(m),
                    str(output),
                    "--compiler",
                    str(compiler),
                    "--sdk",
                    str(sdk),
                    "--cache",
                    str(cache),
                    "--rw-image",
                    "0",
                    "0",
                    "--wave-size",
                    "32",
                ]

            def write_modified_case(name: str, dword_index: int, value: int) -> Path:
                _, code, rsrc2, group, _ = CASES[name]
                words = list(struct.unpack(f"<{len(code) // 4}I", code))
                words[dword_index] = value
                key = write_request(req_dir, struct.pack(f"<{len(words)}I", *words), state_a(group, rsrc2))
                return req_dir / f"req_{key:016x}.c67"

            # A translator bug (bad literal) must keep its traceback instead of
            # being swallowed into a clean "error:" line.
            bad_marker = write_modified_case("F", 4, 0xbf84C07F)
            with self.assertRaises(ValueError) as raised:
                _run_main(compile_lds_request, make_argv(bad_marker))
            self.assertIn("invalid literal for int()", str(raised.exception))
            proc = _run_subprocess(*make_argv(bad_marker))
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertIn("Traceback", proc.stderr)
            self.assertIn("invalid literal for int()", proc.stderr)

            # An out-of-range branch target is refused cleanly as unsupported.
            branch_marker = write_modified_case("extracted-tid", 9, 0xbf852005)
            self._assert_clean_failure(
                make_argv(branch_marker),
                fragments=("branch target", "outside the compute stage"),
            )

    def test_cli_arg_errors_key_mismatch_conditional_barrier_and_compiler_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, output, cache = _setup_env(base)
            _, marker = _write_case_f(req_dir)

            def make_argv(m: Path = marker, rw: tuple[str, str] = ("0", "0"), wave: str = "32") -> list[str]:
                return [
                    str(m),
                    str(output),
                    "--compiler",
                    str(compiler),
                    "--sdk",
                    str(sdk),
                    "--cache",
                    str(cache),
                    "--rw-image",
                    rw[0],
                    rw[1],
                    "--wave-size",
                    wave,
                ]

            # Argparse / --rw-image / marker name errors exit 2 cleanly without tracebacks.
            for argv, frags in (
                (make_argv(m=req_dir / "bad_marker.c67"), ("req_<16 hex digits>.c67",)),
                (make_argv(rw=("-1", "0")), ("nonnegative",)),
                (make_argv(rw=("0", "-32")), ("nonnegative",)),
                (make_argv(rw=("0", "16")), ("32-byte-aligned",)),
                (make_argv(wave="16"), ("invalid choice",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(argv, expected_code=2, fragments=frags)

            # Compiler nonzero exit (subprocess.CalledProcessError) cleans temp dir and exits 1.
            def failing_run(cmd, check=False, **_kwargs):
                raise subprocess.CalledProcessError(2, cmd)

            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=failing_run):
                self._assert_clean_failure(make_argv(), expected_code=1, fragments=("returned non-zero exit status",), check_subprocess=False)
            self.assertEqual(list(output.iterdir()), [])

            # Conditional s_barrier in synthetic .c67 request is rejected cleanly.
            cond_dir = base / "cond"
            cond_dir.mkdir()
            # s_cbranch_scc0 1 (+4 bytes over s_barrier) ; s_barrier ; s_endpgm
            cond_code = bytes.fromhex("010084bf" "00008abf" "000081bf")
            cond_key = write_request(cond_dir, cond_code, state_a((64, 1, 1), USER7 | TGID_X | (1 << 15)))
            cond_marker = cond_dir / f"req_{cond_key:016x}.c67"
            with mock.patch.object(
                compile_lds_request.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run called for conditional barrier"),
            ):
                self._assert_clean_failure(
                    make_argv(m=cond_marker),
                    expected_code=1,
                    fragments=("workgroup barrier is conditional",),
                )
            self.assertEqual(list(output.iterdir()), [])

            # Damaged .data bytes fail key verification with expected message and __main__ propagates SystemExit(1).
            data_path = marker.with_suffix(".data")
            data_path.write_bytes(bytes(b ^ 1 for b in data_path.read_bytes()))
            self._assert_clean_failure(make_argv(), expected_code=1, fragments=("do not match the key",))
            self.assertEqual(list(output.iterdir()), [])

            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), *make_argv()]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout_m.getvalue(), "")
            self.assertNotIn("Traceback", stderr_m.getvalue())
            self.assertIn("do not match the key", stderr_m.getvalue())

    def test_fast_self_mutation_controls(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            req_dir = base / "req"
            req_dir.mkdir()
            compiler, sdk, output, cache = _setup_env(base)
            key, marker = _write_case_f(req_dir)
            file_target = base / "regular_file"
            file_target.write_text("KEEP", encoding="utf-8")

            cond_dir = base / "cond"
            cond_dir.mkdir()
            cond_code = bytes.fromhex("010084bf" "00008abf" "000081bf")
            cond_key = write_request(cond_dir, cond_code, state_a((64, 1, 1), USER7 | TGID_X | (1 << 15)))
            cond_marker = cond_dir / f"req_{cond_key:016x}.c67"

            def argv_for(m=marker, o=output, comp=compiler, s=sdk, ca=cache, rw=("0", "0")):
                return [
                    str(m), str(o),
                    "--compiler", str(comp),
                    "--sdk", str(s),
                    "--cache", str(ca),
                    "--rw-image", rw[0], rw[1],
                    "--wave-size", "32",
                ]

            # Build one valid bundle so drop_bundle_sha256_check can probe damaged artifact detection.
            with mock.patch.object(compile_lds_request.subprocess, "run", side_effect=_fake_compile_dxil()):
                code_init, out_init, err_init = _run_main(compile_lds_request, argv_for())
            self.assertEqual(code_init, 0, err_init)
            dest = Path(out_init.strip())
            cso_path = dest / f"{key:016x}.cs.cso"
            saved_cso = cso_path.read_bytes()

            def check_drop_marker_guard(mod):
                # Dropping the guard no longer refuses the missing marker
                # cleanly: the tool either reaches the copy and crashes on the
                # missing file, or reports a different error. Either way the
                # guard's refusal is gone, which is what this control detects.
                try:
                    with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil()):
                        _, _, err = _run_main(mod, argv_for(m=req_dir / "req_0123456789abcdef.c67"))
                except Exception:
                    return
                self.assertNotIn("request marker file not found", err.lower())

            def check_drop_companion_guard(mod):
                root_file = marker.with_suffix(".root")
                saved = root_file.read_bytes()
                root_file.unlink()
                try:
                    try:
                        with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil()):
                            _, _, err = _run_main(mod, argv_for())
                    except Exception:
                        return
                    self.assertNotIn("companion .root file not found", err.lower())
                finally:
                    root_file.write_bytes(saved)

            def check_drop_compiler_guard(mod):
                with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil(validate_paths=False)):
                    _, _, err = _run_main(mod, argv_for(comp=base / "missing.exe"))
                self.assertNotIn("compiler file not found", err.lower())

            def check_drop_sdk_guard(mod):
                with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil(validate_paths=False)):
                    _, _, err = _run_main(mod, argv_for(s=base / "missing_sdk"))
                self.assertNotIn("sdk directory not found", err.lower())

            def check_drop_rw_image_guard(mod):
                with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil()):
                    code, _, err = _run_main(mod, argv_for(rw=("-1", "0")))
                self.assertEqual(code, 0, err)
                self.assertNotIn("nonnegative", err.lower())

            def check_drop_bundle_sha256(mod):
                cso_path.write_bytes(b"DAMAGED")
                try:
                    with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil()):
                        code, _, err = _run_main(mod, argv_for())
                    self.assertEqual(code, 0, err)
                finally:
                    cso_path.write_bytes(saved_cso)

            def check_drop_conditional_barrier(mod):
                with mock.patch.object(mod.subprocess, "run", side_effect=_fake_compile_dxil()):
                    code, _, err = _run_main(mod, argv_for(m=cond_marker))
                self.assertNotIn("workgroup barrier is conditional", err.lower())
                self.assertEqual(code, 0, err)

            # Mutation targets are regular expressions, not exact source lines:
            # they pin the distinctive structure (call, guard, loop) without
            # depending on quote style, exception type or message wording, so
            # a reword or requote of a guard does not silently unpin the control.
            mutations = (
                (
                    "drop_marker_guard",
                    r"    _validate_regular_file\(args\.marker, [^\n]*\)\n",
                    "    pass\n",
                    check_drop_marker_guard,
                ),
                (
                    "drop_companion_guard",
                    r"    for ext in \([^\n]*\):\n"
                    r"        _validate_regular_file\(args\.marker\.with_suffix\(ext\), [^\n]*\)\n",
                    "    pass\n",
                    check_drop_companion_guard,
                ),
                (
                    "drop_compiler_guard",
                    r"    _validate_regular_file\(args\.compiler, [^\n]*\)\n",
                    "    pass\n",
                    check_drop_compiler_guard,
                ),
                (
                    "drop_sdk_guard",
                    r"    if not args\.sdk\.exists\(\):\n"
                    r"        raise [^\n]*\n"
                    r"    if not args\.sdk\.is_dir\(\):\n"
                    r"        raise [^\n]*\n",
                    "    pass\n",
                    check_drop_sdk_guard,
                ),
                (
                    "drop_rw_image_guard",
                    r"    if param < 0 or off < 0 or off ?% ?32:\n"
                    r"        parser\.error\([^\n]*\)\n",
                    "    pass\n",
                    check_drop_rw_image_guard,
                ),
                (
                    "drop_bundle_sha256",
                    r"                for relative, digest in metadata\[['\"]sha256['\"]\]\.items\(\):\n"
                    r"                    try:\n"
                    r"                        data = \(dest / relative\)\.read_bytes\(\)\n"
                    r"                    except OSError:\n"
                    r"                        raise [^\n]*\n"
                    r"                    if hashlib\.sha256\(data\)\.hexdigest\(\) != digest:\n"
                    r"                        raise [^\n]*\n",
                    "                pass\n",
                    check_drop_bundle_sha256,
                ),
                (
                    "drop_conditional_barrier",
                    r"                    if i not in post\[0\]:\n"
                    r"                        raise [^\n]*\n",
                    "                    pass\n",
                    check_drop_conditional_barrier,
                ),
            )
            for label, pattern, replacement, check_fn in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(len(re.findall(pattern, source)), 1, f"target not unique for {label}")
                    mod = types.ModuleType(f"compile_lds_request_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(re.sub(pattern, replacement, source, count=1), str(TOOL), "exec"), mod.__dict__)
                        check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
