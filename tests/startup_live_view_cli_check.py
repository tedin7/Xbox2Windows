#!/usr/bin/env python3
"""CLI, safe-import, path-guard, OSError, and mutation suite for tools/startup/live_view.py."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup" / "live_view.py"
sys.path.insert(0, str(ROOT / "tools" / "startup"))


def _run_main(live_view_mod, argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = live_view_mod.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def _short_path(path: Path) -> str | None:
    if os.name != "nt":
        return None
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if length else None


class _ReconfigurableStringIO(io.StringIO):
    def __init__(self):
        super().__init__()
        self.reconfigure_calls = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)


class LiveViewCliPositiveTests(unittest.TestCase):
    def test_safe_import_has_zero_side_effects(self) -> None:
        # Importing live_view with no CLI args must not touch sys.argv, sys.stdout, os.system, or block.
        fake_stdout = _ReconfigurableStringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [str(TOOL)]),
            mock.patch("os.system", side_effect=AssertionError("os.system called on import")) as sys_mock,
            mock.patch("time.sleep", side_effect=AssertionError("time.sleep called on import")) as sleep_mock,
            redirect_stdout(fake_stdout),
            redirect_stderr(stderr),
        ):
            if "live_view" in sys.modules:
                mod = importlib.reload(sys.modules["live_view"])
            else:
                mod = importlib.import_module("live_view")
        sys_mock.assert_not_called()
        sleep_mock.assert_not_called()
        self.assertEqual(fake_stdout.reconfigure_calls, [])
        self.assertEqual(fake_stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(callable(mod.main))

    def test_main_and_subprocess_preserve_output_reconfigure_and_split_utf8(self) -> None:
        import live_view

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rundir = root / "run"
            rundir.mkdir()
            pkg_dir = root / "pkg"
            pkg_dir.mkdir()
            (rundir / "events.txt").write_bytes(
                b"1.0 load engine.dll at 0x1000\n"
                b"2.0 debug-output hello from runtime\n"
                b"3.0 exit 0x0\n"
            )
            (rundir / "files.txt").write_bytes(
                b"1.0 NtCreateFile G:\\caf\xc3\xa9.bin from 0x1000 -> 0x00000000\r\n"
                b"2.0 NtOpenFile " + str(pkg_dir.resolve()).encode("utf-8") + b"\\data.bin from 0x1000 -> 0xc0000034\r\n"
                b"3.0 NtCreateFile C:\\Windows\\sys.dll from 0x1000 -> 0x00000000\r\n"
            )
            (rundir / "trap.log").write_text(
                "x2w-d3d12 heartbeat: 30 frames presented in 1.0 s (30.0 fps overall)\n",
                encoding="utf-8",
            )
            (rundir / "finished.txt").write_text("clean exit\n", encoding="utf-8")

            reconf_out = _ReconfigurableStringIO()
            stderr = io.StringIO()
            with (
                mock.patch("os.system") as system_mock,
                mock.patch("time.sleep") as sleep_mock,
                redirect_stdout(reconf_out),
                redirect_stderr(stderr),
            ):
                code = live_view.main([str(rundir), str(pkg_dir)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(reconf_out.reconfigure_calls, [{"encoding": "utf-8", "errors": "replace"}])
            system_mock.assert_called_once()
            self.assertEqual(sleep_mock.call_count, 10)  # 10s countdown loop (0 polling sleeps since finished.txt exists)

            out = reconf_out.getvalue()
            self.assertIn("Xbox2Windows local startup experiment: live log of ", out)
            self.assertIn("module    engine.dll", out)
            self.assertIn("runtime   hello from runtime", out)
            self.assertIn("EXIT      3.0 exit 0x0", out)
            self.assertIn("file      G:\\café.bin", out)
            self.assertIn("data.bin   (failed 0xc0000034)", out)
            self.assertNotIn("sys.dll", out)
            self.assertIn("frames    30 frames presented, 30.0 fps", out)
            self.assertIn("clean exit", out)
            self.assertIn("Run ended. Full logs are in ", out)

            # Verify sys.argv forwarding when argv=None and __main__ execution via runpy and subprocess.
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(rundir)]),
                mock.patch("os.system"),
                mock.patch("time.sleep"),
            ):
                code_none, out_none, err_none = _run_main(live_view, None)
            self.assertIs(type(code_none), int)
            self.assertEqual(code_none, 0)
            self.assertEqual(err_none, "")
            self.assertIn("Run ended.", out_none)

            runpy_out, runpy_err = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(rundir)]),
                mock.patch("os.system"),
                mock.patch("time.sleep"),
                redirect_stdout(runpy_out),
                redirect_stderr(runpy_err),
            ):
                try:
                    runpy.run_path(str(TOOL), run_name="__main__")
                except SystemExit as exc:
                    self.assertEqual(exc.code, 0)
            self.assertEqual(runpy_err.getvalue(), "")
            self.assertIn("Run ended.", runpy_out.getvalue())

            # Subprocess check (with finished.txt present, runs ~10s countdown cleanly and exits 0).
            proc = _run_subprocess(rundir, pkg_dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("Run ended.", proc.stdout)
            self.assertEqual(
                sorted(p.name for p in rundir.iterdir()),
                ["events.txt", "files.txt", "finished.txt", "trap.log"],
            )

    def test_short_path_alias_matches_resolved_package_file(self) -> None:
        import live_view

        with TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            package.mkdir()
            short = _short_path(package)
            if not short or os.path.normcase(short) == os.path.normcase(os.path.abspath(package)):
                self.skipTest("the temp volume has no distinct 8.3 alias")
            viewer = live_view._LogViewer(short)
            out = io.StringIO()
            with redirect_stdout(out):
                viewer.on_file(
                    "1.0 NtOpenFile "
                    + str(package.resolve())
                    + "\\data.bin from 0x1000 -> 0xc0000034"
                )
            self.assertIn("data.bin   (failed 0xc0000034)", out.getvalue())


class LiveViewCliNegativeAndMutationTests(unittest.TestCase):
    def test_rundir_and_package_path_guards_exit_2_without_looping(self) -> None:
        import live_view

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_rundir = root / "run"
            valid_rundir.mkdir()
            regular_file = root / "not_a_dir.txt"
            regular_file.write_text("payload", encoding="utf-8")
            missing_dir = root / "missing_dir"

            for argv, frag, check_subproc in (
                ([str(missing_dir)], "directory", True),
                ([str(regular_file)], "directory", True),
                (["", ], "directory", False),
                ([str(valid_rundir), str(regular_file)], "package", True),
            ):
                with self.subTest(argv=argv):
                    with mock.patch("time.sleep", side_effect=AssertionError("must not enter polling loop")):
                        code, out, err = _run_main(live_view, argv)
                    self.assertEqual(code, 2, err)
                    self.assertEqual(out, "")
                    self.assertNotIn("Traceback", err)
                    self.assertTrue(err.startswith("live_view: "), err)
                    self.assertIn(frag, err)

                    if check_subproc:
                        proc = _run_subprocess(*argv)
                        self.assertEqual(proc.returncode, 2, proc.stderr)
                        self.assertEqual(proc.stdout, "")
                        self.assertNotIn("Traceback", proc.stderr)
                        self.assertTrue(proc.stderr.startswith("live_view: "), proc.stderr)
                        self.assertIn(frag, proc.stderr)

    def test_oserror_caught_cleanly_and_py314_argparse_errors(self) -> None:
        import live_view

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_rundir = root / "run"
            valid_rundir.mkdir()

            with mock.patch("os.system", side_effect=OSError("simulated console failure")):
                code, out, err = _run_main(live_view, [str(valid_rundir)])
            self.assertEqual(code, 1, err)
            self.assertEqual(out, "")
            self.assertNotIn("Traceback", err)
            self.assertTrue(err.startswith("live_view: "), err)
            self.assertIn("simulated console failure", err)

            for argv, frag in (
                ([], "usage:"),
                ([str(valid_rundir), str(root), "extra_arg"], "extra_arg"),
                ([str(valid_rundir), "--unknown-flag"], "--unknown-flag"),
            ):
                with self.subTest(argv=argv):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(stdout), redirect_stderr(stderr):
                            live_view.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn(frag, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

            proc = _run_subprocess()
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertIn("usage:", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
