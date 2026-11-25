"""CLI validation and helper unit suite for tools/startup/launch.py."""

from contextlib import redirect_stderr, redirect_stdout
import ctypes
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup" / "launch.py"
sys.path.insert(0, str(TOOL.parent))
import launch  # noqa: E402


class _FakeKernel32:
    def __init__(self, *, create_ok=True, set_info_ok=True, assign_ok=True, wait_result=0, exit_code=0):
        self.create_ok = create_ok
        self.set_info_ok = set_info_ok
        self.assign_ok = assign_ok
        self.wait_result = wait_result
        self.exit_code = exit_code
        self.next_handle = 100
        self.open_handles = set()
        self.closed_handles = []
        self.error_modes = []
        self.job_limits = []
        self.cmdlines = []
        self.wait_timeouts = []
        self.terminated_jobs = []
        self.terminated_processes = []
        self.resumed_threads = []

    def _alloc_handle(self):
        self.next_handle += 1
        self.open_handles.add(self.next_handle)
        return self.next_handle

    def SetErrorMode(self, mode):
        self.error_modes.append(mode)
        return 0

    def CreateJobObjectW(self, _sa, _name):
        return self._alloc_handle()

    def SetInformationJobObject(self, _job, _cls, ptr, _size):
        info = ctypes.cast(ptr, ctypes.POINTER(launch.EXTENDED_LIMIT)).contents
        self.job_limits.append((info.Basic.LimitFlags, info.Basic.ActiveProcessLimit, info.JobMemoryLimit))
        return 1 if self.set_info_ok else 0

    def CreateFileW(self, *_args):
        return self._alloc_handle()

    def CreateProcessW(self, _exe, cmdline, *_args):
        if not self.create_ok:
            return 0
        self.cmdlines.append(cmdline.value)
        pi_ptr = _args[-1]
        pi = pi_ptr._obj
        pi.hProcess = self._alloc_handle()
        pi.hThread = self._alloc_handle()
        pi.dwProcessId = 4321
        pi.dwThreadId = 8765
        return 1

    def AssignProcessToJobObject(self, _job, _proc):
        return 1 if self.assign_ok else 0

    def TerminateProcess(self, proc, code):
        self.terminated_processes.append((proc.value, code))
        return 1

    def ResumeThread(self, thread):
        self.resumed_threads.append(thread.value)
        return 1

    def WaitForSingleObject(self, _proc, timeout_ms):
        self.wait_timeouts.append(timeout_ms)
        return self.wait_result

    def TerminateJobObject(self, job, code):
        self.terminated_jobs.append((job.value, code))
        return 1

    def GetExitCodeProcess(self, _proc, code_ptr):
        code_ptr._obj.value = self.exit_code
        return 1

    def QueryInformationJobObject(self, _job, _cls, ptr, _size, _ret):
        info = ctypes.cast(ptr, ctypes.POINTER(launch.EXTENDED_LIMIT)).contents
        info.PeakJobMemoryUsed = 4 * 2**20
        return 1

    def CloseHandle(self, handle):
        val = handle.value
        self.closed_handles.append(val)
        self.open_handles.discard(val)
        return 1


class LaunchCliPositiveTests(unittest.TestCase):
    def test_parse_args_defaults_options_and_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "app.exe"
            cwd = root / "work"
            logdir = root / "logs"

            parsed = launch.parse_args([str(exe), str(cwd), str(logdir)])
            self.assertEqual(parsed, (os.path.abspath(exe), os.path.abspath(cwd), os.path.abspath(logdir), 15.0, 16.0, 1, []))

            parsed_opts = launch.parse_args([
                str(exe),
                str(cwd),
                str(logdir),
                "--timeout",
                "2.5",
                "--memory-gib",
                "0.5",
                "--max-processes",
                "0",
                "--",
                "--flag",
                "two words",
            ])
            self.assertEqual(
                parsed_opts,
                (os.path.abspath(exe), os.path.abspath(cwd), os.path.abspath(logdir), 2.5, 0.5, 0, ["--flag", "two words"]),
            )

    def test_quote_argument_and_helper_functions(self):
        vectors = (
            ("plain", '"plain"'),
            ("", '""'),
            ("with space", '"with space"'),
            ("D:\\saves\\", '"D:\\saves\\\\"'),
            ('say "hi"', '"say \\"hi\\""'),
            ('a\\"b', '"a\\\\\\"b"'),
        )
        for raw, expected in vectors:
            self.assertEqual(launch.quote_argument(raw), expected)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()
            env = {}
            msg = launch.prepare_storage_path(env, str(logdir), default_storage=str(root / "missing"), exists_fn=lambda _: False)
            self.assertIsNotNone(msg)
            self.assertIn(str(logdir / "storage"), msg)
            self.assertEqual(env["X2W_STORAGE_PATH"], str(logdir / "storage"))
            self.assertTrue((logdir / "storage").is_dir())

            preserved = {"X2W_STORAGE_PATH": str(root / "custom")}
            self.assertIsNone(launch.prepare_storage_path(preserved, str(logdir), exists_fn=lambda _: False))
            self.assertEqual(preserved["X2W_STORAGE_PATH"], str(root / "custom"))

            cache_dir = root / "cache"
            cache_dir.mkdir()
            self.assertIsNone(launch.check_shader_cache_warning({"X2W_SHADER_CACHE": str(cache_dir)}))
            (cache_dir / "entry.dxil").write_bytes(b"dxil")
            warn = launch.check_shader_cache_warning({"X2W_SHADER_CACHE": "cache"}, cwd=str(root))
            self.assertIsNotNone(warn)
            self.assertIn("stale", warn)
            self.assertIn("cache", warn)

            diag = launch.format_fast_fail_diagnostic(7, 0.25, str(logdir / "trap.log"), env)
            self.assertIsNotNone(diag)
            self.assertIn("0x00000007", diag)
            self.assertIn("trap.log", diag)
            self.assertIn(str(logdir / "storage"), diag)
            self.assertIn("(unset)", diag)
            self.assertIsNone(launch.format_fast_fail_diagnostic(0, 0.25, str(logdir / "trap.log"), env))
            self.assertIsNone(launch.format_fast_fail_diagnostic(7, 1.0, str(logdir / "trap.log"), env))

    def test_main_stubbed_kernel32_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "probe.exe"
            cwd = root / "cwd"
            cwd.mkdir()
            logdir = root / "logs"
            logdir.mkdir()
            (logdir / "stdout.txt").write_text("child out\n", encoding="utf-8")
            (logdir / "stderr.txt").write_text("child err\n", encoding="utf-8")
            (logdir / "trap.log").write_text("stale trap\n", encoding="utf-8")

            fake = _FakeKernel32(wait_result=0, exit_code=0x7)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(launch, "k32", fake), redirect_stdout(stdout), redirect_stderr(stderr):
                rc = launch.main([
                    str(exe),
                    str(cwd),
                    str(logdir),
                    "--timeout",
                    "2.5",
                    "--memory-gib",
                    "0.5",
                    "--max-processes",
                    "3",
                    "--",
                    "arg one",
                    "D:\\dir\\",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(fake.open_handles, set())
            self.assertFalse((logdir / "trap.log").exists())
            self.assertEqual(fake.job_limits, [(0x2000 | 0x400 | 0x200 | 0x8, 3, int(0.5 * 2**30))])
            self.assertEqual(fake.wait_timeouts, [2500])
            self.assertIn('"arg one" "D:\\dir\\\\"', fake.cmdlines[0])
            self.assertIn("pid 4321 exited in", stdout.getvalue())
            self.assertIn("exit code 0x00000007", stdout.getvalue())
            self.assertIn("peak job memory 4.0 MiB", stdout.getvalue())
            self.assertIn("fast-fail: exited with 0x00000007", stdout.getvalue())
            self.assertIn("--- trap.log: (absent)", stdout.getvalue())

            # Timeout path kills the job with 0xDEAD0001 and closes all handles.
            fake_timeout = _FakeKernel32(wait_result=0x102, exit_code=0xDEAD0001)
            stdout_to = io.StringIO()
            with mock.patch.object(launch, "k32", fake_timeout), redirect_stdout(stdout_to):
                rc_to = launch.main([str(exe), str(cwd), str(logdir), "--timeout", "1", "--memory-gib", "0", "--max-processes", "0"])
            self.assertEqual(rc_to, 0)
            self.assertEqual(fake_timeout.open_handles, set())
            self.assertEqual(fake_timeout.job_limits, [(0x2000 | 0x400, 0, 0)])
            self.assertEqual(len(fake_timeout.terminated_jobs), 1)
            self.assertEqual(fake_timeout.terminated_jobs[0][1], 0xDEAD0001)
            self.assertIn("killed after 1 s timeout", stdout_to.getvalue())


class LaunchCliNegativeTests(unittest.TestCase):
    def _assert_exit_2(self, argv, expected_substr=""):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = launch.main(argv)
        self.assertEqual(rc, 2, (argv, stdout.getvalue(), stderr.getvalue()))
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip(), argv)
        if expected_substr:
            self.assertIn(expected_substr, stderr.getvalue())

    def test_rejects_missing_positionals_and_empty_args(self):
        for bad in (
            [],
            ["app.exe"],
            ["app.exe", "cwd"],
            ["app.exe", "cwd", "--", "extra"],
            ["", "cwd", "logs"],
            ["app.exe", "   ", "logs"],
            ["app.exe", "cwd", ""],
        ):
            with self.subTest(argv=bad):
                self._assert_exit_2(bad)

        with tempfile.TemporaryDirectory() as tmp:
            uncreated_logdir = Path(tmp) / "should_not_exist"
            self._assert_exit_2(["app.exe", str(tmp), "--", str(uncreated_logdir)])
            self.assertFalse(uncreated_logdir.exists())

    def test_rejects_unpaired_unknown_and_invalid_numeric_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [str(root / "app.exe"), str(root), str(root / "logs")]

            # Unpaired trailing option flags.
            for flag in ("--timeout", "--memory-gib", "--max-processes"):
                self._assert_exit_2([*base, flag], "requires a value")
            self._assert_exit_2([*base, "--timeout", "10", "--memory-gib"], "requires a value")

            # Unknown options and stray positional arguments before '--'.
            for unknown in ("--bogus", "--timeout=10", "-t", "stray"):
                self._assert_exit_2([*base, unknown], "unknown option")

            # Invalid --timeout (non-numeric, <= 0, NaN, inf).
            for bad_timeout in ("abc", "", "0", "-1", "-0.5", "nan", "NaN", "inf", "-inf",
                                "0.0009", "4294967.295", "1e300"):
                self._assert_exit_2([*base, "--timeout", bad_timeout], "--timeout")

            # Invalid --memory-gib (non-numeric, < 0, NaN, inf).
            for bad_mem in ("abc", "", "-1", "-0.01", "nan", "inf", "-inf", "1e-20", "1e300",
                            str(launch.SIZE_T_LIMIT // launch.GIB)):
                self._assert_exit_2([*base, "--memory-gib", bad_mem], "--memory-gib")

            # Invalid --max-processes (non-int, < 0).
            for bad_procs in ("abc", "", "1.5", "-1", "nan", "inf", "True", "4294967296"):
                self._assert_exit_2([*base, "--max-processes", bad_procs], "--max-processes")

            # Failed validation must not create logdir.
            self.assertFalse((root / "logs").exists())
            accepted = launch.parse_args([*base, "--timeout", "4294967.294", "--memory-gib", "0",
                                          "--max-processes", "4294967295"])
            self.assertEqual(int(accepted[3] * 1000), launch.DWORD_MAX - 1)
            self.assertEqual(accepted[4:6], (0, launch.DWORD_MAX))

    def test_stubbed_create_process_and_job_failures_clean_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "missing.exe"
            logdir = root / "logs"
            logdir.mkdir()

            # CreateProcessW failure preserves exact stdout message + exit 2 + closes handles.
            fake_create_fail = _FakeKernel32(create_ok=False)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(launch, "k32", fake_create_fail),
                mock.patch.object(launch.ctypes, "get_last_error", return_value=2),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                rc = launch.main([str(exe), str(root), str(logdir), "--timeout", "5"])
            self.assertEqual(rc, 2)
            self.assertIn("CreateProcess failed: error 2", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(fake_create_fail.open_handles, set())

            # SetInformationJobObject failure exits 2 without traceback and closes job handle.
            fake_set_fail = _FakeKernel32(set_info_ok=False)
            stderr_set = io.StringIO()
            with mock.patch.object(launch, "k32", fake_set_fail), redirect_stderr(stderr_set):
                rc_set = launch.main([str(exe), str(root), str(logdir)])
            self.assertEqual(rc_set, 2)
            self.assertNotIn("Traceback", stderr_set.getvalue())
            self.assertIn("SetInformationJobObject", stderr_set.getvalue())
            self.assertEqual(fake_set_fail.open_handles, set())

            # AssignProcessToJobObject failure terminates child, exits 2 without traceback, closes all handles.
            fake_assign_fail = _FakeKernel32(assign_ok=False)
            stderr_assign = io.StringIO()
            with mock.patch.object(launch, "k32", fake_assign_fail), redirect_stderr(stderr_assign):
                rc_assign = launch.main([str(exe), str(root), str(logdir)])
            self.assertEqual(rc_assign, 2)
            self.assertNotIn("Traceback", stderr_assign.getvalue())
            self.assertIn("AssignProcessToJobObject", stderr_assign.getvalue())
            self.assertEqual(len(fake_assign_fail.terminated_processes), 1)
            self.assertEqual(fake_assign_fail.open_handles, set())

    def test_subprocess_cli_rejections_and_missing_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()
            for bad_args in (
                ["only_one"],
                ["app.exe", str(root), str(logdir), "--timeout"],
                ["app.exe", str(root), str(logdir), "--timeout", "nan"],
                ["app.exe", str(root), str(logdir), "--unknown", "1"],
            ):
                proc = subprocess.run(
                    [sys.executable, "-B", str(TOOL), *bad_args],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 2, (bad_args, proc.stdout, proc.stderr))
                self.assertNotIn("Traceback", proc.stderr)
                self.assertTrue(proc.stderr.strip())

            if os.name == "nt":
                proc = subprocess.run(
                    [sys.executable, "-B", str(TOOL), str(root / "missing.exe"), str(root), str(logdir), "--timeout", "5"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 2, (proc.stdout, proc.stderr))
                self.assertIn("CreateProcess failed: error", proc.stdout)
                self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
