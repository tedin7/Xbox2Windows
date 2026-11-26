"""Check tools/startup/launch.py's containment on a source-built probe: exit code, captured
stdout/stderr and trap log, the timeout kill, the commit cap and the missing-EXE path.

launch.py runs one executable in a kill-on-close job with a timeout, a commit cap and captured
stdout/stderr. tests/startup_launch_probe.cpp is the only input the check needs: it writes known
bytes, or holds memory, spawns a child or stays alive so a job limit has to stop it. No GPU and no
game files are involved.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "tools/startup/launch.py"

sys.path.insert(0, str(LAUNCH.parent))
import launch as launch_mod  # noqa: E402


def launch(probe, work, name, options, arguments, env=None):
    logdir = work / name
    logdir.mkdir()
    command = [sys.executable, "-B", str(LAUNCH), str(probe), str(work), str(logdir), *options, "--", *arguments]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    return result, logdir, time.monotonic() - started


def captured(logdir, name):
    path = logdir / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def run_unit_tests():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        # 1. X2W_STORAGE_PATH fallback creation when unset/empty and default missing.
        logdir = root / "run_unset"
        logdir.mkdir()
        env = {}
        msg = launch_mod.prepare_storage_path(env, str(logdir), default_storage=str(root / "missing_drive"),
                                              exists_fn=lambda _: False)
        expected_storage = str(logdir / "storage")
        assert msg and expected_storage in msg, msg
        assert env.get("X2W_STORAGE_PATH") == expected_storage, env
        assert (logdir / "storage").is_dir()

        env_empty = {"X2W_STORAGE_PATH": "   "}
        logdir_empty = root / "run_empty"
        logdir_empty.mkdir()
        msg_empty = launch_mod.prepare_storage_path(env_empty, str(logdir_empty), exists_fn=lambda _: False)
        assert msg_empty and str(logdir_empty / "storage") in msg_empty, msg_empty
        assert env_empty["X2W_STORAGE_PATH"] == str(logdir_empty / "storage")

        # 2. X2W_STORAGE_PATH preservation when already set or when default drive exists.
        logdir_set = root / "run_set"
        logdir_set.mkdir()
        custom = str(root / "custom_storage")
        env_set = {"X2W_STORAGE_PATH": custom}
        assert launch_mod.prepare_storage_path(env_set, str(logdir_set), exists_fn=lambda _: False) is None
        assert env_set["X2W_STORAGE_PATH"] == custom
        assert not (logdir_set / "storage").exists()

        logdir_default = root / "run_default"
        logdir_default.mkdir()
        env_default = {}
        assert launch_mod.prepare_storage_path(env_default, str(logdir_default), exists_fn=lambda _: True) is None
        assert "X2W_STORAGE_PATH" not in env_default
        assert not (logdir_default / "storage").exists()

        # 3. Stale X2W_SHADER_CACHE warning for non-empty dir vs silence for empty/non-existent/unset.
        non_empty_cache = root / "cache_non_empty"
        non_empty_cache.mkdir()
        cached_file = non_empty_cache / "req_abc.dxil"
        cached_file.write_text("cached", encoding="utf-8")
        warn = launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": str(non_empty_cache)})
        assert warn and "stale" in warn and str(non_empty_cache) in warn, warn
        assert cached_file.exists(), "shader cache check must not delete cached files"

        empty_cache = root / "cache_empty"
        empty_cache.mkdir()
        assert launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": str(empty_cache)}) is None
        assert launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": str(root / "cache_missing")}) is None
        assert launch_mod.check_shader_cache_warning({}) is None

        # 3b. Relative cache resolves against the child cwd; unlistable cache never raises.
        child = root / "child_cwd"
        (child / "rel_cache").mkdir(parents=True)
        (child / "rel_cache" / "req.dxil").write_text("cached", encoding="utf-8")
        assert launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": "rel_cache"}, str(child)), "relative cache not resolved against cwd"
        assert launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": "rel_cache"}, str(root)) is None
        real_scandir = os.scandir
        def denied(_path):
            raise PermissionError("denied")
        os.scandir = denied
        try:
            assert launch_mod.check_shader_cache_warning({"X2W_SHADER_CACHE": str(non_empty_cache)}) is None
        finally:
            os.scandir = real_scandir

        # 4. Fast-fail diagnostic emission on quick non-zero exit vs silence on zero exit or slow exit.
        diag = launch_mod.format_fast_fail_diagnostic(
            7, 0.25, str(logdir / "trap.log"),
            {"X2W_STORAGE_PATH": expected_storage, "X2W_SHADER_CACHE": str(non_empty_cache)},
        )
        assert diag and "0x00000007" in diag and "trap.log" in diag, diag
        assert expected_storage in diag and str(non_empty_cache) in diag, diag

        assert launch_mod.format_fast_fail_diagnostic(0, 0.10, str(logdir / "trap.log"), {}) is None
        assert launch_mod.format_fast_fail_diagnostic(7, 1.00, str(logdir / "trap.log"), {}) is None
        assert launch_mod.format_fast_fail_diagnostic(7, 2.50, str(logdir / "trap.log"), {}) is None

        # 5. Arguments reach the child unchanged: a trailing backslash must not swallow the closing
        # quote, and embedded quotes are escaped. Expected command lines are written out by hand.
        for argument, quoted in (
            ("plain", '"plain"'),
            ("", '""'),
            ("with space", '"with space"'),
            ("with\ttab", '"with\ttab"'),
            ("D:\\saves\\", '"D:\\saves\\\\"'),
            ("S:\\", '"S:\\\\"'),
            ("C:\\dir\\file.txt", '"C:\\dir\\file.txt"'),
            ('say "hi"', '"say \\"hi\\""'),
            ('""', '"\\"\\""'),
            ('a\\"b', '"a\\\\\\"b"'),
            ("x\\\\\\", '"x\\\\\\\\\\\\"'),
        ):
            assert launch_mod.quote_argument(argument) == quoted, (argument, launch_mod.quote_argument(argument), quoted)
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as wintypes
            shell32 = ctypes.WinDLL("shell32", use_last_error=True)
            shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
            shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.LocalFree.restype = wintypes.HLOCAL
            kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
            arguments = ["C:\\Program Files\\app.exe", "D:\\saves\\", "with\ttab", 'say "hi"', '""', 'a\\"b', "x\\\\", "", "plain"]
            count = ctypes.c_int()
            parsed = shell32.CommandLineToArgvW(" ".join(launch_mod.quote_argument(a) for a in arguments), ctypes.byref(count))
            assert parsed, f"CommandLineToArgvW failed: error {ctypes.get_last_error()}"
            try:
                assert [parsed[i] for i in range(count.value)] == arguments, [parsed[i] for i in range(count.value)]
            finally:
                kernel32.LocalFree(parsed)

    print("launch-unit: storage fallback/preservation, stale shader cache warning, fast-fail diagnostics and argument quoting passed")
    return 0


def main(probe):
    run_unit_tests()
    probe = Path(probe).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)

        # A finished non-zero run in under 1 s: reports exit code, streams, trap log and fast-fail diagnostic.
        base_env = {k: v for k, v in os.environ.items() if k not in ("X2W_STORAGE_PATH", "X2W_SHADER_CACHE")}
        result, logdir, _ = launch(probe, work, "report", ["--timeout", "10"], ["report", "7"], env=base_env)
        assert result.returncode == 0, result.stderr
        assert "exited" in result.stdout and "exit code 0x00000007" in result.stdout, result.stdout
        assert "peak job memory" in result.stdout, result.stdout
        assert "fast-fail:" in result.stdout and "trap.log" in result.stdout, result.stdout
        assert "X2W_STORAGE_PATH=" in result.stdout and "X2W_SHADER_CACHE=" in result.stdout, result.stdout
        if not os.path.exists("S:\\"):
            assert "created fallback storage:" in result.stdout, result.stdout
            assert (logdir / "storage").is_dir()
        assert captured(logdir, "stdout.txt") == "launch-probe stdout\n", captured(logdir, "stdout.txt")
        assert captured(logdir, "stderr.txt") == "launch-probe stderr\n", captured(logdir, "stderr.txt")
        assert captured(logdir, "trap.log") == "x2w-trap launch-probe\n", captured(logdir, "trap.log")

        # Quick zero exit with explicit X2W_STORAGE_PATH and empty X2W_SHADER_CACHE: no fast-fail, no fallback, no warning.
        empty_cache = work / "empty_cache"
        empty_cache.mkdir()
        custom_storage = work / "custom_storage"
        custom_storage.mkdir()
        clean_env = dict(base_env, X2W_STORAGE_PATH=str(custom_storage), X2W_SHADER_CACHE=str(empty_cache))
        result, logdir, _ = launch(probe, work, "noop", ["--timeout", "10"], ["noop"], env=clean_env)
        assert result.returncode == 0, result.stderr
        assert "fast-fail:" not in result.stdout, result.stdout
        assert "created fallback storage:" not in result.stdout, result.stdout
        assert "warning: X2W_SHADER_CACHE" not in result.stdout, result.stdout
        assert not (logdir / "storage").exists()

        # Non-empty X2W_SHADER_CACHE emits a warning without deleting cached files.
        stale_cache = work / "stale_cache"
        stale_cache.mkdir()
        marker = stale_cache / "shader.dxil"
        marker.write_text("dxil", encoding="utf-8")
        stale_env = dict(clean_env, X2W_SHADER_CACHE=str(stale_cache))
        result, logdir, _ = launch(probe, work, "stale-cache", ["--timeout", "10"], ["noop"], env=stale_env)
        assert result.returncode == 0, result.stderr
        assert "warning: X2W_SHADER_CACHE" in result.stdout and "stale" in result.stdout, result.stdout
        assert marker.exists()

        # A child that overstays the timeout is terminated, not left running (and >= 1 s exit is not fast-fail).
        result, logdir, elapsed = launch(probe, work, "timeout", ["--timeout", "1"], ["sleep", "30000"], env=clean_env)
        assert result.returncode == 0, result.stderr
        assert "killed after 1 s timeout" in result.stdout, result.stdout
        assert "exit code 0xdead0001" in result.stdout, result.stdout
        assert "fast-fail:" not in result.stdout, result.stdout
        assert elapsed < 20, f"the launcher took {elapsed:.1f} s to enforce a 1 s timeout"
        assert captured(logdir, "stdout.txt") == "launch-probe sleeping\n", captured(logdir, "stdout.txt")

        # The commit cap is a ceiling on the child; the same request fits under a larger cap.
        result, logdir, _ = launch(probe, work, "cap", ["--timeout", "10", "--memory-gib", "0.5"], ["allocate", "1024"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt").startswith("allocate failed error="), captured(logdir, "stdout.txt")
        result, logdir, _ = launch(probe, work, "cap-control", ["--timeout", "10", "--memory-gib", "4"], ["allocate", "1024"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt") == "allocate ok 1024 MiB\n", captured(logdir, "stdout.txt")

        # The process limit contains helpers the child starts; a larger limit lets the same helper run.
        result, logdir, _ = launch(probe, work, "one-process", ["--timeout", "10", "--max-processes", "1"], ["spawn"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt").startswith("spawn failed error="), captured(logdir, "stdout.txt")
        result, logdir, _ = launch(probe, work, "two-processes", ["--timeout", "10", "--max-processes", "2"], ["spawn"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt") == "spawn ok\n", captured(logdir, "stdout.txt")

        # X2W_LOW_ADDRESSES=1 starts the child with bottom-up ASLR off, so its blocks stay below 2^40 (the package
        # acceleration-structure builder refuses a destination with bits set above bit 39, #1758).
        limit = 1 << 40

        def highest(env_extra, name):
            env = dict(os.environ, **env_extra)
            result, logdir, _ = launch(probe, work, name, ["--timeout", "20"], ["addresses"], env=env)
            assert result.returncode == 0, result.stderr
            text = captured(logdir, "stdout.txt")
            assert text and text.startswith("addresses highest=0x"), (text, result.stdout)
            return int(text.split("=")[1], 16), result.stdout

        for attempt in range(4):
            top, out = highest({"X2W_LOW_ADDRESSES": "1"}, f"low-{attempt}")
            assert top < limit, f"with X2W_LOW_ADDRESSES=1 a block ends at 0x{top:x}, above 2^40"
            assert "bottom-up ASLR off" in out, out
        # Control: without the switch, the same child gets blocks above 2^40 on some runs (the default ASLR picks a random start).
        above = [highest({}, f"default-{attempt}")[0] >= limit for attempt in range(8)]
        assert any(above), "the control never saw a block above 2^40, so the check above proves nothing on this host"

        # A missing executable is reported instead of looking like a finished run.
        logdir = work / "missing"
        logdir.mkdir()
        result = subprocess.run([sys.executable, "-B", str(LAUNCH), str(work / "missing.exe"), str(work), str(logdir),
                                 "--timeout", "10"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
        assert "CreateProcess failed: error" in result.stdout, result.stdout

        print("launch: exit code, stdout/stderr/trap capture, 1 s timeout kill, 0.5 GiB commit cap "
              "(and its 4 GiB control), 1-process limit (and its 2-process control) and the missing-EXE path passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--unit":
        sys.exit(run_unit_tests())
    if len(sys.argv) != 2:
        sys.exit("usage: startup_launch_check.py <startup_launch_probe.exe> | --unit")
    sys.exit(main(sys.argv[1]))