"""The parallel check runner reports every script's output and fails when any script fails."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "tools", "run_checks_parallel.py")


def run(*scripts, jobs="2"):
    return subprocess.run([sys.executable, "-B", RUNNER, "-j", jobs, *scripts],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def group(out, script, status):
    """Return the text between this script's group header and its endgroup line."""
    header = f"::group::{script} ({status}, "
    assert out.count(header) == 1, (header, out)
    start = out.index(header)
    return out[start:out.index("::endgroup::", start)]


with tempfile.TemporaryDirectory() as tmp:
    good = os.path.join(tmp, "good.py")
    bad = os.path.join(tmp, "bad.py")
    fail = os.path.join(tmp, "fail.py")
    with open(good, "w") as f:
        # The usage line promises `python -B <script>`.
        f.write("import sys\nassert sys.flags.dont_write_bytecode\nprint('good ran')\n")
    with open(bad, "w") as f:
        # A child may exit with a partial line on stderr.
        f.write("import sys\nsys.stderr.write('bad ran')\nsys.exit(3)\n")
    with open(fail, "w") as f:
        # An uncaught assertion exits 1 and writes its traceback to stderr, the usual way a check fails.
        f.write("assert False, 'fail ran'\n")
    ok = run(good, good)
    assert ok.returncode == 0 and ok.stdout.count("good ran") == 2, ok.stdout
    assert "all 2 checks passed" in ok.stdout, ok.stdout
    mixed = run(bad, good)
    assert mixed.returncode == 1, mixed.stdout
    assert "bad ran" in mixed.stdout and "good ran" in mixed.stdout and "FAILED: " + bad in mixed.stdout, mixed.stdout
    assert mixed.stdout.index("bad ran") < mixed.stdout.index("good ran"), "output must keep the given order"
    # Two failures: the status is still 1, the FAILED line names both in the given order, each
    # header carries its exit code, and stderr lands inside the failing script's own group.
    many = run(bad, fail, good)
    assert many.returncode == 1, many.stdout
    assert "FAILED: " + bad + " " + fail + "\n" in many.stdout, many.stdout
    assert group(many.stdout, bad, "exit 3").endswith("bad ran\n"), many.stdout
    assert "AssertionError: fail ran" in group(many.stdout, fail, "exit 1"), many.stdout
    assert group(many.stdout, good, "ok").endswith("good ran\n"), many.stdout
    assert many.stdout.splitlines().count("::endgroup::") == 3, many.stdout
    # Exit 77 is a skip: it is listed apart from failures, and the status stays 1 so a skip never
    # turns a run green.
    skip = os.path.join(tmp, "skip.py")
    with open(skip, "w") as f:
        f.write("import sys\nprint('SKIP missing tool')\nsys.exit(77)\n")
    skipped = run(skip, good)
    assert skipped.returncode == 1, skipped.stdout
    assert "SKIPPED: " + skip + "\n" in skipped.stdout, skipped.stdout
    assert "FAILED:" not in skipped.stdout and "checks passed" not in skipped.stdout, skipped.stdout
    assert group(skipped.stdout, skip, "exit 77").endswith("SKIP missing tool\n"), skipped.stdout
    both = run(skip, bad, good)
    assert both.returncode == 1, both.stdout
    assert "FAILED: " + bad + "\n" in both.stdout, both.stdout
    assert "SKIPPED: " + skip + "\n" in both.stdout, both.stdout
    # -j 0 is clamped to one worker instead of reaching ThreadPoolExecutor(max_workers=0).
    serial = run(good, jobs="0")
    assert serial.returncode == 0 and "all 1 checks passed" in serial.stdout, serial.stdout

# Per-command arguments and environment overrides must not affect siblings.
with tempfile.TemporaryDirectory() as tmp:
    echo = os.path.join(tmp, "echo.py")
    with open(echo, "w") as f:
        f.write("import os, sys\nprint(sys.argv[1], os.environ.get('RCPC_TEST_TOKEN', 'missing'))\n")
    checks = [
        ([echo, "first"], {"RCPC_TEST_TOKEN": "alpha"}),
        ([echo, "second"], None),
    ]
    driver = os.path.join(tmp, "driver.py")
    with open(driver, "w") as f:
        f.write("import runpy, sys\n")
        f.write("module = runpy.run_path(sys.argv[1], run_name='run_checks_parallel')\n")
        f.write("checks = " + repr(checks) + "\n")
        f.write("sys.exit(module['run_checks'](checks, jobs=2))\n")
    env = {**os.environ, "RCPC_TEST_TOKEN": "inherited"}
    out = subprocess.run([sys.executable, "-B", driver, RUNNER], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert out.returncode == 0, out.stdout
    # Both commands run the same script, so the groups tie on their header; check the bodies.
    assert out.stdout.count("::group::" + echo) == 2, out.stdout
    assert out.stdout.splitlines().count("::endgroup::") == 2, out.stdout
    assert out.stdout.index("first alpha") < out.stdout.index("second inherited"), out.stdout
    assert "first alpha\n" in out.stdout and "second inherited\n" in out.stdout, out.stdout
    assert "all 2 checks passed" in out.stdout, out.stdout
print("run_checks_parallel_check: ok")
