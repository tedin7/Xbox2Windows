"""tests/pyc_main.py <module> behaves like running tests/<module>.py by path.

For pixel_aot_check and compute_store_check, a run that fails fast without DXC or a GPU must return the
same exit status through the entry as by path, show the forwarded arguments, and (pixel_aot_check) still
write the form-coverage record at exit.
"""
import os
import subprocess
import sys
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))
NO_SUCH_MODE = "no-such-mode-for-pyc-main-check"


def run(argv, record=None):
    env = dict(os.environ)
    if record:
        env["FORM_COVERAGE_RECORD"] = record
    return subprocess.run([sys.executable, "-B", *argv], capture_output=True, text=True, env=env,
                          timeout=120, cwd=TESTS)


def pixel_aot_forwards_status_args_and_record():
    # Missing arguments: this mode is rejected before the script runs its control suite, so both runs stay fast.
    args = ["draw-e64"]
    with tempfile.TemporaryDirectory() as tmp:
        direct_record = os.path.join(tmp, "direct.txt")
        entry_record = os.path.join(tmp, "entry.txt")
        direct = run([os.path.join(TESTS, "pixel_aot_check.py"), *args], direct_record)
        entry = run([os.path.join(TESTS, "pyc_main.py"), "pixel_aot_check", *args], entry_record)
        assert direct.returncode != 0, "control run unexpectedly passed"
        assert entry.returncode == direct.returncode, (entry.returncode, direct.returncode, entry.stderr[-400:])
        assert "check_e64_render() missing" in entry.stderr, entry.stderr[-400:]
        assert os.path.exists(entry_record), "exit record was not written"
        with open(direct_record, encoding="utf-8") as a, open(entry_record, encoding="utf-8") as b:
            assert a.read() == b.read()


def compute_store_forwards_status_and_args():
    direct = run([os.path.join(TESTS, "compute_store_check.py"), NO_SUCH_MODE])
    entry = run([os.path.join(TESTS, "pyc_main.py"), "compute_store_check", NO_SUCH_MODE])
    assert direct.returncode != 0, "control run unexpectedly passed"
    assert entry.returncode == direct.returncode, (entry.returncode, direct.returncode, entry.stderr[-400:])
    assert NO_SUCH_MODE in entry.stderr, entry.stderr[-400:]


def main():
    pixel_aot_forwards_status_args_and_record()
    compute_store_forwards_status_and_args()
    print("pyc_main forwards arguments, exit status and the exit record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
