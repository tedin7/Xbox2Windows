"""Run the XCRDAPI lifetime fixture with and without an escaping queue exception."""
import ctypes
import os
import re
import subprocess
import sys
import tempfile


WORKER_LINE = re.compile(r"^x2w-xcrdapi worker exception:", re.MULTILINE)
QUEUE_THROW = re.compile(
    r'^x2w-xcrdapi worker exception: role=queue type=[^\r\n]+ what="x2w test worker throw"$',
    re.MULTILINE,
)


def main(driver, dll):
    command = [driver, dll]
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ, TMP=temp, TEMP=temp)
        env.pop("X2W_TEST_XCRDAPI_THROW", None)
        options = dict(capture_output=True, text=True, errors="replace", timeout=40)
        clean = subprocess.run(command, env=env, **options)
        assert clean.returncode == 0, clean.stderr[-3000:]
        assert not WORKER_LINE.search(clean.stderr), clean.stderr[-3000:]

        # The child inherits the error mode. An unhandled exception must exit
        # even on a machine configured to show Windows Error Reporting dialogs.
        set_error_mode = ctypes.windll.kernel32.SetErrorMode
        previous = set_error_mode(0x0001 | 0x0002)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX
        try:
            crash = subprocess.run(command, env=dict(env, X2W_TEST_XCRDAPI_THROW="1"), **options)
        finally:
            set_error_mode(previous)
        assert crash.returncode != 0, "the worker exception did not end the child"
        assert len(WORKER_LINE.findall(crash.stderr)) == 1, crash.stderr[-3000:]
        assert QUEUE_THROW.search(crash.stderr), crash.stderr[-3000:]


if __name__ == "__main__":
    main(*sys.argv[1:])
