"""Check fatal first-touch commit failure in a child process."""

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

from xmem_page_runs_check import MEM_COMMIT, MEM_RESERVE, PAGE, Shim


EXIT_CODE = 0xE2571455
AFTER_TOUCH = "after touch"
FAILURE_PREFIX = "x2w-xmem lazy commit failed: address="


def touch(path):
    shim = Shim(path)
    base = shim.whole
    pages = shim.pages(3)
    assert shim.contiguous(base, len(pages), pages) == base, C.get_last_error()
    address = base + PAGE
    assert shim.query(address)[3] == MEM_RESERVE
    print("pages before touch: " + " ".join(f"{shim.query(base + i * PAGE)[3]:#x}" for i in range(len(pages))), flush=True)
    print(f"touch address={address:#x}", flush=True)
    C.c_ubyte.from_address(address).value = 0xA5
    print(AFTER_TOUCH, flush=True)
    assert shim.query(address)[3] == MEM_COMMIT
    assert C.c_ubyte.from_address(address).value == 0xA5


def check(path, inject_failure):
    env = dict(os.environ, X2W_XMEM_LAZY_COMMIT="1")
    env.pop("X2W_XMEM_TEST_COMMIT_FAILURE", None)
    if inject_failure:
        env["X2W_XMEM_TEST_COMMIT_FAILURE"] = "touch"  # refuse the fault handler's commit, not a mapping one
    child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), path, "child"],
                           env=env, capture_output=True, text=True, timeout=25)
    code = child.returncode & 0xFFFFFFFF
    diagnostic = f"exit {code:#010x}\nstdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    stdout = child.stdout.splitlines()
    targets = [line.removeprefix("touch address=") for line in stdout if line.startswith("touch address=")]
    assert len(targets) == 1, diagnostic
    failures = [line for line in child.stderr.splitlines() if line.startswith(FAILURE_PREFIX)]
    if inject_failure:
        assert code == EXIT_CODE, diagnostic
        assert AFTER_TOUCH not in stdout, diagnostic
        assert len(failures) == 1, diagnostic
        assert failures[0].startswith(f"{FAILURE_PREFIX}{targets[0]} (write)"), diagnostic
        assert " error=1455;" in failures[0], diagnostic
    else:
        assert code == 0 and AFTER_TOUCH in stdout and not failures, diagnostic


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "child":
        touch(sys.argv[1])
    else:
        check(sys.argv[1], inject_failure=False)
        check(sys.argv[1], inject_failure=True)
        print("ok: first touch succeeds normally and stops on commit failure")
