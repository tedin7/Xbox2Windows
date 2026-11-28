"""Draw and validate the WARP occlusion-query fixture with DRED on and off."""
import os
import re
import subprocess
import sys


DRED_LINE = re.compile(r"^x2w-dred:", re.MULTILINE)
DRED_ON = re.compile(r"^x2w-dred: auto-breadcrumbs and page faults on$", re.MULTILINE)


def main(driver):
    env = dict(os.environ, X2W_DRED="1")
    for enabled in (True, False):
        if not enabled:
            env.pop("X2W_DRED")
        result = subprocess.run([driver, "warp"], env=env, capture_output=True, text=True,
                                errors="replace", timeout=40)
        assert result.returncode == 0, (result.stdout[-1500:], result.stderr[-1500:])
        assert len(DRED_LINE.findall(result.stderr)) == int(enabled), result.stderr[-3000:]
        if enabled:
            assert DRED_ON.search(result.stderr), result.stderr[-3000:]


if __name__ == "__main__":
    main(*sys.argv[1:])
