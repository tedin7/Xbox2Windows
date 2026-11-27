#!/usr/bin/env python3
"""Run the stand-in WARP readbacks and check the null-input controls."""

import os
import re
import subprocess
import sys


def main(executable, mode):
    env = dict(os.environ, X2W_TEST_NULL_CENSUS_OMIT_INDEX="1" if mode == "omit-index" else "0")
    env["X2W_DRAW_NULL_ZERO"] = "1" if mode == "empty-zero" else "0"
    if mode in ("empty-zero", "empty-legacy"):
        env["X2W_TEST_EMPTY_DRAW_INPUT"] = "1"
    else:
        env.pop("X2W_TEST_EMPTY_DRAW_INPUT", None)
    result = subprocess.run([executable, "warp", "null-census"], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, file=sys.stderr, end="")
    if mode in ("empty-zero", "empty-legacy"):
        fixture = re.search(r"empty input key ([0-9a-f]{16}) table slot 0 offset (\d+)", result.stdout)
        if not fixture:
            return 1
        rows = [line for line in result.stderr.splitlines() if "x2w-draw zero input:" in line]
        if mode == "empty-legacy":
            return 0 if (result.returncode == 4 and not rows and
                         "empty input readback differs from CPU reference in buffer size" in result.stderr) else 1
        key, offset = fixture.groups()
        expected = f"x2w-draw zero input: key {key} input 2 source table slot 0 offset {offset} declared 28 bound 28"
        return 0 if (result.returncode == 0 and len(rows) == 1 and rows[0].endswith(expected) and
                     "empty input readbacks match CPU zero buffer" in result.stdout) else 1
    if mode == "omit-index":
        return 0 if result.returncode == 3 and "null input index discriminator failed" in result.stderr else 1
    return 0 if result.returncode == 0 and "WARP readbacks, target shares and binding totals match" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
