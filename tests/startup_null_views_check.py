#!/usr/bin/env python3
"""Run the stand-in WARP readbacks and check the input-index negative control."""

import os
import subprocess
import sys


def main(executable, mode):
    env = dict(os.environ, X2W_TEST_NULL_CENSUS_OMIT_INDEX="1" if mode == "omit-index" else "0")
    result = subprocess.run([executable, "warp", "null-census"], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, file=sys.stderr, end="")
    if mode == "omit-index":
        return 0 if result.returncode == 3 and "null input index discriminator failed" in result.stderr else 1
    return 0 if result.returncode == 0 and "WARP readbacks, target shares and binding totals match" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
