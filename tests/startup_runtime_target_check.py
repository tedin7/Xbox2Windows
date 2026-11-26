#!/usr/bin/env python3
"""Check the `startup_runtime` target (`cmake --build --preset runtime`): the eight DLLs tools/startup/run.ps1 copies
and no executable.

usage: startup_runtime_target_check.py <cmake> <build dir> <run.ps1> <dll>...

1. `cmake --build <build dir> --target startup_runtime` succeeds and leaves every DLL on disk.
2. The DLL names are exactly the ones run.ps1 copies next to the title.
3. Ninja's recursive inputs of the target contain no executable: the target does not pull in a test program.
"""
import os
import re
import subprocess
import sys


def main(argv):
    cmake, build, run_ps1, dlls = argv[1], argv[2], argv[3], argv[4:]
    built = subprocess.run([cmake, "--build", build, "--target", "startup_runtime"], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]
    for dll in dlls:
        assert os.path.isfile(dll), "startup_runtime did not leave " + dll
    with open(run_ps1, encoding="utf-8") as handle:
        copied = re.search(r"foreach \(\$dll in (.*?)\) \{", handle.read()).group(1)
    wanted = sorted(name.lower() for name in re.findall(r'"([A-Za-z0-9_]+)"', copied))
    got = sorted(os.path.splitext(os.path.basename(dll))[0].lower() for dll in dlls)
    assert got == wanted, "run.ps1 copies %s, the target builds %s" % (wanted, got)
    inputs = subprocess.run(["ninja", "-C", build, "-t", "inputs", "startup_runtime"], capture_output=True, text=True)
    assert inputs.returncode == 0, inputs.stdout[-2000:] + inputs.stderr[-2000:]
    executables = [line for line in inputs.stdout.splitlines() if line.strip().lower().endswith(".exe")]
    assert not executables, "startup_runtime depends on executables: %s" % executables[:5]
    assert inputs.stdout.strip(), "ninja listed no inputs for startup_runtime"
    print("startup_runtime builds the %d DLLs run.ps1 copies and no executable" % len(dlls))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
