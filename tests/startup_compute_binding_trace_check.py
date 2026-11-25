"""Check the opt-in slot 62 binding diagnostic on a source-built command list."""
from pathlib import Path
import re
import subprocess
import sys


def check(executable, mode):
    args = [str(Path(executable).resolve())]
    if mode:
        args.append(mode)
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
    key = re.search(r"^key=([0-9a-f]{16})$", result.stdout, re.MULTILINE)[1]
    lines = [line for line in result.stderr.splitlines() if line.startswith("x2w-compute-bindings ")]
    if mode:
        assert not lines, (mode, lines)
        return
    prefix = f"x2w-compute-bindings key={key} Dispatch="
    destination = "p1=4:r0:s0:n0/set45/placed-copy/low4/mask0000000000000000"
    constants = "p2=1:r0:s0:n16/set0/none/low0/mask0000000000000032"
    empty = "set0/none/low0/mask0000000000000000"
    assert lines and re.fullmatch(rf"x2w-compute-bindings key={key} Create slot=62 bytes=[1-9][0-9]+", lines[0]), lines
    assert lines[1:] == [
        prefix + "2x1x1 root=match desktop-root=1 desktop-pso=0 params=3 "
        f"p0=3:r0:s0:n0/set43/upload/low0/mask0000000000000000 {destination} {constants}",
        prefix + "3x1x1 root=match desktop-root=1 desktop-pso=0 params=3 "
        f"p0=3:r0:s0:n0/set43/upload-unaligned/low4/mask0000000000000000 {destination} {constants}",
        prefix + "4x1x1 root=different desktop-root=1 desktop-pso=0 params=3 "
        f"p0=3:r0:s0:n0/{empty} p1=4:r0:s0:n0/{empty} p2=1:r0:s1:n16/{empty}",
        prefix + "5x1x1 root=absent desktop-root=0 desktop-pso=0 params=0",
    ], lines


for setting in (None, "off", "unselected"):
    check(sys.argv[1], setting)
print("slot 62 binding trace, opt-out, and filter controls passed")
