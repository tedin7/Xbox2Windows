#!/usr/bin/env python3
"""Checks for tools/gdk_runtime_oracle.py.

Without a GDK, this covers normalization, the unified diff and the skip path
when cl.exe is absent. It does not claim a desktop runtime was measured.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_runtime_oracle as oracle  # noqa: E402


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def check_normalization():
    cases = {
        "queue_handle=0000026D29CDFD60": "queue_handle=<handle>",
        "xpackage_mount_handle=00007FFFEA53C588": "xpackage_mount_handle=<handle>",
        "console_value=58325731.00000000.00000000.00000000.01": "console_value=<console-id>",
        "pls_path_value=C:\\Users\\ROLAND~1\\AppData\\Local\\Temp\\xbox-recomp-pls\\x2w-synthetic-title-0001":
            "pls_path_value=<localappdata>\\Temp\\xbox-recomp-pls\\x2w-synthetic-title-0001",
        "xpackage_path_value=C:\\Users\\ROLAND~1\\AppData\\Local\\Temp\\xbox-recomp-packages\\x2w-nonexistent-package":
            "xpackage_path_value=<localappdata>\\Temp\\xbox-recomp-packages\\x2w-nonexistent-package",
        "title_id=0x80070490": "title_id=0x80070490",
        "queue_dispatch=0": "queue_dispatch=0",
    }
    for line, expected in cases.items():
        actual = oracle.normalize_line(line)
        check(actual == expected, f"{line!r}: {actual!r} != {expected!r}")
    print(f"normalization: {len(cases)} cases passed")


def check_diff():
    reference = ["init=0x00000000", "device=1", "queue_dispatch=0"]
    candidate = ["init=0x00000000", "device=1", "queue_dispatch=1"]
    difference = oracle.diff_lines(reference, candidate)
    check(difference[0] == "--- reference", f"header {difference[0]!r}")
    check(difference[1] == "+++ candidate", f"header {difference[1]!r}")
    check("-queue_dispatch=0" in difference, f"removed line {difference}")
    check("+queue_dispatch=1" in difference, f"added line {difference}")
    check(oracle.diff_lines(reference, reference) == [], "identical logs differ")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "reference.txt").write_text("\n".join(reference) + "\n", encoding="utf-8")
        (directory / "candidate.txt").write_text("\n".join(candidate) + "\n", encoding="utf-8")
        code = oracle.main(["diff", str(directory / "reference.txt"), str(directory / "candidate.txt")])
        check(code == 1, f"diff exit {code}")
        (directory / "candidate.txt").write_text("\n".join(reference) + "\n", encoding="utf-8")
        code = oracle.main(["diff", str(directory / "reference.txt"), str(directory / "candidate.txt")])
        check(code == 0, f"identical exit {code}")
    print("diff: headers, changed line and exit codes passed")


def check_skip():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gdk = root / "grdk"
        (gdk / "GameKit" / "Include").mkdir(parents=True)
        (gdk / "GameKit" / "Lib" / "amd64").mkdir(parents=True)
        shim = root / "shim"
        shim.mkdir()
        fixture = root / "fixture.cpp"
        fixture.write_text("int main() { return 0; }\n", encoding="utf-8")
        code = oracle.main([
            "run",
            "--gdk-root", str(gdk),
            "--shim-dir", str(shim),
            "--fixture", str(fixture),
            "--out", str(root / "out"),
            "--cl", str(root / "missing-cl.exe"),
        ])
        check(code == oracle.SKIP, f"missing cl exit {code}")
    print("run: missing cl reports 77")


def check_version_parser():
    with tempfile.TemporaryDirectory() as directory:
        include = Path(directory)
        (include / "GRDK.h").write_text(
            "#define _GRDKVER_PRODUCTBUILDVER_FULL 10.0.26100.9661\n", encoding="utf-8"
        )
        check(oracle.grdk_version(include) == "10.0.26100.9661", "GRDK version")
        check(oracle.grdk_version(Path(directory) / "missing") == "unknown", "missing GRDK version")
    print("version parser: present and missing headers passed")


def main():
    check_normalization()
    check_diff()
    check_skip()
    check_version_parser()
    print("gdk_runtime_oracle_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
