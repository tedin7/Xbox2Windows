#!/usr/bin/env python3
"""Compare one GDK runtime fixture against the GRDK and the Xbox2Windows shim.

The reference executable links the installed gaming runtime and is run with the
inbox xgameruntime.dll. The candidate links the shim's import library and runs
with that library next to it. Both print one observation per line. Only the
normalized lines are compared: handles, console ids and per-user temp paths are
masked so two runs can be diffed.

Commands:
  run --gdk-root <GRDK> --shim-dir <dir> --fixture <cpp> --out <dir>
  normalize <log>
  diff <reference.txt> <candidate.txt>

run returns 77 when cl.exe or a required path is missing, so CI can skip a host
without the GDK. The normalization and diff commands need no SDK.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = 77

HANDLE_RE = re.compile(
    r"\b(queue_handle|xpackage_mount_handle)=(?:0x)?[0-9A-Fa-f]+\b"
)
CONSOLE_RE = re.compile(r"\b(console_value=)[0-9A-Fa-f.]+")
PATH_RE = re.compile(
    r"\b(pls_path_value=|xpackage_path_value=)"
    r"(?:[A-Za-z]:\\[^\r\n]*?)\\Temp\\(xbox-recomp-(?:pls|packages))\\([^\\\s]+)"
)


def normalize_line(line):
    """Mask volatile handles, console ids and per-user temp paths."""
    line = HANDLE_RE.sub(lambda m: m.group(1) + "=<handle>", line)
    line = CONSOLE_RE.sub(lambda m: m.group(1) + "<console-id>", line)
    line = PATH_RE.sub(
        lambda m: m.group(1) + "<localappdata>\\Temp\\" + m.group(2) + "\\" + m.group(3),
        line,
    )
    return line


def normalize(lines):
    return [normalize_line(str(line).rstrip("\r\n")) for line in lines]


def diff_lines(reference, candidate):
    """Return a unified diff of two normalized logs, without file timestamps."""
    return list(
        difflib.unified_diff(
            list(reference),
            list(candidate),
            fromfile="reference",
            tofile="candidate",
            lineterm="",
            n=2,
        )
    )


def read_lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def grdk_version(include_dir):
    header = Path(include_dir) / "GRDK.h"
    if not header.exists():
        return "unknown"
    match = re.search(
        r"#define _GRDKVER_PRODUCTBUILDVER_FULL\s+([0-9.]+)",
        header.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else "unknown"


def file_version(path):
    path = Path(path)
    if not path.exists():
        return "missing"
    if sys.platform != "win32":
        return "unavailable"
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-Item -LiteralPath '" + str(path).replace("'", "''") + "').VersionInfo.FileVersion",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"


def build_command(cl, fixture, include, output, lib, shim_dir=None):
    command = [str(cl), "/nologo", "/std:c++17", "/EHsc", "/W4"]
    if shim_dir is not None:
        command.append("/DX2W_SHIM")
    command += ["/I" + str(include), str(fixture), "/link"]
    command.append("/LIBPATH:" + str(shim_dir if shim_dir is not None else lib))
    command += ["advapi32.lib", "/OUT:" + str(output)]
    return command


def run_fixture(executable, cwd, timeout):
    result = subprocess.run(
        [str(executable)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = (result.stdout + result.stderr).splitlines()
    return result.returncode, lines


def msvc_environment(vcvars):
    """Return os.environ plus the variables vcvars64.bat sets, or (None, log)."""
    command = 'call "' + str(vcvars) + '" >nul && set'
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return None, result.stdout + result.stderr
    environment = os.environ.copy()
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            for existing in [name for name in environment if name.lower() == key.lower()]:
                del environment[existing]
            environment[key] = value
    return environment, ""


def run_build(command, environment):
    return subprocess.run(
        command, capture_output=True, text=True, timeout=120, env=environment
    )


def command_run(args):
    gdk_root = Path(args.gdk_root)
    include = gdk_root / "GameKit" / "Include"
    lib = gdk_root / "GameKit" / "Lib" / "amd64"
    shim_dir = Path(args.shim_dir)
    fixture = Path(args.fixture)
    for path in (include, lib, shim_dir, fixture):
        if not path.exists():
            print(f"missing required path: {path}", file=sys.stderr)
            return 2

    environment = None
    if args.vcvars:
        environment, error = msvc_environment(args.vcvars)
        if environment is None:
            print("vcvars failed: " + error, file=sys.stderr)
            return 2
        path = next((value for key, value in environment.items() if key.lower() == "path"), "")
        cl = args.cl or shutil.which("cl", path=path)
        if cl is None:
            print("cl.exe not found in the vcvars environment", file=sys.stderr)
            return 2
    else:
        cl = args.cl or shutil.which("cl")
        if cl is None or not Path(cl).exists():
            print("cl.exe not found; skipping the GDK runtime oracle", file=sys.stderr)
            return SKIP

    out = Path(args.out)
    reference_dir = out / "reference"
    candidate_dir = out / "candidate"
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    reference_exe = reference_dir / "fixture.exe"
    candidate_exe = candidate_dir / "fixture.exe"
    builds = (
        ("reference", build_command(cl, fixture, include, reference_exe, lib)),
        ("candidate", build_command(cl, fixture, include, candidate_exe, lib, shim_dir)),
    )
    for name, command in builds:
        result = run_build(command, environment)
        (out / f"build-{name}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            print(f"{name} fixture build failed; see {out / f'build-{name}.log'}", file=sys.stderr)
            return 2

    runtime_dll = shim_dir / "xgameruntime.dll"
    if runtime_dll.exists():
        shutil.copy2(runtime_dll, candidate_dir / runtime_dll.name)

    reference_exit, reference_lines = run_fixture(reference_exe, reference_dir, args.timeout)
    candidate_exit, candidate_lines = run_fixture(candidate_exe, candidate_dir, args.timeout)
    reference_normalized = normalize(reference_lines)
    candidate_normalized = normalize(candidate_lines)
    differences = diff_lines(reference_normalized, candidate_normalized)

    (out / "reference.txt").write_text("\n".join(reference_lines) + "\n", encoding="utf-8")
    (out / "candidate.txt").write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")
    (out / "diff.txt").write_text("\n".join(differences) + "\n", encoding="utf-8")
    report = {
        "sdk": grdk_version(include),
        "fixture": str(fixture),
        "reference": {
            "exit": reference_exit,
            "runtime": file_version(Path("C:/Windows/System32/xgameruntime.dll")),
            "normalized": reference_normalized,
        },
        "candidate": {
            "exit": candidate_exit,
            "shim": str(shim_dir),
            "normalized": candidate_normalized,
        },
        "differences": differences,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"reference exit={reference_exit}; candidate exit={candidate_exit}")
    print("\n".join(differences) if differences else "no normalized differences")
    if args.fail_on_difference and differences:
        return 1
    return 0


def command_normalize(args):
    lines = normalize(read_lines(args.log))
    print("\n".join(lines))
    return 0


def command_diff(args):
    reference = normalize(read_lines(args.reference))
    candidate = normalize(read_lines(args.candidate))
    differences = diff_lines(reference, candidate)
    print("\n".join(differences) if differences else "no normalized differences")
    return 1 if differences else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build and run the reference and shim fixtures")
    run.add_argument("--gdk-root", required=True, help="GRDK root (contains GameKit)")
    run.add_argument("--shim-dir", required=True, help="directory with the shim import library")
    run.add_argument("--fixture", required=True, help="C++ fixture source")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--cl", help="explicit cl.exe path")
    run.add_argument("--vcvars", help="Visual Studio vcvars64.bat; runs cl in its environment")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--fail-on-difference", action="store_true")
    run.set_defaults(func=command_run)

    normalize_command = sub.add_parser("normalize", help="print a normalized log")
    normalize_command.add_argument("log")
    normalize_command.set_defaults(func=command_normalize)

    diff = sub.add_parser("diff", help="compare two normalized logs")
    diff.add_argument("reference")
    diff.add_argument("candidate")
    diff.set_defaults(func=command_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
