"""Run independent Python check scripts side by side and fail if any of them fails.

Usage:
    python tools/run_checks_parallel.py [-j N] [--from-dir DIR] tests/a_check.py tests/b_check.py ...

`--from-dir DIR` also runs the commands listed in `DIR/*.txt` (see `read_check_files`), after
the scripts given on the command line. `merge_checks(inline, directory)` does the same for
the pairs passed to `run_checks`.

Each script runs as `python -B <script>` from the current directory. Output is printed one
script at a time, in the order given, so a failure log stays readable. The exit status is 1
when any script exits non-zero. Scripts that exit 77 (skip) are listed under `SKIPPED:`,
apart from the `FAILED:` line; a skip still makes the exit status 1. A script named in
`tests/checks/needs_arguments.txt` and run with no arguments is not started: it is reported as
a skip, because only CTest (or a listed command with arguments) can run it.

`run_checks(checks, jobs=None)` runs the same runner directly over check pairs. Each pair is
`(command, extra_env)`: `command` is the argument list placed after `python -B` and
`extra_env` (empty or None to inherit) is merged over the current environment for that
check only.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SKIP_EXIT_CODE = 77
NEEDS_ARGUMENTS_FILE = Path(__file__).resolve().parent.parent / "tests" / "checks" / "needs_arguments.txt"


def read_needs_arguments(path=NEEDS_ARGUMENTS_FILE):
    """Return the script file names listed in `tests/checks/needs_arguments.txt` (empty if absent)."""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return frozenset()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def _validate_script_path(raw_path: str) -> str:
    if not raw_path or not raw_path.strip():
        raise ValueError("script path must not be empty")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"script path does not exist: {raw_path}")
    if path.is_dir():
        raise IsADirectoryError(f"script path is a directory: {raw_path}")
    if not path.is_file():
        raise ValueError(f"script path is not a regular file: {raw_path}")
    return raw_path


def _split_windows(line):
    """Split on whitespace; quotes group text inside a word and are dropped, backslashes stay."""
    words, current, quote, started = [], [], None, False
    for char in line:
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in "\"'":
            quote, started = char, True
        elif char.isspace():
            if started:
                words.append("".join(current))
                current, started = [], False
        else:
            current.append(char)
            started = True
    if quote:
        raise ValueError("No closing quotation")
    if started:
        words.append("".join(current))
    return words


def split_command(line, windows=None):
    """Split a listed line like a shell would, but keep backslashes in Windows paths.

    POSIX splitting treats a backslash as an escape and would turn `C:\\x2w\\a.py` into
    `C:x2wa.py`. On Windows the line is split without escapes; quotes group words, also inside
    one (`--llvm="C:\\Program Files\\a.dll"`), and are removed.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return shlex.split(line)
    return _split_windows(line)


def read_check_files(directory):
    """Return the check commands listed in `directory/*.txt`, files in name order.

    One command per line: a script path, optionally followed by its arguments (split like a
    shell would). Blank lines and lines starting with `#` are ignored. A leading UTF-8
    byte-order mark is skipped. A missing directory, an
    unreadable file, a script that does not exist, or the same command listed twice all raise,
    so a typo cannot silently drop a check.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"check list directory does not exist: {directory}")
    commands = []
    seen = {}
    for listing in sorted(root.glob("*.txt")):
        for number, line in enumerate(listing.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{listing}:{number}"
            try:
                command = split_command(line)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
            try:
                _validate_script_path(command[0])
            except (ValueError, OSError) as exc:
                raise type(exc)(f"{where}: {exc}") from exc
            key = tuple(command)
            if key in seen:
                raise ValueError(f"{where}: duplicate of {seen[key]}: {line}")
            seen[key] = where
            commands.append(command)
    return commands


def merge_checks(inline, directory):
    """Return the inline `(command, extra_env)` pairs followed by the listed checks.

    Listed commands already present inline are dropped, so an entry can move from the inline
    list to a file without running twice. The listed commands keep their sorted file order.
    """
    inline = list(inline)
    known = {tuple(command) for command, _ in inline}
    return inline + [(command, {}) for command in read_check_files(directory)
                     if tuple(command) not in known]


def run_one(check):
    command, extra_env = check
    if not command:
        raise ValueError("check command must not be empty")
    env = {**os.environ, **extra_env} if extra_env else None
    start = time.monotonic()
    if len(command) == 1 and Path(command[0]).name in read_needs_arguments():
        msg = (f"SKIP {command[0]} needs arguments (binary, compiler, renderer or backend); "
               "CTest passes them.\n")
        return command[0], SKIP_EXIT_CODE, msg, 0.0
    try:
        proc = subprocess.run([sys.executable, "-B", *command], env=env,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        msg = f"{type(exc).__name__}: {detail}\n" if detail else f"{type(exc).__name__}\n"
        return command[0], 1, msg, time.monotonic() - start
    return command[0], proc.returncode, proc.stdout, time.monotonic() - start


def run_checks(checks, jobs=None):
    checks = list(checks)
    failed = []
    skipped = []
    max_workers = max(1, os.cpu_count() or 2) if jobs is None else max(1, jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for script, code, output, secs in pool.map(run_one, checks):
            print(f"::group::{script} ({'ok' if code == 0 else f'exit {code}'}, {secs:.1f}s)", flush=True)
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                print()
            print("::endgroup::", flush=True)
            if code == SKIP_EXIT_CODE:
                skipped.append(script)
            elif code != 0:
                failed.append(script)
    if failed:
        print("FAILED: " + " ".join(failed))
    if skipped:
        print("SKIPPED: " + " ".join(skipped))
    if failed or skipped:
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--from-dir", metavar="DIR",
                        help="also run the checks listed in DIR/*.txt")
    parser.add_argument("scripts", nargs="*")
    args = parser.parse_args(argv)
    if not args.scripts and not args.from_dir:
        parser.error("the following arguments are required: scripts (or --from-dir)")
    try:
        scripts = [_validate_script_path(script) for script in args.scripts]
        for number, script in enumerate(scripts):
            if script in scripts[:number]:
                raise ValueError(f"duplicate command-line script: {script}")
        checks = [([script], None) for script in scripts]
        if args.from_dir:
            checks = merge_checks(checks, args.from_dir)
        return run_checks(checks, jobs=args.jobs)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        msg = f"{detail}\n" if detail else f"{type(exc).__name__}\n"
        parser.exit(1, msg)


if __name__ == "__main__":
    sys.exit(main())
