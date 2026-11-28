#!/usr/bin/env python3
"""Run the startup UI CTest fixtures and write a Markdown/JSON regression matrix.

The default selection covers the sampler/BC7 fixes from #3153, the XG mip-0 and
CopyTextureRegion paths from #3141, and the indexed rectangle fixture from #3142.
Only tests registered in the supplied build are run. A test process that exits
with 77 is reported by CTest as skipped when its CTest property says so; this
tool keeps that state separate from both pass and fail.
"""

from __future__ import annotations

import argparse
import datetime as datetime_module
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


DEFAULT_FILTER = (
    r"^(startup_sampler_heap_wrapped|startup_xg_bc7_draw|startup_rect_gap|"
    r"startup_ui_texture_bands)_(warp|hardware)$"
)
ADAPTERS = ("warp", "hardware")
PASS = "pass"
SKIP = "skip"
FAIL = "fail"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3


class MatrixError(Exception):
    """A setup or selection error that prevents a trustworthy matrix."""


def _ctest_command(value: str) -> list[str]:
    """Resolve --ctest to a command; a .py fixture runs under this interpreter."""
    value = value.strip()
    if not value:
        raise MatrixError("--ctest must not be empty")
    candidate = Path(value).expanduser()
    if candidate.is_file():
        path = candidate.resolve()
        if path.suffix.lower() == ".py":
            return [sys.executable, "-B", str(path)]
        return [str(path)]
    found = shutil.which(value)
    if found:
        return [found]
    raise MatrixError(f"CTest executable not found: {value}")


def _run(command: list[str], build_dir: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise MatrixError(f"cannot run {command[0]!r}: {error}") from error


def _diagnostic(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    if not text:
        return f"exit code {proc.returncode}"
    tail = text.splitlines()[-8:]
    return "\n".join(tail)[:2000]


def selected_test_names(ctest: list[str], build_dir: Path, pattern: str) -> list[str]:
    """Return the CTest names matching pattern, in CTest's reported order."""
    proc = _run([*ctest, "--show-only=json-v1", "-R", pattern], build_dir)
    if proc.returncode != 0:
        raise MatrixError(f"CTest could not list tests (exit {proc.returncode}):\n{_diagnostic(proc)}")
    text = proc.stdout.lstrip("\ufeff \r\n\t")
    start = text.find("{")
    if start < 0:
        raise MatrixError("CTest --show-only=json-v1 did not return JSON:\n" + text[-2000:])
    try:
        listing = json.loads(text[start:])
    except json.JSONDecodeError as error:
        raise MatrixError(f"invalid CTest JSON listing: {error}") from error
    tests = listing.get("tests") if isinstance(listing, dict) else None
    if not isinstance(tests, list):
        raise MatrixError("CTest JSON listing has no tests array")
    names = []
    seen = set()
    for test in tests:
        name = test.get("name") if isinstance(test, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise MatrixError("CTest JSON listing contains a test without a name")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def adapter_of(name: str) -> str | None:
    for adapter in ADAPTERS:
        if name.endswith("_" + adapter):
            return adapter
    return None


def _exact_filter(names: list[str]) -> str:
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def _node_detail(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        node = case.find(tag)
        if node is None:
            continue
        pieces = []
        message = (node.get("message") or "").strip()
        if message:
            pieces.append(message)
        text = (node.text or "").strip()
        if text and text != message:
            pieces.append(text)
        return "\n".join(pieces)[:4000]
    return ""


def _duration(case: ET.Element) -> float | None:
    try:
        value = float(case.get("time", ""))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _case_status(case: ET.Element) -> tuple[str, str]:
    """Classify a CTest JUnit case without treating an unknown state as a pass."""
    status = case.get("status", "")
    if case.find("failure") is not None or case.find("error") is not None:
        return FAIL, _node_detail(case) or "CTest reported a failure"
    detail = _node_detail(case)
    if status == "notrun":
        # CTest also uses notrun for a missing executable, a missing configuration or a
        # disabled test. Only the explicit capability markers are skips; the rest failed
        # before the fixture could measure anything.
        if detail.startswith(("SKIP_RETURN_CODE", "SKIP_REGULAR_EXPRESSION")):
            return SKIP, detail
        return FAIL, detail or "CTest marked the test not run"
    if case.find("skipped") is not None:
        return SKIP, detail or "CTest reported a skip"
    if status == "run":
        return PASS, ""
    return FAIL, f"unrecognized CTest result status {status!r}"


def read_junit(junit: Path) -> list[dict[str, object]]:
    try:
        root = ET.parse(junit).getroot()
    except (ET.ParseError, OSError) as error:
        raise MatrixError(f"cannot read CTest JUnit report {junit}: {error}") from error
    if root.tag != "testsuite":
        raise MatrixError(f"expected a single CTest <testsuite> root, found <{root.tag}>")
    results = []
    for case in root.iter("testcase"):
        name = (case.get("name") or "").strip()
        if not name:
            raise MatrixError("CTest JUnit contains a testcase without a name")
        status, detail = _case_status(case)
        results.append({
            "name": name,
            "adapter": adapter_of(name),
            "status": status,
            "duration_seconds": _duration(case),
            "detail": detail,
        })
    return results


def run_tests(
    ctest: list[str], build_dir: Path, names: list[str]
) -> tuple[list[dict[str, object]], int, list[str]]:
    """Run exactly names and return results, CTest exit status and its command."""
    exact = _exact_filter(names)
    with tempfile.TemporaryDirectory(prefix="x2w-ui-matrix-") as temporary:
        junit = Path(temporary) / "ctest-junit.xml"
        command = [
            *ctest,
            "--output-junit",
            str(junit),
            "--output-on-failure",
            "--no-tests=error",
            "-R",
            exact,
        ]
        proc = _run(command, build_dir)
        if not junit.is_file():
            if "No tests were found" in (proc.stdout or "") + (proc.stderr or ""):
                raise MatrixError("CTest selected no tests")
            raise MatrixError(
                f"CTest did not write a JUnit report (exit {proc.returncode}):\n{_diagnostic(proc)}"
            )
        results = read_junit(junit)
    by_name = {str(item["name"]): item for item in results}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise MatrixError("CTest did not report: " + ", ".join(missing))
    if not results:
        raise MatrixError("CTest selected no tests")
    # Keep exactly the selected order; a concurrent registration cannot inject a row.
    ordered = [by_name[name] for name in names]
    return ordered, proc.returncode, command


def _source_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def summarize(results: list[dict[str, object]]) -> dict[str, int]:
    return {
        "selected": len(results),
        "passed": sum(item["status"] == PASS for item in results),
        "skipped": sum(item["status"] == SKIP for item in results),
        "failed": sum(item["status"] == FAIL for item in results),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _md_code(value: str) -> str:
    return "`" + value.replace("`", "'") + "`"


def _short_detail(value: object, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def markdown_report(
    build_dir: Path,
    selection: str,
    adapter: str,
    ctest_returncode: int,
    results: list[dict[str, object]],
    summary: dict[str, int],
) -> str:
    lines = [
        "# Startup UI regression matrix",
        "",
        f"- Build directory: {_md_code(_display_path(build_dir))}",
        f"- Selection: {_md_code(selection)}",
        f"- Adapter: {_md_code(adapter)}",
        f"- CTest exit code: {ctest_returncode}",
        f"- Result: {summary['passed']} passed, {summary['skipped']} skipped, "
        f"{summary['failed']} failed ({summary['selected']} selected)",
        "",
        "| Test | Result | Time | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for item in results:
        status = str(item["status"]).upper()
        duration = item["duration_seconds"]
        when = f"{duration:.3f} s" if isinstance(duration, float) else "-"
        lines.append(
            f"| {_md_code(str(item['name']))} | {status} | {when} | "
            f"{_short_detail(item['detail'])} |"
        )
    lines += [
        "",
        "A skipped fixture did not complete its measurement. It is not a pass; "
        "its capability or environment requirement has to be satisfied before "
        "that row can support a result.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected startup UI CTest fixtures and write pass/skip/fail "
            "Markdown and JSON matrices. Empty selections are errors."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("out/build/dev"),
        help="CMake build directory (default: out/build/dev)",
    )
    parser.add_argument(
        "--ctest",
        default="ctest",
        help="CTest executable, or a Python fixture script (default: ctest)",
    )
    parser.add_argument(
        "--adapter",
        choices=("all",) + ADAPTERS,
        default="all",
        help="restrict rows to tests ending _warp or _hardware (default: all)",
    )
    parser.add_argument("--json", type=Path, help="write the matrix as JSON")
    parser.add_argument("--output", type=Path, help="write the matrix as Markdown")
    parser.add_argument(
        "--filter",
        dest="filter_option",
        help="CTest name regex; default is the sampler/BC7/rect/bands matrix",
    )
    parser.add_argument(
        "positional_filter",
        nargs="?",
        metavar="FILTER",
        help="CTest name regex (alternative to --filter)",
    )
    return parser


def _write(path: Path, content: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.filter_option is not None and args.positional_filter is not None:
        parser.error("give the filter either positionally or with --filter, not both")
    selection = args.filter_option if args.filter_option is not None else args.positional_filter
    if selection is None:
        selection = DEFAULT_FILTER
    if not selection.strip():
        parser.error("the filter must not be empty")

    build_dir = args.build_dir.expanduser().resolve()
    if not build_dir.is_dir():
        parser.error(f"build directory does not exist: {build_dir}")
    try:
        ctest = _ctest_command(args.ctest)
    except MatrixError as error:
        parser.error(str(error))

    try:
        names = selected_test_names(ctest, build_dir, selection)
        if args.adapter != "all":
            names = [name for name in names if adapter_of(name) == args.adapter]
        if not names:
            raise MatrixError(
                f"no CTest tests matched the selection for adapter {args.adapter!r}: {selection}"
            )
        results, ctest_returncode, command = run_tests(ctest, build_dir, names)
    except MatrixError as error:
        print(f"ui_regression_matrix: error: {error}", file=sys.stderr)
        return EXIT_USAGE

    summary = summarize(results)
    unexplained_failure = ctest_returncode != 0 and summary["failed"] == 0
    markdown = markdown_report(build_dir, selection, args.adapter, ctest_returncode, results, summary)
    payload = {
        "schema": "xbox2windows-startup-ui-regression-matrix-v1",
        "generated_at": datetime_module.datetime.now(
            datetime_module.timezone.utc
        ).isoformat(timespec="seconds"),
        "source_commit": _source_commit(),
        "build_dir": _display_path(build_dir),
        "ctest": args.ctest,
        "ctest_command": command,
        "ctest_returncode": ctest_returncode,
        "selection": selection,
        "adapter": args.adapter,
        "tests": results,
        "summary": summary,
    }

    if args.output:
        _write(args.output, markdown)
        print(f"ui_regression_matrix: wrote {args.output}")
    else:
        print(markdown)
    if args.json:
        _write(args.json, json.dumps(payload, indent=2) + "\n")
        print(f"ui_regression_matrix: wrote {args.json}")
    print(
        "ui_regression_matrix: "
        f"{summary['passed']} passed, {summary['skipped']} skipped, "
        f"{summary['failed']} failed ({summary['selected']} selected)"
    )

    if summary["failed"] or unexplained_failure:
        if unexplained_failure:
            print(
                "ui_regression_matrix: CTest exited with "
                f"{ctest_returncode} without a failed test result",
                file=sys.stderr,
            )
        return EXIT_FAILED
    if summary["skipped"]:
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
