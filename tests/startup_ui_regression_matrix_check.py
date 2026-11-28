#!/usr/bin/env python3
"""Check tools/startup/ui_regression_matrix.py with a fake CTest executable.

The fixture covers a full mixed matrix, a 77 capability skip, a failure, an
adapter-only selection, a custom filter and empty selections. It runs without
CMake, D3D12 or game files.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "tools" / "startup" / "ui_regression_matrix.py"

FAKE_CTEST = r'''#!/usr/bin/env python3
"""Small CTest stand-in: list tests as JSON and write a CTest-shaped JUnit file."""
import datetime
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def option(args, name):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit("fake ctest: missing %s" % name)


args = sys.argv[1:]
scenario_path = Path(os.environ["UI_MATRIX_SCENARIO"])
record_path = Path(os.environ["UI_MATRIX_INVOCATIONS"])
tests = json.loads(scenario_path.read_text(encoding="utf-8"))["tests"]
exit_code = json.loads(scenario_path.read_text(encoding="utf-8")).get("exit_code")
with record_path.open("a", encoding="utf-8") as record:
    record.write(json.dumps(args) + "\n")

pattern = option(args, "-R") if "-R" in args else ".*"
selected = [test for test in tests if re.search(pattern, test["name"])]

if "--show-only=json-v1" in args:
    print(json.dumps({
        "kind": "ctestInfo",
        "version": {"major": 3, "minor": 31, "patch": 0},
        "tests": [{"name": test["name"]} for test in selected],
    }))
    raise SystemExit(0)

junit = Path(option(args, "--output-junit"))
if not selected:
    print("No tests were found!!!", file=sys.stderr)
    raise SystemExit(8)

failures = sum(test.get("status") == "fail" for test in selected)
skips = sum(test.get("status") == "skip" for test in selected)
duration = sum(float(test.get("duration", 0)) for test in selected)
root = ET.Element("testsuite", {
    "name": "fake-ctest",
    "tests": str(len(selected)),
    "failures": str(failures),
    "errors": "0",
    "skipped": str(skips),
    "time": "%.3f" % duration,
    "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
})
for test in selected:
    status = test.get("status", "pass")
    junit_status = "run" if status == "pass" else "fail" if status == "fail" else "notrun"
    case = ET.SubElement(root, "testcase", {
        "name": test["name"],
        "status": junit_status,
        "time": str(test.get("duration", 0)),
    })
    detail = str(test.get("detail", ""))
    if status in ("skip", "notrun"):
        ET.SubElement(case, "skipped", {"message": detail or "SKIP_RETURN_CODE=77: skipped"})
    elif status == "fail":
        ET.SubElement(case, "failure", {"message": detail or "failure"}).text = detail

ET.ElementTree(root).write(junit, encoding="utf-8", xml_declaration=True)
print("%d tests ran" % len(selected))
raise SystemExit(exit_code if exit_code is not None else (1 if failures else 0))
'''


def test_case(name, status, detail="", duration=0.1):
    return {"name": name, "status": status, "detail": detail, "duration": duration}


FULL_SCENARIO = {
    "tests": [
        test_case("startup_sampler_heap_wrapped_warp", "pass", duration=0.2),
        test_case(
            "startup_sampler_heap_wrapped_hardware",
            "skip",
            "SKIP_RETURN_CODE=77: hardware adapter unavailable",
        ),
        test_case(
            "startup_xg_bc7_draw_warp",
            "fail",
            "pixel (4, 0): read 000000ff, expected ff0000ff",
            duration=1.5,
        ),
        test_case(
            "startup_xg_bc7_draw_hardware",
            "skip",
            "SKIP_RETURN_CODE=77: D3D12 feature level 12 unavailable",
        ),
        test_case("startup_rect_gap_warp", "pass", duration=0.3),
        test_case("startup_rect_gap_hardware", "skip", "SKIP_RETURN_CODE=77: no physical adapter"),
        test_case("startup_ui_texture_bands_warp", "pass", duration=0.4),
        test_case(
            "startup_ui_texture_bands_hardware",
            "skip",
            "SKIP_RETURN_CODE=77: D3D12 debug layer unavailable",
        ),
    ]
}


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.build = self.root / "build"
        self.build.mkdir()
        self.fake = self.root / "fake_ctest.py"
        self.fake.write_text(FAKE_CTEST, encoding="utf-8")
        self.serial = 0

    def run(self, scenario, *extra, expect_outputs=True):
        self.serial += 1
        stem = "matrix-%d" % self.serial
        scenario_path = self.root / (stem + ".scenario.json")
        invocations = self.root / (stem + ".invocations.jsonl")
        output = self.root / (stem + ".md")
        report = self.root / (stem + ".json")
        scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
        env = dict(os.environ, UI_MATRIX_SCENARIO=str(scenario_path),
                   UI_MATRIX_INVOCATIONS=str(invocations))
        command = [
            sys.executable, "-B", str(MATRIX),
            "--build-dir", str(self.build),
            "--ctest", str(self.fake),
            "--json", str(report),
            "--output", str(output),
            *extra,
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=30,
        )
        calls = []
        if invocations.is_file():
            calls = [json.loads(line) for line in invocations.read_text(encoding="utf-8").splitlines()]
        if expect_outputs:
            assert output.is_file(), proc.stderr
            assert report.is_file(), proc.stderr
        return proc, output, report, calls


def rows(markdown, names):
    found = {}
    for name in names:
        matches = [line for line in markdown.splitlines() if name in line]
        assert len(matches) == 1, (name, matches)
        found[name] = matches[0]
    return found


def main():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Fixture(temporary)
        names = [test["name"] for test in FULL_SCENARIO["tests"]]

        proc, output, report, calls = fixture.run(FULL_SCENARIO)
        assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
        assert len(calls) == 2, calls
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 8, "passed": 3, "skipped": 4, "failed": 1}, data
        statuses = {item["name"]: item["status"] for item in data["tests"]}
        assert statuses["startup_sampler_heap_wrapped_hardware"] == "skip", statuses
        assert statuses["startup_xg_bc7_draw_warp"] == "fail", statuses
        skipped = next(item for item in data["tests"]
                       if item["name"] == "startup_sampler_heap_wrapped_hardware")
        assert "77" in skipped["detail"], skipped
        markdown = output.read_text(encoding="utf-8")
        table = rows(markdown, names)
        assert "| PASS |" in table["startup_sampler_heap_wrapped_warp"], table
        assert "| SKIP |" in table["startup_sampler_heap_wrapped_hardware"], table
        assert "| FAIL |" in table["startup_xg_bc7_draw_warp"], table
        assert "not a pass" in markdown, markdown

        proc, output, report, _ = fixture.run(
            FULL_SCENARIO, "--filter", "^startup_rect_gap_hardware$")
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 1, "passed": 0, "skipped": 1, "failed": 0}, data
        assert "| SKIP |" in output.read_text(encoding="utf-8")

        proc, output, report, _ = fixture.run(
            FULL_SCENARIO, "--filter", "^startup_rect_gap_warp$")
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 1, "passed": 1, "skipped": 0, "failed": 0}, data

        proc, output, report, calls = fixture.run(FULL_SCENARIO, "--adapter", "warp")
        assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 4, "passed": 3, "skipped": 0, "failed": 1}, data
        assert all(item["adapter"] == "warp" for item in data["tests"]), data
        selected_run = calls[-1]
        exact = selected_run[selected_run.index("-R") + 1]
        assert "hardware" not in exact, exact

        proc, output, report, _ = fixture.run(
            FULL_SCENARIO, "--filter", "^startup_ui_texture_bands_")
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 2, "passed": 1, "skipped": 1, "failed": 0}, data

        proc, output, report, _ = fixture.run({"tests": []}, expect_outputs=False)
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert "no CTest tests matched" in proc.stderr, proc.stderr
        assert not output.exists() and not report.exists()

        proc, output, report, _ = fixture.run(
            FULL_SCENARIO, "--filter", "^startup_rect_gap_warp$", "--adapter", "hardware",
            expect_outputs=False)
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert "no CTest tests matched" in proc.stderr, proc.stderr

        proc, output, report, calls = fixture.run(
            FULL_SCENARIO, "--filter", "", expect_outputs=False)
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert "must not be empty" in proc.stderr, proc.stderr
        assert not calls

        # A binary that was never built is not a capability skip.
        proc, output, report, _ = fixture.run({
            "tests": [test_case("startup_ui_texture_bands_warp", "notrun", "Unable to find executable")]
        })
        assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"] == {"selected": 1, "passed": 0, "skipped": 0, "failed": 1}, data
        assert data["tests"][0]["status"] == "fail", data
        assert "Unable to find executable" in data["tests"][0]["detail"], data
        assert "| FAIL |" in output.read_text(encoding="utf-8")

        # A nonzero CTest exit must not become exit 0 when every JUnit row says pass.
        proc, output, report, _ = fixture.run({
            "tests": [test_case("startup_ui_texture_bands_warp", "pass")],
            "exit_code": 8,
        })
        assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
        assert "without a failed test result" in proc.stderr, proc.stderr

    print("startup_ui_regression_matrix_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
