#!/usr/bin/env python3
"""Check tools/coverage_svg.py: treemap layout, square fitting, group boxes, SVG rendering,
and CLI error/collision/atomic-write handling without tracebacks or temp-file leaks."""
from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "coverage_svg.py"
sys.path.insert(0, str(ROOT / "tools"))
import coverage_svg as cs


def _sample_coverage():
    return {
        "x64_zen2": {
            "covered": 4,
            "total": 1007,
            "blocked_by_skip": ["SSE movss"],
            "groups": {
                "Integer": {"covered": 3, "total": 1000},
                "SSE": {"covered": 1, "total": 6},
                "Tiny": {"covered": 0, "total": 1},
                "Zero": {"covered": 0, "total": 0},
            },
        },
        "gfx1013": {
            "covered": 0,
            "total": 0,
            "blocked_by_skip": [],
            "groups": {
                "ZeroOnly": {"covered": 0, "total": 0},
            },
        },
        "gfx1030": {
            "covered": 2,
            "total": 5,
            "blocked_by_skip": ["VOP3 v_add3_u32"],
            "groups": {
                "SOP1": {"covered": 2, "total": 3},
                "VOP3": {"covered": 0, "total": 2},
            },
        },
    }


def _expect_exit(argv, expected_code=1):
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            cs.main(argv)
    except SystemExit as exc:
        assert exc.code == expected_code, (exc.code, stderr.getvalue())
    else:
        raise AssertionError(f"main({argv!r}) did not raise SystemExit")
    err = stderr.getvalue()
    assert err.strip(), f"expected error message on stderr for {argv!r}"
    assert "Traceback" not in err, err
    return err


def check_squarify():
    """Positive and negative treemap layout controls."""
    for w, h in ((940.0, 220.0), (220.0, 940.0)):
        items = [("a", 6), ("b", 3), ("c", 1)]
        boxes = cs.squarify(items, 30.0, 120.0, w, h)
        assert set(boxes) == {"a", "b", "c"}, boxes
        total_area = sum(bw * bh for _, _, bw, bh in boxes.values())
        assert math.isclose(total_area, w * h, rel_tol=1e-6), (total_area, w * h)
        for key, weight in items:
            bx, by, bw, bh = boxes[key]
            assert bw > 0 and bh > 0, (key, boxes[key])
            assert 30.0 - 1e-6 <= bx and bx + bw <= 30.0 + w + 1e-6, (key, boxes[key])
            assert 120.0 - 1e-6 <= by and by + bh <= 120.0 + h + 1e-6, (key, boxes[key])
            assert math.isclose(bw * bh, w * h * weight / 10.0, rel_tol=1e-6), (key, boxes[key])

    # Empty items, zero/non-positive total weight, and zero/non-positive dimensions guard ZeroDivisionError.
    assert cs.squarify([], 0, 0, 100, 100) == {}
    assert cs.squarify([("a", 0)], 0, 0, 100, 100) == {}
    assert cs.squarify([("a", -5)], 0, 0, 100, 100) == {}
    assert cs.squarify([("a", 5), ("b", -5)], 0, 0, 100, 100) == {}
    assert cs.squarify([("a", 5), ("b", -10)], 0, 0, 100, 100) == {}
    assert cs.squarify([("a", 5)], 0, 0, 0, 100) == {}
    assert cs.squarify([("a", 5)], 0, 0, 100, -1) == {}
    mixed = cs.squarify([("a", 10), ("b", 0)], 10, 20, 100, 50)
    assert set(mixed) == {"a"}, mixed
    assert mixed["a"] == (10, 20, 100.0, 50.0), mixed


def check_fit():
    """Square fitting positive and non-positive controls."""
    assert cs.fit(1, 10, 10) == (10, 1)
    assert cs.fit(4, 5, 5) == (2, 2)
    assert cs.fit(100, 10, 10) is None
    for args in (
        (0, 100, 100),
        (-3, 100, 100),
        (4, 0, 100),
        (4, 100, 0),
        (4, -10, 50),
        (4, 50, -10),
        (0, 0, 0),
    ):
        assert cs.fit(*args) is None, args


def check_group():
    """Group box rendering for covered/held/open, zero-total, and too-small-to-label boxes."""
    svg, labelled = cs.group("Integer", (3, 2, 7), 10, 20, 160, 80)
    assert labelled is True
    root = ET.fromstring(svg)
    assert root.find(".//text").text == "Integer 3/7"
    fills = {g.attrib["fill"]: len(g.findall("rect")) for g in root.findall("g")}
    assert fills == {cs.COVERED: 3, cs.HELD: 2, cs.OPEN: 2}, fills

    zero_svg, zero_labelled = cs.group("Empty", (0, 0, 0), 0, 0, 100, 60)
    assert zero_labelled is False
    zero_root = ET.fromstring(zero_svg)
    assert zero_root.find(".//text") is None
    assert {g.attrib["fill"]: len(g.findall("rect")) for g in zero_root.findall("g")} == {
        cs.COVERED: 0,
        cs.HELD: 0,
        cs.OPEN: 0,
    }

    small_svg, small_labelled = cs.group("VeryLongGroupName", (1, 0, 2), 0, 0, 24, 24)
    assert small_labelled is False
    small_root = ET.fromstring(small_svg)
    assert small_root.find(".//text") is None
    assert {g.attrib["fill"]: len(g.findall("rect")) for g in small_root.findall("g")} == {
        cs.COVERED: 1,
        cs.HELD: 0,
        cs.OPEN: 1,
    }


def check_render():
    """SVG rendering across covered, held, zero-total, and unlabelled groups plus validation."""
    data = _sample_coverage()
    banner = "abc1234, win-test & warp <1>"
    svg = cs.render(data, banner)
    root = ET.fromstring(svg)
    texts = [el.text for el in root.iter() if el.tag.endswith("text") and el.text]
    assert any("Measured on abc1234, win-test & warp <1>." in t for t in texts), texts
    assert any("Too small to label:" in t and "Tiny 0/1" in t for t in texts), texts
    assert not any("Zero 0/0" in t or "ZeroOnly 0/0" in t for t in texts), texts
    assert any("x64 (Zen 2 user-mode mnemonics): 4 of 1007 covered, 1 held back by skips" in t for t in texts), texts
    assert any("gfx1013 forms: 0 of 0 covered, 0 held back by skips" in t for t in texts), texts
    assert any("gfx1030 (RDNA2) forms: 2 of 5 covered, 1 held back by skips" in t for t in texts), texts

    # Negative render() controls pinning exact exceptions.
    bad_count = _sample_coverage()
    bad_count["x64_zen2"]["groups"]["Integer"]["total"] = -1
    for fn, exc_type in (
        (lambda: cs.render([], "s"), TypeError),
        (lambda: cs.render({}, "s"), KeyError),
        (lambda: cs.render(bad_count, "s"), ValueError),
    ):
        try:
            fn()
        except exc_type:
            pass
        else:
            raise AssertionError(f"expected {exc_type.__name__}")


def check_cli():
    """main(argv) positive SVG generation, CLI error exits, collision check, and atomic cleanup."""
    data = _sample_coverage()
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        cov = work / "coverage.json"
        cov.write_text(json.dumps(data), encoding="utf-8")
        out_dir = work / "out"
        out_svg = out_dir / "dashboard.svg"

        rc = cs.main([str(cov), str(out_svg), "--source", "test-commit, test-os, warp"])
        assert rc == 0, rc
        assert out_svg.is_file()
        assert out_svg.read_text(encoding="utf-8") == cs.render(data, "test-commit, test-os, warp")
        assert [p.name for p in out_dir.iterdir()] == ["dashboard.svg"]

        # End-to-end subprocess check with --source banner.
        sub_svg = out_dir / "sub.svg"
        done = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(cov), str(sub_svg), "--source", "sub-banner"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        assert done.stderr == ""
        assert "Measured on sub-banner." in sub_svg.read_text(encoding="utf-8")

        # Missing --source exits with argparse usage code 2 and no traceback.
        _expect_exit([str(cov), str(out_svg)], expected_code=2)

        # Missing input JSON -> clean exit(1) without traceback or temp leaks.
        missing_out = work / "missing_out.svg"
        _expect_exit([str(work / "does_not_exist.json"), str(missing_out), "--source", "x"])
        assert not missing_out.exists()

        # Unreadable input (directory path triggers OSError) -> clean exit(1).
        dir_in = work / "as_dir"
        dir_in.mkdir()
        _expect_exit([str(dir_in), str(missing_out), "--source", "x"])
        assert not missing_out.exists()

        # Invalid JSON syntax -> clean exit(1).
        bad_json = work / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        _expect_exit([str(bad_json), str(missing_out), "--source", "x"])
        assert not missing_out.exists()

        # Missing required keys -> clean exit(1).
        incomplete_json = work / "incomplete.json"
        incomplete_json.write_text('{"x64_zen2": {}}', encoding="utf-8")
        _expect_exit([str(incomplete_json), str(missing_out), "--source", "x"])
        assert not missing_out.exists()

        # Output == input collision (exact and non-normalized alias) preserves input without temp files.
        original_bytes = cov.read_bytes()
        err = _expect_exit([str(cov), str(cov), "--source", "x"])
        assert "must not match input" in err, err
        assert cov.read_bytes() == original_bytes
        alias = work / "out" / ".." / "coverage.json"
        _expect_exit([str(cov), str(alias), "--source", "x"])
        assert cov.read_bytes() == original_bytes

        # Write failure cleanup: output path is an existing directory (real OSError).
        out_as_dir = work / "out_dir_target"
        out_as_dir.mkdir()
        _expect_exit([str(cov), str(out_as_dir), "--source", "x"])
        assert list(out_as_dir.iterdir()) == []
        assert not any(p.name.endswith(".tmp") for p in work.iterdir())

        # Write failure cleanup: simulated replace failure preserves existing file and leaves zero temp files.
        sentinel_dir = work / "sentinel_dir"
        sentinel_dir.mkdir()
        sentinel = sentinel_dir / "keep.svg"
        sentinel.write_text("sentinel", encoding="utf-8")
        orig_replace = Path.replace

        def fail_replace(self, target):
            raise OSError("simulated replace failure")

        Path.replace = fail_replace
        try:
            err = _expect_exit([str(cov), str(sentinel), "--source", "x"])
            assert "simulated replace failure" in err, err
        finally:
            Path.replace = orig_replace
        assert sentinel.read_text(encoding="utf-8") == "sentinel"
        assert [p.name for p in sentinel_dir.iterdir()] == ["keep.svg"]

        # Subprocess negative check for missing JSON (no traceback on stderr).
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(work / "nope.json"), str(missing_out), "--source", "x"],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1
        assert sub_err.stderr.strip() and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    check_squarify()
    check_fit()
    check_group()
    check_render()
    check_cli()
    print("coverage_svg: treemap guards, fitting, rendering, CLI errors, and atomic writes passed")


if __name__ == "__main__":
    main()
