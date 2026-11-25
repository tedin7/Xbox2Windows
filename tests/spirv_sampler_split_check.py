#!/usr/bin/env python3
"""Check tools/spirv_sampler_split.py: pass-through, single/multi-sampler splitting,
and CLI error/collision/atomic-write handling without tracebacks or temp-file leaks."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "spirv_sampler_split.py"
FIXTURES = ROOT / "tests" / "spirv_window"
sys.path.insert(0, str(ROOT / "tools"))
import spirv_sampler_split as sss

SHARED = (FIXTURES / "shared_sampler.spvasm").read_text(encoding="utf-8")
SHARED_MULTI = (FIXTURES / "shared_sampler_multi.spvasm").read_text(encoding="utf-8")
INDEXED_TABLE = (FIXTURES / "sampler.spvasm").read_text(encoding="utf-8")


def _sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _expect_value_error(assembly, ordinary_binding, fragment):
    try:
        sss.split_comparison_sampler(assembly, ordinary_binding)
    except ValueError as error:
        assert fragment in str(error), str(error)
        return str(error)
    raise AssertionError(f"expected ValueError containing {fragment!r}")


def _expect_exit(argv, expected_code=1, fragment=""):
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            sss.main(argv)
    except SystemExit as exc:
        assert exc.code == expected_code, (exc.code, stderr.getvalue())
    else:
        raise AssertionError(f"main({argv!r}) did not raise SystemExit")
    err = stderr.getvalue()
    assert err.strip(), f"expected error message on stderr for {argv!r}"
    assert "Traceback" not in err, err
    if fragment:
        assert fragment in err, err
    return err


def check_positive_controls():
    """Pass-through, single OpAccessChain, and multi-sampler OpInBoundsAccessChain splitting."""
    # 1. Pass-through: already-separated samplers, comparison-only sampler, and indexed table.
    separated = _sub(SHARED, "%41 = OpLoad %type_sampler %cmpSampler", "%41 = OpLoad %type_sampler %plainSampler")
    cmp_only = _sub(SHARED, "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
                    "%43 = OpImageSampleDrefExplicitLod %float %42 %15 %float_0_25 Lod %float_0")
    for name, module in (("separated", separated), ("comparison_only", cmp_only), ("indexed_table", INDEXED_TABLE)):
        assert sss.split_comparison_sampler(module, 5) == (module, None), name

    # 2. Single mixed sampler with OpAccessChain storage buffer indexing.
    assert "OpAccessChain" in SHARED
    lowered, mapping = sss.split_comparison_sampler(SHARED, 5)
    assert mapping == {"descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": 5}, mapping
    lines = [line.strip() for line in lowered.splitlines()]
    assert "%x2w_plain_sampler = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant" in lines
    assert "OpDecorate %x2w_plain_sampler DescriptorSet 0" in lines
    assert "OpDecorate %x2w_plain_sampler Binding 5" in lines
    assert "%x2w_plain_sampler_0 = OpLoad %type_sampler %x2w_plain_sampler" in lines
    assert "%42 = OpSampledImage %type_sampled_image %40 %x2w_plain_sampler_0" in lines
    assert sum(line.endswith("%type_sampler %cmpSampler") for line in lines) == 2
    entry_line = next(line for line in lines if line.startswith("OpEntryPoint"))
    assert entry_line.split()[-1] == "%x2w_plain_sampler", entry_line
    assert any("OpAccessChain" in line for line in lines)

    # 3. Multi-sampler module with OpInBoundsAccessChain storage buffer indexing.
    inbounds_multi = SHARED_MULTI.replace("OpAccessChain", "OpInBoundsAccessChain")
    assert inbounds_multi != SHARED_MULTI
    multi_lowered, multi_mapping = sss.split_comparison_sampler(inbounds_multi, [5, 7])
    assert multi_mapping == [
        {"comparison_sampler": "%cmpA", "plain_sampler": "%x2w_plain_sampler_0",
         "descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": 5},
        {"comparison_sampler": "%cmpB", "plain_sampler": "%x2w_plain_sampler_1",
         "descriptor_set": 0, "comparison_binding": 6, "ordinary_binding": 7},
    ], multi_mapping
    multi_lines = [line.strip() for line in multi_lowered.splitlines()]
    for plain, binding, cmp_id, img_id, arg_id in (
        ("%x2w_plain_sampler_0", 5, "%cmpA", "%43", "%41"),
        ("%x2w_plain_sampler_1", 7, "%cmpB", "%48", "%41"),
    ):
        assert f"{plain} = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant" in multi_lines
        assert f"OpDecorate {plain} DescriptorSet 0" in multi_lines
        assert f"OpDecorate {plain} Binding {binding}" in multi_lines
        assert f"{plain}_load_0 = OpLoad %type_sampler {plain}" in multi_lines
        assert f"{img_id} = OpSampledImage %type_sampled_image {arg_id} {plain}_load_0" in multi_lines
        assert sum(line.endswith(f"%type_sampler {cmp_id}") for line in multi_lines) == 2, cmp_id
    multi_entry = next(line for line in multi_lines if line.startswith("OpEntryPoint"))
    assert multi_entry.split()[-2:] == ["%x2w_plain_sampler_0", "%x2w_plain_sampler_1"], multi_entry
    assert multi_lowered.count("OpInBoundsAccessChain") == inbounds_multi.count("OpInBoundsAccessChain")
    return separated, lowered, inbounds_multi, multi_lowered


def check_negative_controls_api(inbounds_multi):
    """Pin exact ValueError diagnostics for invalid bindings, interface, decorations, slots, and IDs."""
    _expect_value_error(SHARED, -1, "Ordinary sampler binding must be nonnegative")
    _expect_value_error(inbounds_multi, [5, -1], "Ordinary sampler binding must be nonnegative")
    _expect_value_error(inbounds_multi, [5, 5], "Ordinary sampler bindings must be distinct")
    _expect_value_error(inbounds_multi, [5], "found 2 samplers")
    _expect_value_error(
        _sub(SHARED, " %cmpSampler %plainSampler %results", " %plainSampler %results"),
        5,
        "%cmpSampler: sampler must be listed in the entry-point interface",
    )
    _expect_value_error(
        _sub(SHARED, "OpDecorate %cmpSampler DescriptorSet 0\n", ""),
        5,
        "%cmpSampler: sampler needs a DescriptorSet and Binding decoration",
    )
    _expect_value_error(
        _sub(SHARED, "OpDecorate %cmpSampler Binding 2\n", ""),
        5,
        "%cmpSampler: sampler needs a DescriptorSet and Binding decoration",
    )
    for occupied in (2, 3, 4):
        _expect_value_error(SHARED, occupied, f"Ordinary sampler binding {occupied} is occupied in set 0")
    _expect_value_error(
        _sub(SHARED, "OpDecorate %plainSampler DescriptorSet 0\n", ""),
        3,
        "Ordinary sampler binding 3 is occupied in set 0 by %plainSampler",
    )
    for reserved in ("%x2w_plain_sampler", "%x2w_plain_sampler_0", "%x2w_plain_sampler_1_load_0"):
        _expect_value_error(
            _sub(SHARED, "%float_0 = OpConstant %float 0", f"%float_0 = OpConstant %float 0\n{reserved} = OpConstant %float 0"),
            5,
            "Reserved sampler split ID is already in use",
        )


def check_cli(separated, lowered, inbounds_multi, multi_lowered):
    """CLI positive runs (single, multi, pass-through) and negative controls without tracebacks or temp leaks."""
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        single_in = work / "shared.spvasm"
        single_in.write_text(SHARED, encoding="utf-8")
        multi_in = work / "multi.spvasm"
        multi_in.write_text(inbounds_multi, encoding="utf-8")
        unchanged_in = work / "separated.spvasm"
        unchanged_in.write_text(separated, encoding="utf-8")

        out_dir = work / "out"
        single_out = out_dir / "single.spvasm"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = sss.main([str(single_in), str(single_out), "--ordinary-binding", "5"])
        assert rc == 0, rc
        assert single_out.read_text(encoding="utf-8") == lowered
        assert stdout.getvalue().split() == ["descriptor_set=0", "comparison_binding=2", "ordinary_binding=5"]
        assert [p.name for p in out_dir.iterdir()] == ["single.spvasm"]

        multi_out = out_dir / "multi.spvasm"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = sss.main([str(multi_in), str(multi_out), "--ordinary-binding", "5", "--ordinary-binding", "7"])
        assert rc == 0, rc
        assert multi_out.read_text(encoding="utf-8") == multi_lowered
        assert stdout.getvalue().splitlines() == [
            "comparison_sampler=%cmpA plain_sampler=%x2w_plain_sampler_0 descriptor_set=0 comparison_binding=2 ordinary_binding=5",
            "comparison_sampler=%cmpB plain_sampler=%x2w_plain_sampler_1 descriptor_set=0 comparison_binding=6 ordinary_binding=7",
        ]

        unchanged_out = out_dir / "unchanged.spvasm"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = sss.main([str(unchanged_in), str(unchanged_out), "--ordinary-binding", "5"])
        assert rc == 0, rc
        assert unchanged_out.read_text(encoding="utf-8") == separated
        assert stdout.getvalue().strip() == "unchanged: no sampler with mixed uses"

        # End-to-end subprocess positive run.
        sub_out = out_dir / "sub.spvasm"
        done = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(single_in), str(sub_out), "--ordinary-binding", "5"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stderr == "", done
        assert sub_out.read_text(encoding="utf-8") == lowered

        # Missing --ordinary-binding -> argparse exit 2, no traceback.
        _expect_exit([str(single_in), str(work / "unused.spvasm")], expected_code=2)

        # Missing input -> OSError / SystemExit(1), no output or temp files.
        missing_in = work / "does_not_exist.spvasm"
        missing_out = work / "missing_out.spvasm"
        try:
            missing_in.read_text(encoding="utf-8")
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for missing input")
        _expect_exit([str(missing_in), str(missing_out), "--ordinary-binding", "5"])
        assert not missing_out.exists()

        # Directory input -> OSError / SystemExit(1), no output or temp files.
        dir_in = work / "dir_input"
        dir_in.mkdir()
        try:
            dir_in.read_text(encoding="utf-8")
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for directory input")
        _expect_exit([str(dir_in), str(missing_out), "--ordinary-binding", "5"])
        assert not missing_out.exists()

        # Invalid UTF-8 input -> UnicodeDecodeError / SystemExit(1), no output or temp files.
        bad_utf8 = work / "bad_utf8.spvasm"
        bad_utf8.write_bytes(b"\xff\xfe\x00\x80")
        try:
            bad_utf8.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("expected UnicodeDecodeError for invalid UTF-8")
        _expect_exit([str(bad_utf8), str(missing_out), "--ordinary-binding", "5"])
        assert not missing_out.exists()

        # Output == input collision (exact and non-normalized alias) preserves input without temp files.
        original_bytes = single_in.read_bytes()
        _expect_exit([str(single_in), str(single_in), "--ordinary-binding", "5"], fragment="must not match input")
        assert single_in.read_bytes() == original_bytes
        alias = work / "out" / ".." / "shared.spvasm"
        _expect_exit([str(single_in), str(alias), "--ordinary-binding", "5"], fragment="must not match input")
        assert single_in.read_bytes() == original_bytes

        # Negative --ordinary-binding rejected cleanly before file-read or path-collision checks.
        _expect_exit([str(single_in), str(missing_out), "--ordinary-binding", "-1"], fragment="nonnegative")
        _expect_exit([str(unchanged_in), str(missing_out), "--ordinary-binding", "-1"], fragment="nonnegative")
        _expect_exit([str(missing_in), str(missing_out), "--ordinary-binding", "-1"], fragment="nonnegative")
        _expect_exit([str(single_in), str(single_in), "--ordinary-binding", "-1"], fragment="nonnegative")
        assert not missing_out.exists()

        # Duplicate --ordinary-binding rejected before file-read or path-collision checks.
        _expect_exit([str(multi_in), str(missing_out), "--ordinary-binding", "5", "--ordinary-binding", "5"],
                     fragment="must be distinct")
        _expect_exit([str(missing_in), str(missing_out), "--ordinary-binding", "5", "--ordinary-binding", "5"],
                     fragment="must be distinct")
        _expect_exit([str(single_in), str(single_in), "--ordinary-binding", "5", "--ordinary-binding", "5"],
                     fragment="must be distinct")
        _expect_exit([str(single_in), str(missing_out), "--ordinary-binding", "3"], fragment="occupied")

        missing_ep = work / "missing_ep.spvasm"
        missing_ep.write_text(_sub(SHARED, " %cmpSampler %plainSampler %results", " %plainSampler %results"), encoding="utf-8")
        _expect_exit([str(missing_ep), str(missing_out), "--ordinary-binding", "5"], fragment="entry-point interface")

        missing_set = work / "missing_set.spvasm"
        missing_set.write_text(_sub(SHARED, "OpDecorate %cmpSampler DescriptorSet 0\n", ""), encoding="utf-8")
        _expect_exit([str(missing_set), str(missing_out), "--ordinary-binding", "5"],
                     fragment="needs a DescriptorSet and Binding decoration")

        reserved_in = work / "reserved.spvasm"
        reserved_in.write_text(
            _sub(SHARED, "%float_0 = OpConstant %float 0",
                 "%float_0 = OpConstant %float 0\n%x2w_plain_sampler = OpConstant %float 0"),
            encoding="utf-8",
        )
        _expect_exit([str(reserved_in), str(missing_out), "--ordinary-binding", "5"],
                     fragment="Reserved sampler split ID is already in use")
        assert not missing_out.exists()

        # Write failure cleanup: output path is an existing directory (real OSError).
        out_as_dir = work / "out_dir_target"
        out_as_dir.mkdir()
        _expect_exit([str(single_in), str(out_as_dir), "--ordinary-binding", "5"])
        assert list(out_as_dir.iterdir()) == []
        assert not any(p.name.endswith(".tmp") for p in work.iterdir())

        # Write failure cleanup: simulated replace failure preserves existing file and leaves zero temp files.
        sentinel_dir = work / "sentinel_dir"
        sentinel_dir.mkdir()
        sentinel = sentinel_dir / "keep.spvasm"
        sentinel.write_text("sentinel", encoding="utf-8")
        orig_replace = Path.replace

        def fail_replace(self, target):
            raise OSError("simulated replace failure")

        Path.replace = fail_replace
        try:
            _expect_exit([str(single_in), str(sentinel), "--ordinary-binding", "5"],
                         fragment="simulated replace failure")
        finally:
            Path.replace = orig_replace
        assert sentinel.read_text(encoding="utf-8") == "sentinel"
        assert [p.name for p in sentinel_dir.iterdir()] == ["keep.spvasm"]

        # Subprocess negative check for missing input (no traceback on stderr).
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(missing_in), str(missing_out), "--ordinary-binding", "5"],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1
        assert sub_err.stderr.strip() and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    separated, lowered, inbounds_multi, multi_lowered = check_positive_controls()
    check_negative_controls_api(inbounds_multi)
    check_cli(separated, lowered, inbounds_multi, multi_lowered)
    print("spirv_sampler_split: pass-through, single/multi splitting, CLI errors, and atomic writes passed")


if __name__ == "__main__":
    main()
