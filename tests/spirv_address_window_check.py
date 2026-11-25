#!/usr/bin/env python3
"""Unit and CLI suite for tools/spirv_address_window.py without external SPIR-V binaries."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "spirv_address_window.py"
FIXTURES = ROOT / "tests" / "spirv_window"
sys.path.insert(0, str(ROOT / "tools"))
import spirv_address_window as saw

SUBTRACT = (FIXTURES / "subtract.spvasm").read_text(encoding="utf-8")
SIGNED = (FIXTURES / "address_signed.spvasm").read_text(encoding="utf-8")
FLOAT = (FIXTURES / "address_float.spvasm").read_text(encoding="utf-8")
VEC4 = (FIXTURES / "address_vec4.spvasm").read_text(encoding="utf-8")
QWORD = (FIXTURES / "qword.spvasm").read_text(encoding="utf-8")


def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _expect_value_error(fn, *args, fragment: str):
    try:
        fn(*args)
    except ValueError as error:
        assert fragment in str(error), f"expected {fragment!r} in {error!r}"
        return str(error)
    raise AssertionError(f"{fn.__name__} did not raise ValueError containing {fragment!r}")


def _expect_exit(argv, expected_code=1, fragment=""):
    stderr = io.StringIO()
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            saw.main(argv)
    except SystemExit as exc:
        assert exc.code == expected_code, (exc.code, stderr.getvalue())
    else:
        raise AssertionError(f"main({argv!r}) did not raise SystemExit")
    err = stderr.getvalue()
    assert err.strip(), f"expected stderr message for {argv!r}"
    assert "Traceback" not in err, err
    if fragment:
        assert fragment in err, f"expected {fragment!r} in {err!r}"
    return err


def check_word_pointer():
    """Verify word_pointer() for direct, indexed, deferred, wide, and vec4 modes."""
    lines, valid = saw.word_pointer("%p", "%addr", None)
    assert valid == "%x2w_address_p_valid", valid
    assert lines == [
        "%x2w_address_p_low = OpBitwiseAnd %ulong %addr %x2w_mask",
        "%x2w_address_p_aligned = OpIEqual %bool %x2w_address_p_low %x2w_zero64",
        "%x2w_address_p_range = OpULessThan %bool %addr %x2w_end",
        "%x2w_address_p_valid = OpLogicalAnd %bool %x2w_address_p_aligned %x2w_address_p_range",
        "%x2w_address_p_word64 = OpShiftRightLogical %ulong %addr %x2w_shift",
        "%x2w_address_p_word = OpUConvert %uint %x2w_address_p_word64",
        "%x2w_address_p_selected = OpSelect %uint %x2w_address_p_valid %x2w_address_p_word %x2w_invalid",
        "%p = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %x2w_address_p_selected",
    ], lines

    idx_lines, idx_valid = saw.word_pointer("%q", "%base", "%idx")
    assert idx_valid == "%x2w_address_q_checked", idx_valid
    assert idx_lines[:3] == [
        "%x2w_address_q_index = OpUConvert %ulong %idx",
        "%x2w_address_q_delta = OpShiftLeftLogical %ulong %x2w_address_q_index %x2w_shift",
        "%x2w_address_q_address = OpIAdd %ulong %base %x2w_address_q_delta",
    ], idx_lines
    assert "%x2w_address_q_nowrap = OpUGreaterThanEqual %bool %x2w_address_q_address %base" in idx_lines
    assert "%x2w_address_q_checked = OpLogicalAnd %bool %x2w_address_q_valid %x2w_address_q_nowrap" in idx_lines
    assert "%q = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %x2w_address_q_selected" in idx_lines

    defer_lines, defer_valid = saw.word_pointer("%d", "%base", None, defer=True)
    assert defer_valid == "%x2w_address_d_valid", defer_valid
    assert "%x2w_address_d_range = OpULessThan %bool %base %x2w_end" in defer_lines
    assert "%x2w_address_d_word = OpUConvert %uint %x2w_address_d_word64" in defer_lines
    assert not any("_selected" in line or "OpAccessChain" in line for line in defer_lines), defer_lines

    wide_lines, wide_valid = saw.word_pointer("%w", "%base", "%idx", wide=True)
    assert wide_valid == "%x2w_address_w_checked", wide_valid
    assert "%x2w_address_w_delta = OpShiftLeftLogical %ulong %x2w_address_w_index %x2w_shift_wide" in wide_lines
    assert "%x2w_address_w_low = OpBitwiseAnd %ulong %x2w_address_w_address %x2w_mask_wide" in wide_lines
    assert "%x2w_address_w_range = OpULessThan %bool %x2w_address_w_address %x2w_end_wide" in wide_lines
    assert not any("_selected" in line or "OpAccessChain" in line for line in wide_lines), wide_lines

    vec4_lines, vec4_valid = saw.word_pointer("%v", "%base", None, wide="vec4")
    assert vec4_valid == "%x2w_address_v_valid", vec4_valid
    assert "%x2w_address_v_low = OpBitwiseAnd %ulong %base %x2w_mask_vec4" in vec4_lines
    assert "%x2w_address_v_range = OpULessThan %bool %base %x2w_end_vec4" in vec4_lines
    assert not any("_selected" in line or "OpAccessChain" in line for line in vec4_lines), vec4_lines


def check_signed_access():
    """Verify signed_access() for loads, stores, SMin/SMax CAS loops, word atomics, and rejections."""
    valid_of = {"%p": "%v_p"}
    words = {"%p": "%w_p"}
    failures = {"%c72": "%x2w_failure_66", "%c0": "%c0"}

    load_lines, next_lbl = saw.signed_access(
        "%r = OpLoad %int %p Aligned 4".split(), {"%p"}, valid_of, words, "%entry", "%s", failures
    )
    assert next_lbl == "%s_merge", next_lbl
    assert load_lines == [
        "OpSelectionMerge %s_merge None",
        "OpBranchConditional %v_p %s_issue %s_merge",
        "%s_issue = OpLabel",
        "%s_ptr = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %w_p",
        "%s_loaded = OpLoad %uint %s_ptr Aligned 4",
        "OpBranch %s_merge",
        "%s_merge = OpLabel",
        "%s_old = OpPhi %uint %s_loaded %s_issue %c0 %entry",
        "%r = OpBitcast %int %s_old",
    ], load_lines

    plain_load, _ = saw.signed_access(
        "%r = OpLoad %int %p".split(), {"%p"}, valid_of, words, "%entry", "%sp", failures
    )
    assert "%sp_loaded = OpLoad %uint %sp_ptr" in plain_load

    store_lines, next_lbl = saw.signed_access(
        "OpStore %p %val Aligned 4".split(), {"%p"}, valid_of, words, "%entry", "%st", failures
    )
    assert next_lbl == "%st_merge"
    assert "%st_value = OpBitcast %uint %val" in store_lines
    assert "OpStore %st_ptr %st_value Aligned 4" in store_lines

    smin_lines, next_lbl = saw.signed_access(
        "%m = OpAtomicSMin %int %p %c1 %c72 %val".split(), {"%p"}, valid_of, words, "%entry", "%mn", failures
    )
    assert next_lbl == "%mn_merge"
    assert "OpLoopMerge %mn_exit %mn_cont None" in smin_lines
    assert "%mn_pick = OpSLessThan %bool %mn_cur_int %val" in smin_lines
    assert "%mn_seen = OpAtomicCompareExchange %uint %mn_ptr %c1 %c72 %x2w_failure_66 %mn_want %mn_cur" in smin_lines

    smax_lines, _ = saw.signed_access(
        "%m = OpAtomicSMax %int %p %c1 %c0 %val".split(), {"%p"}, valid_of, words, "%entry", "%mx", failures
    )
    assert "%mx_pick = OpSGreaterThan %bool %mx_cur_int %val" in smax_lines
    assert "%mx_seen = OpAtomicCompareExchange %uint %mx_ptr %c1 %c0 %c0 %mx_want %mx_cur" in smax_lines

    for op in ("OpAtomicIAdd", "OpAtomicExchange", "OpAtomicAnd", "OpAtomicOr", "OpAtomicXor"):
        wa_lines, wa_lbl = saw.signed_access(
            f"%o = {op} %int %p %c1 %c0 %val".split(), {"%p"}, valid_of, words, "%entry", "%wa", failures
        )
        assert wa_lbl == "%wa_merge"
        assert "%wa_value = OpBitcast %uint %val" in wa_lines
        assert f"%wa_old = {op} %uint %wa_ptr %c1 %c0 %wa_value" in wa_lines
        assert "%o = OpBitcast %int %wa_old_phi" not in wa_lines
        assert "%o = OpBitcast %int %wa_result" in wa_lines

    cmpxchg_lines, _ = saw.signed_access(
        "%o = OpAtomicCompareExchange %int %p %c1 %c72 %c66 %val %cmp".split(),
        {"%p"}, valid_of, words, "%entry", "%cx", failures,
    )
    assert "%cx_value = OpBitcast %uint %val" in cmpxchg_lines
    assert "%cx_cmp = OpBitcast %uint %cmp" in cmpxchg_lines
    assert "%cx_old = OpAtomicCompareExchange %uint %cx_ptr %c1 %c72 %c66 %cx_value %cx_cmp" in cmpxchg_lines

    for bad_line, ptrs in (
        ("%r = OpLoad %int %p Volatile", {"%p"}),
        ("%r = OpLoad %uint %p Aligned 4", {"%p"}),
        ("OpStore %p %val Aligned 8", {"%p"}),
        ("%r = OpAtomicISub %int %p %c1 %c0 %val", {"%p"}),
        ("%r = OpAtomicSMin %int %p %c1 %c0", {"%p"}),
        ("%r = OpLoad %int %p Aligned 4", {"%p", "%q"}),
        ("%r = OpLoad %int %other Aligned 4", {"%p"}),
    ):
        _expect_value_error(
            saw.signed_access, bad_line.split(), ptrs, valid_of, words, "%entry", "%bad", failures,
            fragment="unhandled physical word pointer use",
        )


def check_float_access():
    """Verify float_access() for loads, stores, bitcasts, and rejections."""
    valid_of = {"%pf": "%v_pf"}
    words = {"%pf": "%w_pf"}

    load_lines, next_lbl = saw.float_access(
        "%f = OpLoad %float %pf Aligned 4".split(), {"%pf"}, valid_of, words, "%entry", "%fl"
    )
    assert next_lbl == "%fl_merge"
    assert load_lines == [
        "OpSelectionMerge %fl_merge None",
        "OpBranchConditional %v_pf %fl_issue %fl_merge",
        "%fl_issue = OpLabel",
        "%fl_ptr = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %w_pf",
        "%fl_loaded = OpLoad %uint %fl_ptr Aligned 4",
        "OpBranch %fl_merge",
        "%fl_merge = OpLabel",
        "%fl_old = OpPhi %uint %fl_loaded %fl_issue %c0 %entry",
        "%f = OpBitcast %float %fl_old",
    ], load_lines

    plain_load, _ = saw.float_access(
        "%f = OpLoad %float %pf".split(), {"%pf"}, valid_of, words, "%entry", "%fp"
    )
    assert "%fp_loaded = OpLoad %uint %fp_ptr" in plain_load

    store_lines, next_lbl = saw.float_access(
        "OpStore %pf %val Aligned 4".split(), {"%pf"}, valid_of, words, "%entry", "%fs"
    )
    assert next_lbl == "%fs_merge"
    assert "%fs_value = OpBitcast %uint %val" in store_lines
    assert "OpStore %fs_ptr %fs_value Aligned 4" in store_lines

    for bad_line, ptrs in (
        ("%f = OpLoad %float %pf Volatile", {"%pf"}),
        ("%f = OpLoad %int %pf Aligned 4", {"%pf"}),
        ("OpStore %pf %val Aligned 8", {"%pf"}),
        ("OpStore %pf %pf Aligned 4", {"%pf"}),
        ("%f = OpAtomicIAdd %float %pf %c1 %c0 %val", {"%pf"}),
        ("%f = OpLoad %float %pf Aligned 4", {"%pf", "%qf"}),
    ):
        _expect_value_error(
            saw.float_access, bad_line.split(), ptrs, valid_of, words, "%entry", "%badf",
            fragment="unhandled physical word pointer use",
        )


def check_vec4_access():
    """Verify vec4_access() for 16-byte aligned loads/stores and rejections."""
    valid_of = {"%pv": "%v_pv"}
    words = {"%pv": "%w_pv"}

    load_lines, next_lbl = saw.vec4_access(
        "%v = OpLoad %v4float %pv Aligned 16".split(), {"%pv"}, valid_of, words, "%entry", "%vl"
    )
    assert next_lbl == "%vl_merge"
    assert "%vl_word_1 = OpIAdd %uint %w_pv %x2w_one" in load_lines
    assert "%vl_word_2 = OpIAdd %uint %w_pv %x2w_two" in load_lines
    assert "%vl_word_3 = OpIAdd %uint %w_pv %x2w_three" in load_lines
    assert "%vl_ptr_0 = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %w_pv" in load_lines
    assert "%vl_ptr_3 = OpAccessChain %_ptr_PSB_uint %x2w_heap %c0 %vl_word_3" in load_lines
    assert "%vl_lane_2 = OpBitcast %float %vl_bits_2" in load_lines
    assert "%vl_value = OpCompositeConstruct %v4float %vl_lane_0 %vl_lane_1 %vl_lane_2 %vl_lane_3" in load_lines
    assert "%v = OpPhi %v4float %vl_value %vl_issue %x2w_vec4_zero %entry" in load_lines

    store_lines, next_lbl = saw.vec4_access(
        "OpStore %pv %vec Aligned 16".split(), {"%pv"}, valid_of, words, "%entry", "%vs"
    )
    assert next_lbl == "%vs_merge"
    for lane in range(4):
        assert f"%vs_lane_{lane} = OpCompositeExtract %float %vec {lane}" in store_lines
        assert f"%vs_bits_{lane} = OpBitcast %uint %vs_lane_{lane}" in store_lines
        assert f"OpStore %vs_ptr_{lane} %vs_bits_{lane} Aligned 4" in store_lines

    for bad_line, ptrs in (
        ("%v = OpLoad %v4float %pv", {"%pv"}),
        ("%v = OpLoad %v4float %pv Aligned 4", {"%pv"}),
        ("OpStore %pv %vec", {"%pv"}),
        ("OpStore %pv %vec Aligned 4", {"%pv"}),
        ("OpStore %pv %pv Aligned 16", {"%pv"}),
        ("%v = OpLoad %float %pv Aligned 16", {"%pv"}),
        ("%v = OpLoad %v4float %pv Aligned 16", {"%pv", "%qv"}),
    ):
        _expect_value_error(
            saw.vec4_access, bad_line.split(), ptrs, valid_of, words, "%entry", "%badv",
            fragment="unhandled physical word pointer use",
        )


def check_merge_pointer_types():
    """Verify merge_pointer_types() deduplicates identical logical pointers and propagates aliases."""
    lines = [
        "%_ptr_PSB_uint = OpTypePointer StorageBuffer %uint",
        "%_ptr_PC_uint = OpTypePointer StorageBuffer %uint",
        "%_ptr_PSB_ptr1 = OpTypePointer Function %_ptr_PSB_uint",
        "%_ptr_PSB_ptr2 = OpTypePointer Function %_ptr_PC_uint",
        "%_ptr_Uniform_uint = OpTypePointer Uniform %uint",
        "%x = OpAccessChain %_ptr_PC_uint %user_data %c0 %c0",
        "%y = OpVariable %_ptr_PSB_ptr2 Function",
    ]
    merged = saw.merge_pointer_types(lines)
    out = merged.splitlines()
    assert out == [
        "%_ptr_PSB_uint = OpTypePointer StorageBuffer %uint",
        "%_ptr_PSB_ptr1 = OpTypePointer Function %_ptr_PSB_uint",
        "%_ptr_Uniform_uint = OpTypePointer Uniform %uint",
        "%x = OpAccessChain %_ptr_PSB_uint %user_data %c0 %c0",
        "%y = OpVariable %_ptr_PSB_ptr1 Function",
    ], out


def check_lower_positive_and_composed():
    """Verify lower() across unsigned, signed, float, vec4, qword, OpPhi rewriting, CAS semantics, and composition."""
    # 1. Unsigned direct + indexed + all 8 word atomics.
    direct = _sub(
        SUBTRACT,
        "%v1 = OpConvertUToPtr %_ptr_PSB_arr %a1\n%p1 = OpAccessChain %_ptr_PSB_uint %v1 %c0",
        "%p1 = OpConvertUToPtr %_ptr_PSB_uint %a1",
    )
    atomic_ops = (
        "%old_add = OpAtomicIAdd %uint %p1 %c1 %atomic_rmw %d",
        "%old_exch = OpAtomicExchange %uint %p2 %c1 %atomic_rmw %old_add",
        "%old_umin = OpAtomicUMin %uint %p2 %c1 %atomic_rmw %old_exch",
        "%old_umax = OpAtomicUMax %uint %p2 %c1 %atomic_rmw %old_umin",
        "%old_and = OpAtomicAnd %uint %p2 %c1 %atomic_rmw %old_umax",
        "%old_or = OpAtomicOr %uint %p2 %c1 %atomic_rmw %old_and",
        "%old_xor = OpAtomicXor %uint %p2 %c1 %atomic_rmw %old_or",
        "%old_cmp = OpAtomicCompareExchange %uint %p2 %c1 %atomic_rmw %atomic_fail %d %old_xor",
    )
    atomics_src = _sub(
        _sub(direct, "%main = OpFunction", "%atomic_rmw = OpConstant %uint 72\n%atomic_fail = OpConstant %uint 66\n%main = OpFunction"),
        "OpStore %p2 %d Aligned 4",
        "\n".join(atomic_ops) + "\nOpStore %p0 %old_cmp Aligned 4",
    )
    atomics_out = saw.lower(atomics_src)
    assert "PhysicalStorageBuffer" not in atomics_out and "OpConvertUToPtr" not in atomics_out
    assert "OpMemoryModel Logical GLSL450" in atomics_out
    assert atomics_out.count("%_ptr_PSB_uint = OpTypePointer StorageBuffer %uint") == 1
    assert "%_ptr_PC_uint =" not in atomics_out
    assert atomics_out.count(" = OpAtomic") == 8

    # 2. Signed + acquire-release and release failure-semantics downgrade in SMin/SMax CAS loops.
    signed_acq_rel = _sub(
        _sub(SIGNED, "%c6 = OpConstant %uint 6", "%c6 = OpConstant %uint 6\n%c72 = OpConstant %uint 72\n%c68 = OpConstant %uint 68\n%c66 = OpConstant %uint 66"),
        "%old_min = OpAtomicSMin %int %min_target %c1 %c0 %operand",
        "%old_min = OpAtomicSMin %int %min_target %c1 %c68 %operand",
    )
    signed_acq_rel = _sub(
        signed_acq_rel,
        "%old_max = OpAtomicSMax %int %max_target %c1 %c0 %operand",
        "%old_max = OpAtomicSMax %int %max_target %c1 %c72 %operand",
    )
    # Also exercise direct %_ptr_PSB_int OpConvertUToPtr and an acquire-only SMin (no downgrade needed).
    signed_acq_rel = _sub(
        signed_acq_rel,
        "%old_add = OpAtomicIAdd %int %chain_target %c1 %c0 %operand",
        "%acq_min = OpAtomicSMin %int %output_first %c1 %c66 %operand\n%old_add = OpAtomicIAdd %int %chain_target %c1 %c0 %acq_min",
    )
    signed_out = saw.lower(signed_acq_rel)
    signed_lines = signed_out.splitlines()
    assert "%x2w_failure_64 = OpConstant %uint 64" in signed_lines
    assert "%x2w_failure_66 = OpConstant %uint 66" in signed_lines
    assert not any(line.startswith("%x2w_failure_66 = ") and line.count("OpConstant") > 1 for line in signed_lines)
    assert any(
        "OpAtomicCompareExchange %uint %x2w_signed_result_old_min_ptr %c1 %c68 %x2w_failure_64 " in line
        for line in signed_lines
    )
    assert any(
        "OpAtomicCompareExchange %uint %x2w_signed_result_old_max_ptr %c1 %c72 %x2w_failure_66 " in line
        for line in signed_lines
    )
    assert any(
        "OpAtomicCompareExchange %uint %x2w_signed_result_acq_min_ptr %c1 %c66 %c66 " in line
        for line in signed_lines
    )

    # 3. Float + vec4 + qword lowering.
    float_out = saw.lower(FLOAT)
    assert "%x2w_float_result_first_old = OpPhi %uint" in float_out
    assert "%first = OpBitcast %float %x2w_float_result_first_old" in float_out
    assert "PhysicalStorageBuffer" not in float_out

    vec4_out = saw.lower(VEC4)
    assert "%x2w_end_vec4 = OpConstant %ulong 4294967280" in vec4_out
    assert "%x2w_vec4_zero = OpConstantNull %v4float" in vec4_out
    assert "%first = OpPhi %v4float %x2w_vec4_result_first_value %x2w_vec4_result_first_issue %x2w_vec4_zero %entry" in vec4_out

    qword_out = saw.lower(QWORD)
    assert "%x2w_end_wide = OpConstant %ulong 4294967288" in qword_out
    assert "%value = OpPhi %ulong %x2w_load_value_value %x2w_load_value_issue %x2w_zero64 %entry" in qword_out

    # 4. OpPhi predecessor rewriting across guarded splits (atomic, float, qword).
    phi_src = _sub(
        _sub(direct, "%main = OpFunction", "%atomic_rmw = OpConstant %uint 72\n%main = OpFunction"),
        "OpStore %p2 %d Aligned 4",
        "%old_add = OpAtomicIAdd %uint %p1 %c1 %atomic_rmw %d\n"
        "OpBranch %next\n"
        "%next = OpLabel\n"
        "%carried = OpPhi %uint %old_add %entry\n"
        "OpStore %p2 %carried Aligned 4",
    )
    phi_out = saw.lower(phi_src).splitlines()
    assert "%carried = OpPhi %uint %old_add %x2w_atomic_old_add_merge" in phi_out

    phi_qword = _sub(
        QWORD,
        "OpReturn\nOpFunctionEnd",
        "OpBranch %exit\n%exit = OpLabel\n%copied = OpPhi %ulong %result %entry\nOpStore %target %copied Aligned 8\nOpReturn\nOpFunctionEnd",
    )
    phi_qword_out = saw.lower(phi_qword).splitlines()
    assert "%copied = OpPhi %ulong %result %x2w_store_2_merge" in phi_qword_out

    # 5. Composed ballot extract + ballot find-lsb + sampler-view splitting + address window.
    composed = (
        SUBTRACT
        .replace("OpCapability Shader\n", "OpCapability Shader\nOpCapability GroupNonUniformBallot\n", 1)
        .replace('OpEntryPoint GLCompute %main "main" %user_data',
                 'OpEntryPoint GLCompute %main "main" %user_data %tex %smp_tbl', 1)
        .replace("OpDecorate %rt_uint ArrayStride 4\n",
                 "OpDecorate %tex DescriptorSet 0\nOpDecorate %tex Binding 0\n"
                 "OpDecorate %smp_tbl DescriptorSet 0\nOpDecorate %smp_tbl Binding 1\n"
                 "OpDecorate %rt_uint ArrayStride 4\n", 1)
        .replace("%void = OpTypeVoid\n",
                 "%void = OpTypeVoid\n%float = OpTypeFloat 32\n%v2f = OpTypeVector %float 2\n"
                 "%v4f = OpTypeVector %float 4\n%v4u = OpTypeVector %uint 4\n"
                 "%img_t = OpTypeImage %float 2D 0 0 0 1 Unknown\n"
                 "%si_t = OpTypeSampledImage %img_t\n%smp_t = OpTypeSampler\n"
                 "%rt_smp = OpTypeRuntimeArray %smp_t\n"
                 "%ptr_img = OpTypePointer UniformConstant %img_t\n"
                 "%ptr_rt_smp = OpTypePointer UniformConstant %rt_smp\n"
                 "%ptr_smp = OpTypePointer UniformConstant %smp_t\n"
                 "%tex = OpVariable %ptr_img UniformConstant\n"
                 "%smp_tbl = OpVariable %ptr_rt_smp UniformConstant\n", 1)
        .replace("%c4 = OpConstant %uint 4\n",
                 "%c4 = OpConstant %uint 4\n%f0 = OpConstant %float 0\n%f05 = OpConstant %float 0.5\n"
                 "%uv = OpConstantComposite %v2f %f05 %f05\n"
                 "%bmask = OpConstantComposite %v4u %c1 %c2 %c3 %c4\n", 1)
        .replace("OpStore %p2 %d Aligned 4",
                 "%sp = OpAccessChain %ptr_smp %smp_tbl %c0\n"
                 "%sv = OpLoad %smp_t %sp\n"
                 "%im = OpLoad %img_t %tex\n"
                 "%si = OpSampledImage %si_t %im %sv\n"
                 "%ord = OpImageSampleExplicitLod %v4f %si %uv Lod %f0\n"
                 "%dref = OpImageSampleDrefExplicitLod %float %si %uv %f05 Lod %f0\n"
                 "%bbit = OpGroupNonUniformBallotBitExtract %bool %c3 %bmask %c0\n"
                 "%blsb = OpGroupNonUniformBallotFindLSB %uint %c3 %bmask\n"
                 "%bsel = OpSelect %uint %bbit %blsb %c0\n"
                 "%df = OpBitcast %uint %dref\n"
                 "%sum = OpIAdd %uint %bsel %df\n"
                 "OpStore %p2 %sum Aligned 4", 1)
    )
    composed_out = saw.lower(composed)
    assert "OpGroupNonUniformBallotBitExtract" not in composed_out
    assert "OpGroupNonUniformBallotFindLSB" not in composed_out
    assert "%x2w_ballot_bbit_" in composed_out
    assert "%x2w_ballot_blsb_" in composed_out
    assert "%bridge_cmp = OpVariable %ptr_rt_smp UniformConstant" in composed_out
    assert "OpDecorate %bridge_cmp DescriptorSet 3" in composed_out
    assert "OpDecorate %x2w_heap DescriptorSet 2" in composed_out
    assert "OpDecorate %user_data DescriptorSet 1" in composed_out
    assert "OpEntryPoint GLCompute %main \"main\" %user_data %tex %smp_tbl %bridge_cmp %x2w_heap" in composed_out


def check_lower_rejections():
    """Verify every rejection path in lower()."""
    direct = _sub(
        SUBTRACT,
        "%v1 = OpConvertUToPtr %_ptr_PSB_arr %a1\n%p1 = OpAccessChain %_ptr_PSB_uint %v1 %c0",
        "%p1 = OpConvertUToPtr %_ptr_PSB_uint %a1",
    )
    cases = [
        (_sub(SUBTRACT, "OpMemoryModel PhysicalStorageBuffer64 GLSL450", "OpMemoryModel Logical GLSL450"),
         "named user-data block"),
        (_sub(SUBTRACT, "%user_data = OpVariable %_ptr_PC_UserData PushConstant",
              "%other_ud = OpVariable %_ptr_PC_UserData PushConstant"),
         "named user-data block"),
        (SUBTRACT.replace("%d ", "%x2w_d "), "unused identifier prefix x2w_"),
        (_sub(SUBTRACT, "OpDecorate %UserData Block",
              "OpDecorate %user_data DescriptorSet 1\nOpDecorate %UserData Block"),
         "descriptor sets 1 and 2"),
        (_sub(SUBTRACT, "OpDecorate %UserData Block",
              "OpDecorate %user_data DescriptorSet 2\nOpDecorate %UserData Block"),
         "descriptor sets 1 and 2"),
        (_sub(SUBTRACT, 'OpEntryPoint GLCompute %main "main" %user_data', ""), "expected one shader entry point"),
        (_sub(SUBTRACT, 'OpEntryPoint GLCompute %main "main" %user_data',
              'OpEntryPoint GLCompute %main "main" %user_data\nOpEntryPoint GLCompute %main "other" %user_data'),
         "expected one shader entry point"),
        (_sub(SUBTRACT, "OpDecorate %rt_uint ArrayStride 4", "OpDecorate %rt_uint ArrayStride 8"),
         "unsupported physical pointer or scalar layout"),
        (_sub(QWORD, "%_ptr_PSB_ulong = OpTypePointer PhysicalStorageBuffer %ulong", ""),
         "unsupported physical pointer or scalar layout"),
        (_sub(QWORD, "OpDecorate %rt_ulong ArrayStride 8", "OpDecorate %rt_ulong ArrayStride 4"),
         "unsupported physical pointer or scalar layout"),
        (_sub(SIGNED, "%_ptr_PSB_int = OpTypePointer PhysicalStorageBuffer %int", ""),
         "unsupported physical pointer or scalar layout"),
        (_sub(FLOAT, "%_ptr_PSB_float = OpTypePointer PhysicalStorageBuffer %float", ""),
         "unsupported physical pointer or scalar layout"),
        (_sub(VEC4, "%_ptr_PSB_v4float = OpTypePointer PhysicalStorageBuffer %v4float", ""),
         "unsupported physical pointer or scalar layout"),
        (_sub(SUBTRACT, "%v0 = OpConvertUToPtr %_ptr_PSB_arr %a0\n", ""), "unknown physical array pointer"),
        (_sub(SIGNED, "%array = OpConvertUToPtr %_ptr_PSB_arri %base\n", ""), "unknown physical array pointer"),
        (_sub(FLOAT, "%array = OpConvertUToPtr %_ptr_PSB_arrf %base\n", ""), "unknown physical array pointer"),
        (_sub(VEC4, "%array = OpConvertUToPtr %_ptr_PSB_arrv4f %base\n", ""), "unknown physical array pointer"),
        (_sub(QWORD, "%array = OpConvertUToPtr %_ptr_PSB_arr64 %base\n", ""), "unknown physical array pointer"),
        (_sub(SUBTRACT, "%main = OpFunction", "%unexpected_pc = OpTypePointer PushConstant %uint\n%main = OpFunction"),
         "unrecognized push-constant declaration"),
        (_sub(SUBTRACT, "OpStore %p2 %d", "%leak = OpConvertPtrToU %ulong %p2\nOpStore %p2 %d"),
         "unhandled physical word pointer use"),
        (_sub(SUBTRACT, "OpReturn", "%unused_psb = OpTypePointer PhysicalStorageBuffer %bool\nOpReturn"),
         "unhandled physical pointer operation"),
        (_sub(SIGNED, "%old_min = OpAtomicSMin %int %min_target %c1 %c0 %operand",
              "%old_min = OpAtomicSMin %int %min_target %c1 %non_const %operand"),
         "signed atomic semantics must be a constant"),
        (_sub(_sub(direct, "%main = OpFunction", "%atomic_rmw = OpConstant %uint 72\n%main = OpFunction"),
              "OpReturn\nOpFunctionEnd",
              "OpBranch %hdr\n%hdr = OpLabel\n%bad = OpAtomicIAdd %uint %p1 %c1 %atomic_rmw %d\n"
              "OpLoopMerge %lexit %lcont None\nOpBranch %lcont\n%lcont = OpLabel\nOpBranch %hdr\n%lexit = OpLabel\nOpReturn\nOpFunctionEnd"),
         "share a block with OpLoopMerge"),
        (_sub(SIGNED, "OpReturn\nOpFunctionEnd",
              "OpBranch %hdr\n%hdr = OpLabel\n%bad = OpLoad %int %load_target Aligned 4\n"
              "OpLoopMerge %lexit %lcont None\nOpBranch %lcont\n%lcont = OpLabel\nOpBranch %hdr\n%lexit = OpLabel\nOpReturn\nOpFunctionEnd"),
         "share a block with OpLoopMerge"),
        (_sub(QWORD, "OpReturn\nOpFunctionEnd",
              "OpBranch %hdr\n%hdr = OpLabel\n%bad = OpLoad %ulong %source Aligned 8\n"
              "OpLoopMerge %lexit %lcont None\nOpBranch %lcont\n%lcont = OpLabel\nOpBranch %hdr\n%lexit = OpLabel\nOpReturn\nOpFunctionEnd"),
         "share a block with OpLoopMerge"),
    ]
    for source, reason in cases:
        _expect_value_error(saw.lower, source, fragment=reason)


def check_cli():
    """Verify main(argv) positive atomic writes, mocked tools, and every error/guard path."""
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        valid_src = work / "subtract.spvasm"
        valid_src.write_text(SUBTRACT, encoding="utf-8")
        sentinel_bytes = b"SENTINEL_KEEP_ME"

        calls = []
        fail_on_call = {}

        def fake_run(cmd, check=False, **_kwargs):
            calls.append(list(cmd))
            index = len(calls)
            if index in fail_on_call:
                raise fail_on_call[index]
            if cmd[0] == "mock-as":
                out_flag = cmd.index("-o")
                Path(cmd[out_flag + 1]).write_bytes(b"\x03\x02\x23\x07mock_spv")
            return subprocess.CompletedProcess(cmd, 0)

        orig_run = saw.subprocess.run
        saw.subprocess.run = fake_run
        try:
            # 1. Positive run into nested missing parent directories.
            nested_out = work / "nested" / "deep" / "window.spv"
            assert not nested_out.parent.exists()
            rc = saw.main([str(valid_src), str(nested_out), "--assembler", "mock-as", "--validator", "mock-val"])
            assert rc == 0, rc
            assert nested_out.read_bytes() == b"\x03\x02\x23\x07mock_spv"
            assert [p.name for p in nested_out.parent.iterdir()] == ["window.spv"]
            assert len(calls) == 4
            assert calls[0][:3] == ["mock-as", "--target-env", "vulkan1.2"] and calls[0][3].name == "original.spvasm"
            assert calls[1][:3] == ["mock-val", "--target-env", "vulkan1.2"]
            assert calls[2][:3] == ["mock-as", "--target-env", "vulkan1.2"] and calls[2][3].name == "logical.spvasm"
            assert calls[3][:3] == ["mock-val", "--target-env", "vulkan1.2"]

            # 2. Second run with existing parent directory (exercises exist_ok=True) and atomic replace.
            calls.clear()
            rc = saw.main([str(valid_src), str(nested_out), "--assembler", "mock-as", "--validator", "mock-val"])
            assert rc == 0, rc

            # 3. Missing required args -> argparse exit 2.
            _expect_exit([str(valid_src)], expected_code=2)

            # 4. Output collides with input via exact path, non-normalized alias, missing path, and samefile/hardlink.
            orig_bytes = valid_src.read_bytes()
            calls.clear()
            _expect_exit([str(valid_src), str(valid_src)], fragment="separate input assembly and output binary")
            _expect_exit([str(valid_src), str(work / "nested" / ".." / "subtract.spvasm")],
                         fragment="separate input assembly and output binary")
            missing_src = work / "missing.spvasm"
            _expect_exit([str(missing_src), str(work / "nested" / ".." / "missing.spvasm")],
                         fragment="separate input assembly and output binary")
            hardlink = work / "subtract_link.spvasm"
            os.link(valid_src, hardlink)
            assert valid_src.resolve() != hardlink.resolve()
            assert valid_src.samefile(hardlink)
            _expect_exit([str(valid_src), str(hardlink)], fragment="separate input assembly and output binary")
            assert valid_src.read_bytes() == orig_bytes
            assert calls == []

            # 5. Output path is an existing directory: rejected immediately before running assembler/validator.
            out_dir = work / "existing_dir"
            out_dir.mkdir()
            calls.clear()
            _expect_exit([str(valid_src), str(out_dir), "--assembler", "mock-as", "--validator", "mock-val"],
                         fragment="directory")
            assert list(out_dir.iterdir()) == []
            assert calls == []

            # 6. Output parent is an existing file -> OSError caught cleanly.
            file_parent = work / "file_parent"
            file_parent.write_bytes(b"not a dir")
            _expect_exit([str(valid_src), str(file_parent / "out.spv"), "--assembler", "mock-as", "--validator", "mock-val"])

            # 7. Missing source and directory source -> OSError caught cleanly, no output or parent created.
            missing_nested = work / "never_created" / "out.spv"
            _expect_exit([str(missing_src), str(missing_nested)])
            assert not missing_nested.parent.exists()
            _expect_exit([str(out_dir), str(missing_nested)])
            assert not missing_nested.parent.exists()

            # 8. Non-UTF8 source -> UnicodeDecodeError caught cleanly.
            bad_utf8 = work / "bad_utf8.spvasm"
            bad_utf8.write_bytes(b"\xff\xfe\x00\x80")
            _expect_exit([str(bad_utf8), str(missing_nested)])
            assert not missing_nested.parent.exists()

            # 9. Lowering ValueError cases through main(): unhandled physical op, PushConstant, OpLoopMerge.
            target_dir = work / "atomic_guard"
            target_dir.mkdir()
            protected = target_dir / "kept.spv"
            protected.write_bytes(sentinel_bytes)

            unhandled_ptr = work / "unhandled_ptr.spvasm"
            unhandled_ptr.write_text(_sub(SUBTRACT, "OpReturn", "%bad_psb = OpTypePointer PhysicalStorageBuffer %bool\nOpReturn"), encoding="utf-8")
            bad_pc = work / "bad_pc.spvasm"
            bad_pc.write_text(_sub(SUBTRACT, "%main = OpFunction", "%bad_pc = OpTypePointer PushConstant %uint\n%main = OpFunction"), encoding="utf-8")
            loop_merge = work / "loop_merge.spvasm"
            loop_merge.write_text(
                _sub(SIGNED, "OpReturn\nOpFunctionEnd",
                     "OpBranch %hdr\n%hdr = OpLabel\n%bad = OpLoad %int %load_target Aligned 4\n"
                     "OpLoopMerge %lexit %lcont None\nOpBranch %lcont\n%lcont = OpLabel\nOpBranch %hdr\n%lexit = OpLabel\nOpReturn\nOpFunctionEnd"),
                encoding="utf-8",
            )
            for bad_asm, frag in (
                (unhandled_ptr, "unhandled physical pointer operation"),
                (bad_pc, "unrecognized push-constant declaration"),
                (loop_merge, "OpLoopMerge"),
            ):
                _expect_exit([str(bad_asm), str(protected), "--assembler", "mock-as", "--validator", "mock-val"],
                             fragment=frag)
                assert protected.read_bytes() == sentinel_bytes
                assert [p.name for p in target_dir.iterdir()] == ["kept.spv"]

            # 10. Subprocess failures: assembler CalledProcessError, validator CalledProcessError,
            # lowered-stage validator failure, missing tool FileNotFoundError, and TimeoutExpired.
            for call_idx, exc in (
                (1, subprocess.CalledProcessError(1, ["mock-as"])),
                (2, subprocess.CalledProcessError(1, ["mock-val"])),
                (3, subprocess.CalledProcessError(1, ["mock-as"])),
                (4, subprocess.CalledProcessError(1, ["mock-val"])),
                (1, FileNotFoundError("no mock-as")),
                (2, subprocess.TimeoutExpired(["mock-val"], 5)),
            ):
                calls.clear()
                fail_on_call.clear()
                fail_on_call[call_idx] = exc
                _expect_exit([str(valid_src), str(protected), "--assembler", "mock-as", "--validator", "mock-val"])
                assert protected.read_bytes() == sentinel_bytes
                assert [p.name for p in target_dir.iterdir()] == ["kept.spv"]
            fail_on_call.clear()

            # 11. Replace failure preserves existing output and leaves no temp artifacts.
            orig_replace = Path.replace

            def fail_replace(self, target):
                raise OSError("simulated replace failure")

            Path.replace = fail_replace
            try:
                _expect_exit([str(valid_src), str(protected), "--assembler", "mock-as", "--validator", "mock-val"],
                             fragment="simulated replace failure")
            finally:
                Path.replace = orig_replace
            assert protected.read_bytes() == sentinel_bytes
            assert [p.name for p in target_dir.iterdir()] == ["kept.spv"]
        finally:
            saw.subprocess.run = orig_run

        # 12. Real subprocess check of __main__ error exit without traceback.
        sub_err = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(work / "no_such_file.spvasm"), str(work / "out.spv")],
            capture_output=True,
            text=True,
        )
        assert sub_err.returncode == 1, sub_err
        assert sub_err.stderr.strip() and "Traceback" not in sub_err.stderr, sub_err.stderr


def main():
    check_word_pointer()
    check_signed_access()
    check_float_access()
    check_vec4_access()
    check_merge_pointer_types()
    check_lower_positive_and_composed()
    check_lower_rejections()
    check_cli()
    print("spirv_address_window: unit lowering, composed passes, and CLI error handling passed")


if __name__ == "__main__":
    main()
