"""Direct compute s_bfe_u64, s_bfe_i64 and s_brev_b64 checks (#3086)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_salu64_check import GUARD_WORDS, LANES, LITERAL, LITERAL_VALUE, MASK32, NEG_ONE, OUTPUT_WORDS, SENTINEL, ZERO, compute_program, set_pair, set_scc
from native_scalar_lds_check import scalar, sop1


S_BREV_B64 = 0x0C
S_BFE_U64, S_BFE_I64 = 0x29, 0x2A
NULL, VCC_LO, EXEC_LO, INLINE_FLOAT_ONE = 125, 106, 126, 242
PAIR = (0x89ABCDEF, 0x01234567)
MASK64 = (1 << 64) - 1


def rev32(w):
    return int(f"{w & MASK32:032b}"[::-1], 2)


def bfe64_oracle(x, ctrl, signed):
    off = ctrl & 63
    wd = min((ctrl >> 16) & 127, 64 - off)
    field = (x >> off) & ((1 << wd) - 1)
    if signed and 0 < wd < 64 and ((field >> (wd - 1)) & 1):
        field |= MASK64 ^ ((1 << wd) - 1)
    return 0 if (signed and wd == 0) else field


def _src64(kind, spec, signed_lit=False):
    if kind == "pair":
        return spec
    if kind == "null":
        return (0, 0)
    if kind == "inline":
        val = (spec - 128) if spec <= 192 else (192 - spec)
        return (val & MASK32, MASK32 if spec >= 193 else 0)
    return (spec, MASK32 if (signed_lit and (spec & 0x80000000)) else 0)


def cases():
    # s_brev_b64: preserves SCC and swaps+reverses 32-bit halves.
    for label, kind, spec, scc in (
        ("brev pair scc=0", "pair", PAIR, 0),
        ("brev pair scc=1", "pair", PAIR, 1),
        ("brev bit0 -> bit63", "pair", (1, 0), 1),
        ("brev -1", "inline", NEG_ONE, 0),
        ("brev literal 0x80000001", "lit", LITERAL_VALUE, 1),
        ("brev null", "null", 0, 1),
    ):
        s0_words = _src64(kind, spec)
        expected = (rev32(s0_words[1]), rev32(s0_words[0]))
        dst = 22 if label == "brev bit0 -> bit63" else 24  # verify in-place s[22:23] on one case
        code = []
        if kind == "pair":
            code += set_pair(spec)
            a_enc = 22
        elif kind == "inline":
            a_enc = spec
        elif kind == "null":
            a_enc = NULL
        else:
            a_enc = LITERAL
        code.append(set_scc(scc))
        code.append(sop1(S_BREV_B64, dst, a_enc))
        if a_enc == LITERAL:
            code.append(spec)
        yield label, code, dst, expected, scc, ("brev", kind, s0_words, 0)

    # s_bfe_u64 and s_bfe_i64: ctrl = (width << 16) | offset
    bfe_specs = [
        ("bfe_u64 width=0", S_BFE_U64, "pair", PAIR, 0x00000004, 1),
        ("bfe_i64 width=0", S_BFE_I64, "pair", PAIR, 0x00000024, 1),
        ("bfe_u64 full 64", S_BFE_U64, "pair", PAIR, 0x00400000, 0),
        ("bfe_i64 width=127 clamp", S_BFE_I64, "pair", (0x12345678, 0x80000001), 0x007F0000, 0),
        ("bfe_u64 cross-word off=28 wd=8", S_BFE_U64, "pair", (0xA0000000, 0x0000000B), (8 << 16) | 28, 0),
        ("bfe_i64 cross-word neg off=28 wd=8", S_BFE_I64, "pair", (0x80000000, 0x00000008), (8 << 16) | 28, 0),
        ("bfe_i64 past-MSB neg off=40 wd=32", S_BFE_I64, "pair", (0, 0x80000000), (32 << 16) | 40, 0),
        ("bfe_i64 past-MSB pos off=40 wd=32", S_BFE_I64, "pair", (0, 0x00800000), (32 << 16) | 40, 0),
        ("bfe_u64 hi-only nonzero scc", S_BFE_U64, "pair", (0x00000000, 0x00000010), (40 << 16) | 0, 0),
        ("bfe_u64 zero extract clears scc", S_BFE_U64, "pair", (0xFFFFFFFF, 0xFFFFFFFF), (0 << 16) | 12, 1),
        ("bfe_i64 -1 off=32 wd=8", S_BFE_I64, "inline", NEG_ONE, (8 << 16) | 32, 0),
        ("bfe_u64 lit off=32 wd=8 (zext=0)", S_BFE_U64, "lit", LITERAL_VALUE, (8 << 16) | 32, 1),
        ("bfe_i64 lit off=32 wd=8 (sext=-1)", S_BFE_I64, "lit", LITERAL_VALUE, (8 << 16) | 32, 0),
        ("bfe_u64 null", S_BFE_U64, "null", 0, (16 << 16) | 8, 1),
    ]
    for label, op, kind, spec, ctrl, init_scc in bfe_specs:
        signed = op == S_BFE_I64
        s0_words = _src64(kind, spec, signed_lit=signed)
        val64 = s0_words[0] | (s0_words[1] << 32)
        out64 = bfe64_oracle(val64, ctrl, signed)
        expected = (out64 & MASK32, (out64 >> 32) & MASK32)
        scc = int(out64 != 0)
        code = [sop1(3, 26, LITERAL), ctrl]
        if kind == "pair":
            code += set_pair(spec)
            a_enc = 22
        elif kind == "inline":
            a_enc = spec
        elif kind == "null":
            a_enc = NULL
        else:
            a_enc = LITERAL
        code.append(set_scc(init_scc))
        code.append(scalar(op, 24, a_enc, 26))
        if a_enc == LITERAL:
            code.append(spec)
        yield label, code, 24, expected, scc, (op, kind, s0_words, ctrl)

    # Overlapping destination+ctrl+source: s_bfe_u64 s[22:23], s[22:23], s22
    ctrl_overlap = (12 << 16) | 16
    pair_overlap = (ctrl_overlap, 0x00AB0000)
    out_overlap = bfe64_oracle(pair_overlap[0] | (pair_overlap[1] << 32), ctrl_overlap, False)
    yield ("bfe_u64 overlapping dst/s0/s1",
           set_pair(pair_overlap) + [set_scc(0), scalar(S_BFE_U64, 22, 22, 22)],
           22, (out_overlap & MASK32, (out_overlap >> 32) & MASK32), int(out_overlap != 0),
           (S_BFE_U64, "pair", pair_overlap, ctrl_overlap))


def _mutant_result(mutant, meta):
    op, kind, s0_words, ctrl = meta
    if op == "brev":
        if mutant == "brev_no_word_swap":
            return ((rev32(s0_words[0]), rev32(s0_words[1])), None)
        return ((rev32(s0_words[1]), rev32(s0_words[0])), None)
    signed = op == S_BFE_I64
    if mutant == "bfe_unsigned_for_signed":
        signed = False
    if mutant == "bfe_no_sign_ext_lit_i64" and op == S_BFE_I64 and kind == "lit":
        s0_words = (s0_words[0], 0)
    eff_ctrl = ctrl
    if mutant == "bfe_ignore_offset_bit5":
        eff_ctrl = (ctrl & ~0x3F) | (ctrl & 0x1F)
    val64 = s0_words[0] | (s0_words[1] << 32)
    wd = (eff_ctrl >> 16) & 127
    if mutant == "bfe_width_zero_passthrough" and wd == 0:
        out64 = val64
    else:
        out64 = bfe64_oracle(val64, eff_ctrl, signed)
    out_words = (out64 & MASK32, (out64 >> 32) & MASK32)
    scc = int(out_words[0] != 0) if mutant == "bfe_scc_low_word_only" else int(out64 != 0)
    return (out_words, scc)


def _compile_and_dis(cli, sdk, work, root, code, name):
    marker = request(work, root, code)
    cso = work / f"{name}.cso"
    cso.unlink(missing_ok=True)
    res = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, cso],
                         capture_output=True, text=True)
    return res, cso


def controls(cli, sdk, work, root):
    # 1. Standalone s_brev_b64 uses unary.i32(30) and zero i64 ops.
    brev_code = set_pair(PAIR) + [sop1(S_BREV_B64, 24, 22)]
    res, cso = _compile_and_dis(cli, sdk, work, root, compute_program(brev_code, 24), "brev_only")
    assert res.returncode == 0 and cso.is_file(), (res.stdout, res.stderr)
    dis_brev = subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", cso],
                              capture_output=True, text=True, check=True).stdout
    assert "@dx.op.unary.i32(i32 30," in dis_brev, dis_brev
    for forbidden in ("64-bit integer", "zext i32", "shl i64", "lshr i64", "ashr i64", "trunc i64"):
        assert forbidden not in dis_brev, (forbidden, dis_brev)

    # 2. Combined program with all cases compiles and emits expected DXIL ops.
    combined = []
    for _, code, _, _, _, _ in cases():
        combined += code
    res, cso_all = _compile_and_dis(cli, sdk, work, root, compute_program(combined, 24), "all_cases")
    assert res.returncode == 0 and cso_all.is_file(), (res.stdout, res.stderr)
    dis_all = subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", cso_all],
                             capture_output=True, text=True, check=True).stdout
    for required in ("@dx.op.unary.i32(i32 30,", "lshr i64", "ashr i64", "shl i64"):
        assert required in dis_all, f"missing {required} in combined DXIL disassembly"
    print("compiled s_brev_b64 (pure 32-bit DXIL) and all 21 s_brev/s_bfe cases cleanly")

    # 3. Symbolic transfer() _u64 check: s_bfe_u64 overwriting descriptor s[32:33] must clear s33 too.
    overwrite_desc = (set_pair(PAIR) +
                      [sop1(3, 26, ZERO + 16),
                       scalar(S_BFE_U64, 32, 22, 26),
                       0xE0741000, 0x80080208,  # buffer_store_dwordx2 using modified s[32:35]
                       0xBF810000])
    res_desc, cso_desc = _compile_and_dis(cli, sdk, work, root, overwrite_desc, "overwrite_desc")
    assert res_desc.returncode != 0 and not cso_desc.exists(), (res_desc.stdout, res_desc.stderr)
    print(f"symbolic transfer _u64 descriptor clobber refused: {res_desc.stderr.strip()}")

    # 4. Negative operand/destination refusal controls (with s[22:27] initialized).
    init = set_pair((0, 0)) + [sop1(3, 24, ZERO), sop1(3, 25, ZERO), sop1(3, 26, ZERO), sop1(3, 27, ZERO)]
    refusals = [
        (init + [sop1(S_BREV_B64, 23, 22), 0xBF810000], "invalid s_brev_b64 destination"),
        (init + [sop1(S_BREV_B64, VCC_LO, 22), 0xBF810000], "invalid s_brev_b64 destination"),
        (init + [sop1(S_BREV_B64, EXEC_LO, 22), 0xBF810000], "invalid s_brev_b64 destination"),
        (init + [sop1(S_BREV_B64, 24, 23), 0xBF810000], "invalid s_brev_b64 source"),
        (init + [sop1(S_BREV_B64, 24, INLINE_FLOAT_ONE), 0xBF810000], "invalid s_brev_b64 source"),
        (init + [scalar(S_BFE_U64, VCC_LO, 22, 26), 0xBF810000], "invalid s_bfe_u64 destination"),
        (init + [scalar(S_BFE_U64, 24, 23, 26), 0xBF810000], "invalid s_bfe_u64 source"),
        (init + [scalar(S_BFE_U64, 24, 22, INLINE_FLOAT_ONE), 0xBF810000], "invalid s_bfe_u64 source"),
        (init + [scalar(S_BFE_I64, 24, EXEC_LO, 26), 0xBF810000], "invalid s_bfe_i64 source"),
    ]
    for idx, (words, err) in enumerate(refusals):
        r, bad_cso = _compile_and_dis(cli, sdk, work, root, words, f"bad_{idx}")
        assert r.returncode and err in r.stderr and not bad_cso.exists(), (idx, err, r.stdout, r.stderr)
        print(f"refused: {err}")

    # 5. Oracle mutant checks.
    expected = [(label, words, scc, meta) for label, _, _, words, scc, meta in cases()]
    for mutant in ("brev_no_word_swap", "bfe_unsigned_for_signed", "bfe_width_zero_passthrough",
                   "bfe_no_sign_ext_lit_i64", "bfe_scc_low_word_only", "bfe_ignore_offset_bit5"):
        caught = 0
        for label, words, scc, meta in expected:
            m_words, m_scc = _mutant_result(mutant, meta)
            if m_words != words or (m_scc is not None and m_scc != scc):
                caught += 1
        assert caught > 0, f"mutant {mutant} agreed on every case"
        print(f"mutant {mutant} caught on {caught} case(s)")
    return 0


def hardware(replay, sdk, work, root):
    for label, code, dst, words, scc, _ in cases():
        marker = request(work, root, compute_program(code, dst))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, label
        assert actual[GUARD_WORDS:end] == words * LANES + (scc, 0) * LANES, (label, words, scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), label
        print(f"{label}: result words, SCC and guards matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
