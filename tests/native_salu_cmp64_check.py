"""Direct compute 64-bit SOPC comparison and bit-test checks."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_direct_container_check import words as pack_words
from native_f16_alu_check import _branch, _exec, _i32, compute_container
from native_salu64_check import (
    GUARD_WORDS, LANES, LITERAL, LITERAL_VALUE, MASK32, NEG_ONE, OUTPUT_WORDS,
    PLANE_BYTES, SENTINEL, ZERO, compute_program, set_pair, set_scc,
)
from native_scalar_lds_check import sop1


S_BITCMP0_B64, S_BITCMP1_B64, S_CMP_EQ_U64, S_CMP_LG_U64 = 0x0E, 0x0F, 0x12, 0x13
NULL = 125
VCC_LO, EXEC_LO, INLINE_FLOAT_ONE = 106, 126, 242
PAIR = (0x89ABCDEF, 0x01234567)


def sopc(op, a, b):
    return 0xBF000000 | (op << 16) | (b << 8) | a


def _src64(kind, spec):
    if kind == "pair":
        return spec
    if kind == "null":
        return (0, 0)
    if kind == "inline":
        val = (spec - 128) if spec <= 192 else (192 - spec)
        return (val & MASK32, MASK32 if spec >= 193 else 0)
    return (spec, 0)


def cases():
    # (label, op, s0_kind, s0_spec, s1_kind, s1_spec, reg_pair)
    cmp_specs = [
        ("eq equal pairs", S_CMP_EQ_U64, "pair", PAIR, "pair", PAIR),
        ("eq hi differs", S_CMP_EQ_U64, "pair", PAIR, "pair", (PAIR[0], PAIR[1] ^ 1)),
        ("eq lo differs", S_CMP_EQ_U64, "pair", PAIR, "pair", (PAIR[0] ^ 1, PAIR[1])),
        ("lg hi differs", S_CMP_LG_U64, "pair", PAIR, "pair", (PAIR[0], PAIR[1] ^ 0x80000000)),
        ("lg equal pairs", S_CMP_LG_U64, "pair", PAIR, "pair", PAIR),
        ("eq -1 vs all ones", S_CMP_EQ_U64, "pair", (MASK32, MASK32), "inline", NEG_ONE),
        ("eq -1 vs lo-only 0xffffffff", S_CMP_EQ_U64, "pair", (MASK32, 0), "inline", NEG_ONE),
        ("eq lit 0x80000001 vs zext", S_CMP_EQ_U64, "pair", (LITERAL_VALUE, 0), "lit", LITERAL_VALUE),
        ("lg lit 0x80000001 vs sext", S_CMP_LG_U64, "pair", (LITERAL_VALUE, MASK32), "lit", LITERAL_VALUE),
        ("eq null vs (0,0)", S_CMP_EQ_U64, "null", 0, "pair", (0, 0)),
        ("eq null vs (0,1)", S_CMP_EQ_U64, "null", 0, "pair", (0, 1)),
    ]
    for label, op, k0, s0, k1, s1 in cmp_specs:
        code = []
        a_enc = 22
        b_enc = 24
        if k0 == "pair":
            code += set_pair(s0)
        elif k0 == "null":
            a_enc = NULL
        elif k0 == "inline":
            a_enc = s0
        else:
            a_enc = LITERAL
        if k1 == "pair":
            # Also preload s[22:23] with PAIR so compute_program(dst=22) verifies SOPC preserves SGPRs.
            if k0 != "pair":
                code += set_pair(PAIR)
            code += [sop1(3, 24, LITERAL), s1[0], sop1(3, 25, LITERAL), s1[1]]
        elif k1 == "inline":
            b_enc = s1
        elif k1 == "null":
            b_enc = NULL
        else:
            b_enc = LITERAL
        expected_pair = s0 if k0 == "pair" else PAIR
        if k0 != "pair" and k1 != "pair":
            code += set_pair(PAIR)
        v0, v1 = _src64(k0, s0), _src64(k1, s1)
        scc = int(v0 == v1) if op == S_CMP_EQ_U64 else int(v0 != v1)
        inst = [sopc(op, a_enc, b_enc)]
        if a_enc == LITERAL:
            inst.append(s0)
        if b_enc == LITERAL:
            inst.append(s1)
        code += [set_scc(1 - scc)] + inst
        yield label, code, 22, expected_pair, scc, (op, k0, v0, k1, v1)

    bit_specs = [
        ("bitcmp1 bit 0 set", S_BITCMP1_B64, "pair", (1, 0), "imm", 0),
        ("bitcmp1 bit 31 set", S_BITCMP1_B64, "pair", (0x80000000, 0), "imm", 31),
        ("bitcmp1 bit 32 set", S_BITCMP1_B64, "pair", (0, 1), "imm", 32),
        ("bitcmp0 bit 32 clear", S_BITCMP0_B64, "pair", (1, 0), "imm", 32),
        ("bitcmp1 bit 63 set", S_BITCMP1_B64, "pair", (0, 0x80000000), "imm", 63),
        ("bitcmp1 wrapped bit 65 (odd sgpr)", S_BITCMP1_B64, "pair", (2, 0), "sgpr27", 65),
        ("bitcmp0 wrapped bit 96 (sgpr)", S_BITCMP0_B64, "pair", (0, 0), "sgpr27", 96),
        ("bitcmp1 -1 bit 63", S_BITCMP1_B64, "inline", NEG_ONE, "imm", 63),
        ("bitcmp1 lit 0x80000001 bit 31", S_BITCMP1_B64, "lit", LITERAL_VALUE, "imm", 31),
        ("bitcmp0 lit 0x80000001 bit 32", S_BITCMP0_B64, "lit", LITERAL_VALUE, "imm", 32),
        ("bitcmp0 null bit 0", S_BITCMP0_B64, "null", 0, "imm", 0),
    ]
    for label, op, k0, s0, k1, bit_idx in bit_specs:
        code = set_pair(s0 if k0 == "pair" else PAIR)
        expected_pair = s0 if k0 == "pair" else PAIR
        if k0 == "pair":
            a_enc = 22
        elif k0 == "inline":
            a_enc = s0
        elif k0 == "null":
            a_enc = NULL
        else:
            a_enc = LITERAL
        if k1 == "imm":
            b_enc = ZERO + bit_idx
        else:
            code += [sop1(3, 27, LITERAL), bit_idx]
            b_enc = 27
        v0 = _src64(k0, s0)
        val64 = v0[0] | (v0[1] << 32)
        bit = (val64 >> (bit_idx & 63)) & 1
        scc = int(bit == 0) if op == S_BITCMP0_B64 else int(bit == 1)
        inst = [sopc(op, a_enc, b_enc)]
        if a_enc == LITERAL:
            inst.append(s0)
        code += [set_scc(1 - scc)] + inst
        yield label, code, 22, expected_pair, scc, (op, k0, v0, "bit", bit_idx)


def _mutant_scc(mutant, meta):
    op, k0, v0, k1, v1 = meta
    if op in (S_CMP_EQ_U64, S_CMP_LG_U64):
        a0, a1 = v0
        b0, b1 = v1
        if mutant == "zero_ext_neg_inline":
            if k0 == "inline":
                a1 = 0
            if k1 == "inline":
                b1 = 0
        elif mutant == "sign_ext_lit":
            if k0 == "lit" and (a0 & 0x80000000):
                a1 = MASK32
            if k1 == "lit" and (b0 & 0x80000000):
                b1 = MASK32
        if mutant == "low_word_only_cmp":
            eq = (a0 == b0)
        elif mutant == "high_word_only_cmp":
            eq = (a1 == b1)
        else:
            eq = (a0, a1) == (b0, b1)
        return int(eq) if op == S_CMP_EQ_U64 else int(not eq)
    # bitcmp
    a0, a1 = v0
    idx = v1
    if mutant == "sign_ext_lit" and k0 == "lit" and (a0 & 0x80000000):
        a1 = MASK32
    val64 = a0 | (a1 << 32)
    eff = (idx & 31) if mutant == "ignore_bit5_bitcmp" else (idx & 63)
    bit = (val64 >> eff) & 1
    if mutant == "inverted_bitcmp":
        return int(bit == 1) if op == S_BITCMP0_B64 else int(bit == 0)
    return int(bit == 0) if op == S_BITCMP0_B64 else int(bit == 1)


def refused_programs():
    init = set_pair((0, 0)) + [sop1(3, 24, ZERO), sop1(3, 25, ZERO)]
    return [
        (init + [sopc(S_CMP_EQ_U64, 23, 24), 0xBF810000], "unaligned scalar mask pair"),
        (init + [sopc(S_CMP_EQ_U64, VCC_LO, 24), 0xBF810000], "invalid s_cmp_eq_u64 source"),
        (init + [sopc(S_CMP_LG_U64, 22, EXEC_LO), 0xBF810000], "invalid s_cmp_lg_u64 source"),
        (init + [sopc(S_CMP_EQ_U64, 22, INLINE_FLOAT_ONE), 0xBF810000], "unsupported wave64 scalar pair"),
        (init + [sopc(S_BITCMP0_B64, 23, ZERO), 0xBF810000], "unaligned scalar mask pair"),
        (init + [sopc(S_BITCMP1_B64, 22, INLINE_FLOAT_ONE), 0xBF810000], "invalid s_bitcmp1_b64 bit index"),
        (init + [sopc(S_BITCMP0_B64, 22, VCC_LO), 0xBF810000], "invalid s_bitcmp0_b64 bit index"),
    ]


def interpret_pair_stores(ir_text):
    """Lane 0 of a straight-line compute module; returns sorted (byte address, lo, hi) stores."""
    body = ir_text.split("define void @main()", 1)[1].split("\n}", 1)[0]
    blocks, label, buf = {}, None, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == "{":
            continue
        if line.endswith(":") and " " not in line:
            if label is not None:
                blocks[label] = buf
            label, buf = line[:-1], []
            continue
        if label is None:
            raise AssertionError(line)
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, raw_stores = {}, {}, []
    label, steps = "entry", 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith("br "):
                label = _branch(line, values)
                followed = True
                break
            if line == "ret void":
                return sorted(raw_stores)
            if "rawBufferStore.i32" in line:
                match = re.search(
                    r"rawBufferStore\.i32\(i32 140, %dx\.types\.Handle \S+, i32 (\S+), i32 \S+, i32 (\S+), i32 (\S+),",
                    line,
                )
                assert match, line
                raw_stores.append((
                    _i32(match.group(1), values),
                    _i32(match.group(2), values),
                    _i32(match.group(3), values),
                ))
            else:
                _exec(line, values, memory, [])
        if not followed:
            raise AssertionError("block fell through: " + label)
    raise AssertionError("step limit")


def check_mutants():
    expected = [(label, scc, meta) for label, _, _, _, scc, meta in cases()]
    for mutant in ("low_word_only_cmp", "high_word_only_cmp", "zero_ext_neg_inline",
                   "sign_ext_lit", "ignore_bit5_bitcmp", "inverted_bitcmp"):
        mismatches = [label for label, scc, meta in expected if _mutant_scc(mutant, meta) != scc]
        assert mismatches, f"mutant {mutant} agreed on every case"
        print(f"mutant {mutant} caught on {len(mismatches)} case(s)")


def ir(dumper):
    all_cases = list(cases())
    refusals = refused_programs()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for index, (label, code, dst, expected_pair, scc, _) in enumerate(all_cases):
            path = work / f"case{index}.cdx"
            path.write_bytes(compute_container(pack_words(*compute_program(code, dst))))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode == 0, (label, run.stderr)
            text = run.stdout
            for forbidden in ("icmp eq i64", "icmp ne i64", "lshr i64", "shl i64"):
                assert forbidden not in text, f"unexpected 64-bit IR op {forbidden} in {label}:\n{text}"
            stores = interpret_pair_stores(text)
            assert len(stores) == 2 and stores[0][0] == 0 and stores[1][0] == PLANE_BYTES, (label, stores)
            assert (stores[0][1], stores[0][2]) == expected_pair, (label, stores[0], expected_pair)
            assert (stores[1][1], stores[1][2]) == (scc, 0), (label, stores[1], scc)

        for index, (prog_words, expected_error) in enumerate(refusals):
            path = work / f"refused{index}.cdx"
            path.write_bytes(compute_container(pack_words(*prog_words)))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode != 0 and expected_error in run.stderr, (expected_error, run.returncode, run.stderr)

    check_mutants()
    print(f"evaluated DXIL IR for {len(all_cases)} 64-bit SOPC cases (pure 32-bit IR) and verified {len(refusals)} refusals")
    return 0


def controls(cli, sdk, work, root):
    # 1. Compile all cases together and check pure 32-bit DXIL IR (no i64 instructions).
    combined = []
    for _, code, _, _, _, _ in cases():
        combined += code
    marker = request(work, root, compute_program(combined, 22))
    cso = work / "combined.cso"
    subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, cso], check=True)
    dis = subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", cso],
                         capture_output=True, text=True, check=True).stdout
    for forbidden in ("icmp eq i64", "icmp ne i64", "lshr i64", "shl i64"):
        assert forbidden not in dis, f"unexpected 64-bit IR op {forbidden} in:\n{dis}"
    for required in ("icmp eq i32", "icmp ne i32", "lshr i32"):
        assert required in dis, f"missing {required} in DXIL disassembly"
    print("compiled all 22 64-bit SOPC cases cleanly with pure 32-bit DXIL IR")

    # 2. Negative operand refusal controls (initialize s[22:25] so only the invalid operand triggers refusal).
    refusals = refused_programs()
    for prog_words, expected_error in refusals:
        bad = request(work, root, prog_words)
        bad_cso = work / "bad.cso"
        bad_cso.unlink(missing_ok=True)
        res = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", bad, bad_cso],
                             capture_output=True, text=True)
        assert res.returncode and expected_error in res.stderr, (prog_words, res.stdout, res.stderr)
        assert not bad_cso.exists()
        print(f"refused: {expected_error}")

    # 3. Mutant controls over expected SCC oracle.
    check_mutants()
    return 0


def hardware(replay, sdk, work, root):
    for label, code, dst, expected_pair, scc, _ in cases():
        marker = request(work, root, compute_program(code, dst))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, label
        assert actual[GUARD_WORDS:end] == expected_pair * LANES + (scc, 0) * LANES, (label, expected_pair, scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), label
        print(f"{label}: preserved pair, SCC and guards matched")
    return 0


def main(mode, tool, sdk=None):
    if mode == "ir":
        return ir(tool)
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
