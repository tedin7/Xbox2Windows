"""Direct Compute DXIL checks for all remaining SOP1 *_saveexec_* and *_wrexec_* ops (#3296, Refs #2935)."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_f16_alu_check import _branch, _exec, _i32, pack, words as pack_words
from native_scalar_lds_check import scalar, sop1, vmove


MOV_B32, MOV_B64 = 3, 4
ZERO, ONE, NEG_ONE, M0, LITERAL = 128, 129, 193, 124, 255
VCC_LO, EXEC_LO, EXEC_HI = 106, 126, 127
MASK32, MASK64 = 0xFFFFFFFF, (1 << 64) - 1
GUARD_WORDS, OUTPUT_WORDS = 8, 2064
GROUP_WORDS, PLANE_WORDS = 128, 256
PLANE_BYTES = PLANE_WORDS * 4
SENTINEL = 0x69696969
END = 0xBF810000
ACTIVE_SENTINEL = 0x11110000

# Remaining 9 wave32 SOP1 ops (0x3e..0x46) and 3 wave64 SOP1 ops (0x37..0x39).
WAVE32_OPS = (
    ("s_xor_saveexec_b32",   0x3E, True),
    ("s_andn2_saveexec_b32", 0x3F, True),
    ("s_orn2_saveexec_b32",  0x40, True),
    ("s_nand_saveexec_b32",  0x41, True),
    ("s_nor_saveexec_b32",   0x42, True),
    ("s_xnor_saveexec_b32",  0x43, True),
    ("s_andn1_saveexec_b32", 0x44, True),
    ("s_orn1_saveexec_b32",  0x45, True),
    ("s_andn1_wrexec_b32",   0x46, False),
)
WAVE64_OPS = (
    ("s_andn1_saveexec_b64", 0x37, True),
    ("s_orn1_saveexec_b64",  0x38, True),
    ("s_andn1_wrexec_b64",   0x39, False),
)


def op_kind(name):
    return name.split("_")[1]


def combine_bits(op, a, b, width):
    mask = (1 << width) - 1
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "andn2":
        return a & (b ^ mask)
    if op == "orn2":
        return (a | (b ^ mask)) & mask
    if op == "andn1":
        return (a ^ mask) & b
    if op == "orn1":
        return ((a ^ mask) | b) & mask
    if op == "nand":
        return (a & b) ^ mask
    if op == "nor":
        return (a | b) ^ mask
    if op == "xnor":
        return (a ^ b) ^ mask
    raise ValueError(op)


def set_scc(value):
    # s_cmp_eq_u32 0, (0 for true, 1 for false)
    return 0xBF060080 | ((ZERO if value else ONE) << 8)


def store(vgpr, plane):
    return [0xE0701000 | (plane * PLANE_BYTES), 0x80080008 | (vgpr << 8)]


def compute_container(code_bytes, wave32=True, threads=None):
    """Build a .cdx container compatible with native_compute_ir_dump (table=s4, group=s5)."""
    if threads is None:
        threads = 32 if wave32 else 64
    body = bytearray(256)
    struct.pack_into("<Q", body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 48, 72):
        struct.pack_into("<Q", body, offset, 0xDEADBEEFDEADBEEF)
    record = 112
    struct.pack_into("<Q", body, 96, record)
    struct.pack_into("<Q", body, record + 40, len(body))
    struct.pack_into("<I", body, record + 56, 1)
    body.extend(pack_words(3, 0, 1, 0, 4))
    struct.pack_into("<QQ", body, record, len(body), len(code_bytes))
    struct.pack_into("<I", body, record + 20, 32 if wave32 else 0)
    struct.pack_into("<II", body, record + 76, 512, 4)
    struct.pack_into("<IIII", body, record + 120, 0x8A, threads, 1, 1)
    body.extend(code_bytes)
    return pack(body)


def request(work, root, wave_size, threads, prog_words):
    z = 1 | (1 << 20 if wave_size == 32 else 0)
    sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), z, 0, 0x8A)
    return write_request(work, prog_words, root, sa)


def wave32_cases():
    # (label, name, opcode, save, threads, old_exec, dst, src_reg, src_val, seed_reg)
    return [
        # Non-zero cases (SCC=1, incoming SCC=0), asymmetric inputs to distinguish every op and operand order.
        ("w32_xor_nz",   "s_xor_saveexec_b32",   0x3E, True,  32, 0xA5A50003, 24,  22,      0x0F0F00F2, None),
        ("w32_andn2_nz", "s_andn2_saveexec_b32", 0x3F, True,  32, 0x0F0F00F2, 24,  22,      0xA5A50003, None),
        ("w32_orn2_nz",  "s_orn2_saveexec_b32",  0x40, True,  32, 0xF0F0FF0C, 24,  22,      0x12340001, None),
        ("w32_nand_nz",  "s_nand_saveexec_b32",  0x41, True,  32, 0xA5A50003, 24,  22,      0xFFFF0002, None),
        ("w32_nor_nz",   "s_nor_saveexec_b32",   0x42, True,  32, 0x0F0F00F2, 24,  22,      0xF0F00004, None),
        ("w32_xnor_nz",  "s_xnor_saveexec_b32",  0x43, True,  32, 0xA5A50003, 24,  22,      0x5A5A00F2, None),
        ("w32_andn1_nz", "s_andn1_saveexec_b32", 0x44, True,  32, 0xA5A50003, 24,  22,      0x0F0F0002, None),
        ("w32_orn1_nz",  "s_orn1_saveexec_b32",  0x45, True,  32, 0x00000011, 24,  22,      0xFFFF00FE, None),
        ("w32_wr1_nz",   "s_andn1_wrexec_b32",   0x46, False, 32, 0xA5A50003, 24,  22,      0x0F0F0002, None),
        # Zero-result cases (SCC=0, incoming SCC=1), covering aliasing, inline, literal, VCC, EXEC_LO, partial wave.
        ("w32_xor_z",    "s_xor_saveexec_b32",   0x3E, True,  32, 0x12345678, 22,  22,      0x12345678, None),  # dst == src
        ("w32_andn2_z",  "s_andn2_saveexec_b32", 0x3F, True,  32, 0xFFFFFFFF, 106, 22,      0x12345678, None),  # dst=vcc_lo
        ("w32_orn2_z",   "s_orn2_saveexec_b32",  0x40, True,  32, 0xFFFFFFFF, 24,  ZERO,    0,          None),  # inline 0
        ("w32_nand_z",   "s_nand_saveexec_b32",  0x41, True,  32, 0xFFFFFFFF, 24,  NEG_ONE, 0xFFFFFFFF, None),  # inline -1
        ("w32_nor_z",    "s_nor_saveexec_b32",   0x42, True,  32, 0xF0F0F0F0, 24,  LITERAL, 0x0F0F0F0F, None),  # literal
        ("w32_xnor_z",   "s_xnor_saveexec_b32",  0x43, True,  32, 0x55AA33CC, 24,  106,     0xAA55CC33, 106),   # src=vcc_lo
        ("w32_andn1_z",  "s_andn1_saveexec_b32", 0x44, True,  8,  0x000000A5, 24,  NEG_ONE, 0xFFFFFFFF, None),  # partial
        ("w32_orn1_z",   "s_orn1_saveexec_b32",  0x45, True,  32, 0x00000000, 24,  NEG_ONE, 0xFFFFFFFF, None),
        ("w32_wr1_z",    "s_andn1_wrexec_b32",   0x46, False, 32, 0x12345679, 22,  22,      0xFFFFFFFF, None),  # dst == src
        ("w32_xor_exec", "s_xor_saveexec_b32",   0x3E, True,  32, 0x000000FF, 24,  EXEC_LO, 0x000000FF, None),  # src=exec_lo
    ]


def wave64_cases():
    # (label, name, opcode, save, threads, old_exec, dst, src_reg, src_val)
    return [
        # Non-zero with bit 0 set (low & high both exercised).
        ("w64_andn1_nz",   "s_andn1_saveexec_b64", 0x37, True,  64,  (0x80000013, 0x40000021), 24, 22,      (0x00000012, 0x40000000)),
        ("w64_orn1_nz",    "s_orn1_saveexec_b64",  0x38, True,  64,  (0x00000001, 0x40000000), 24, 22,      (0xFFFFFF0E, 0xBFFFFF0F)),
        ("w64_wr1_nz",     "s_andn1_wrexec_b64",   0x39, False, 128, (0x80000013, 0x40000021), 22, 22,      (0x00000012, 0x40000000)),  # dst == src
        # High-word-only non-zero (lo == 0, hi != 0 -> verifies SCC combines hi word, bit 0 == 0).
        ("w64_andn1_hi",   "s_andn1_saveexec_b64", 0x37, True,  64,  (0x000000FF, 0x80000010), 24, 22,      (0x000000FF, 0x0000000F)),
        ("w64_orn1_hi",    "s_orn1_saveexec_b64",  0x38, True,  64,  (0x00000000, 0x00000001), 24, 22,      (0xFFFFFFFF, 0xFFFFFFFE)),
        ("w64_wr1_hi",     "s_andn1_wrexec_b64",   0x39, False, 64,  (0x000000FF, 0x80000010), 24, 22,      (0x000000FF, 0x0000000F)),
        # Zero-result cases (SCC=0, incoming SCC=1).
        ("w64_andn1_z",    "s_andn1_saveexec_b64", 0x37, True,  64,  (0x80000012, 0x40000009), 24, NEG_ONE, (MASK32, MASK32)),
        ("w64_orn1_z",     "s_orn1_saveexec_b64",  0x38, True,  64,  (0x00000000, 0x00000000), 24, NEG_ONE, (MASK32, MASK32)),
        ("w64_wr1_z",      "s_andn1_wrexec_b64",   0x39, False, 64,  (0x80000012, 0x40000009), 24, 22,      (MASK32, MASK32)),
    ]


def build_wave32_program(case):
    _, name, opcode, save, _, old_exec, dst, src_reg, src_val, seed_reg = case
    new_exec = combine_bits(op_kind(name), src_val, old_exec, 32)
    dst_val = old_exec if save else new_exec
    scc = int(new_exec != 0)
    # v8 = 512 * group.x + 4 * local.x; v1 = 0x11110000 + local.x
    prog = [
        sop1(MOV_B32, 20, 4), sop1(MOV_B32, 21, ZERO),
        0xF408080A, 0xFA000020,          # s_load_dwordx4 s[32:35], s[20:21], 0x20
        scalar(0x1E, 7, 5, ZERO + 9),    # s_lshl_b32 s7, s5, 9
        vop2(0x1A, 8, ZERO + 2, 0),      # v_lshlrev_b32 v8, 2, v0
        vop2(0x25, 8, 7, 8),             # v_add_nc_u32 v8, s7, v8
        vop2(0x25, 1, LITERAL, 0), ACTIVE_SENTINEL,  # v_add_nc_u32 v1, ACTIVE_SENTINEL, v0
        0xBF8CC07F,                      # s_waitcnt lgkmcnt(0)
        sop1(MOV_B32, EXEC_LO, LITERAL), old_exec,
    ]
    if seed_reg is not None:
        prog += [sop1(MOV_B32, seed_reg, LITERAL), src_val]
    elif src_reg < 106:
        prog += [sop1(MOV_B32, src_reg, LITERAL), src_val]
    prog.append(set_scc(1 - scc))
    prog.append(sop1(opcode, dst, src_reg))
    if src_reg == LITERAL:
        prog.append(src_val)
    prog += [
        *store(1, 0),                    # active-lane write under new EXEC
        scalar(0x0A, 26, ONE, ZERO),     # s_cselect_b32 s26, 1, 0 (captures SCC)
        sop1(MOV_B32, 27, EXEC_LO),      # capture new EXEC
        sop1(MOV_B32, EXEC_LO, NEG_ONE),
    ]
    for plane, reg in ((1, dst), (2, 27), (3, 26)):
        prog += [vmove(2, reg), *store(2, plane)]
    prog.append(END)
    return prog, new_exec, dst_val, scc


def build_wave64_program(case):
    _, name, opcode, save, _, old_exec, dst, src_reg, src_val = case
    old64 = old_exec[0] | (old_exec[1] << 32)
    src64 = src_val[0] | (src_val[1] << 32)
    new64 = combine_bits(op_kind(name), src64, old64, 64)
    new_exec = (new64 & MASK32, new64 >> 32)
    dst_val = old_exec if save else new_exec
    scc = int(new64 != 0)
    prog = [
        sop1(MOV_B64, 126, NEG_ONE),
        sop1(MOV_B32, 20, 4), sop1(MOV_B32, 21, ZERO),
        0xF408080A, 0xFA000020,
        scalar(0x1E, 7, 5, ZERO + 9),
        vop2(0x1A, 8, ZERO + 2, 0),
        vop2(0x25, 8, 7, 8),
        vop2(0x25, 1, LITERAL, 0), ACTIVE_SENTINEL,
        0xBF8CC07F,
        sop1(MOV_B32, EXEC_LO, LITERAL), old_exec[0],
        sop1(MOV_B32, EXEC_HI, LITERAL), old_exec[1],
    ]
    if src_reg < 106:
        prog += [
            sop1(MOV_B32, src_reg, LITERAL), src_val[0],
            sop1(MOV_B32, src_reg + 1, LITERAL), src_val[1],
        ]
    prog += [
        set_scc(1 - scc),
        sop1(opcode, dst, src_reg),
        *store(1, 0),                    # plane 0: active lanes under new EXEC
        scalar(0x0A, 26, ONE, ZERO),     # s26 = SCC
        sop1(MOV_B64, 28, EXEC_LO),      # s[28:29] = new EXEC
        sop1(MOV_B64, EXEC_LO, NEG_ONE),
    ]
    for plane, reg in ((1, dst), (2, dst + 1), (3, 26)):
        prog += [vmove(2, reg), *store(2, plane)]
    # Advance v8 by 4096 so planes 4 and 5 stay within the 12-bit MUBUF offset field.
    prog += [0x4A1010FF, 0x1000]
    for plane, reg in ((0, 28), (1, 29)):
        prog += [vmove(2, reg), *store(2, plane)]
    prog.append(END)
    return prog, new_exec, dst_val, scc


def interpret(ir_text):
    """Evaluate lane 0 of a straight-line DXIL compute module."""
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
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, shared_mem, stores = {}, {}, {}, []
    label, prev_label, steps = "entry", None, 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith("br "):
                prev_label, label = label, _branch(line, values)
                followed = True
                break
            if line == "ret void":
                return sorted(stores)
            if (m := re.fullmatch(r"(%\w+) = call i32 @dx\.op\.waveActiveOp\.i32\(.+\)", line)):
                values[m.group(1)] = ("i32", 0)
            elif line.startswith("call void @dx.op.barrier("):
                continue
            elif (m := re.fullmatch(r"(%\w+) = getelementptr .+, i32 (\S+)$", line)):
                values[m.group(1)] = ("sptr", _i32(m.group(2), values))
            elif (m := re.fullmatch(r"store i32 (\S+), i32 addrspace\(3\)\* (%\w+), align 4", line)):
                shared_mem[values[m.group(2)][1]] = _i32(m.group(1), values)
            elif (m := re.fullmatch(r"(%\w+) = load i32, i32 addrspace\(3\)\* (%\w+), align 4", line)):
                values[m.group(1)] = ("i32", shared_mem.get(values[m.group(2)][1], 0))
            elif (m := re.fullmatch(r"(%\w+) = phi i32 (.+)", line)):
                chosen = None
                for val_tok, pred in re.findall(r"\[([^,\]]+), %([^\]]+)\]", m.group(2)):
                    if pred == prev_label:
                        chosen = _i32(val_tok.strip(), values)
                        break
                assert chosen is not None, (line, prev_label)
                values[m.group(1)] = ("i32", chosen)
            else:
                _exec(line, values, memory, stores)
        if not followed:
            raise AssertionError("block fell through: " + label)
    raise AssertionError("step limit")


def refusal_programs():
    init64 = [sop1(MOV_B64, 22, NEG_ONE), sop1(MOV_B64, VCC_LO, NEG_ONE)]
    return [
        (True,  32, [sop1(0x3E, EXEC_LO, NEG_ONE), END], "unsupported wave32 SAVEEXEC destination"),
        (True,  32, [sop1(0x46, EXEC_LO, NEG_ONE), END], "unsupported wave32 SAVEEXEC destination"),
        (True,  32, [sop1(0x44, M0, NEG_ONE), END],      "unsupported wave32 SAVEEXEC destination"),
        (False, 64, [sop1(0x3E, 24, NEG_ONE), END],      "direct DXIL lowering for s_xor_saveexec_b32"),
        (False, 64, [sop1(0x46, 24, NEG_ONE), END],      "direct DXIL lowering for s_andn1_wrexec_b32"),
        (False, 64, [*init64, sop1(0x39, 24, VCC_LO), END], "unsupported wave64 scalar operation"),
        (False, 64, [*init64, sop1(0x39, 25, 22), END],  "unsupported wave64 scalar operation"),
        (False, 64, [*init64, sop1(0x37, 25, 22), END],  "unsupported wave64 scalar operation"),
    ]


def check_oracles():
    # Ensure mutant references (swapped n1/n2, dropped ~ on nand/nor/xnor, save-vs-wrexec swap) differ cleanly.
    for op in ("andn1", "orn1"):
        for a, b in ((0x0F0F0002, 0xA5A50003),):
            assert combine_bits(op, a, b, 32) != combine_bits(op.replace("n1", "n2"), a, b, 32)
    for op, alt in (("nand", "and"), ("nor", "or"), ("xnor", "xor")):
        assert combine_bits(op, 0x0F0F00F2, 0xA5A50003, 32) != combine_bits(alt, 0x0F0F00F2, 0xA5A50003, 32)


def ir(dumper):
    check_oracles()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for case in wave32_cases():
            label = case[0]
            prog, new_exec, dst_val, scc = build_wave32_program(case)
            path = work / f"{label}.cdx"
            path.write_bytes(compute_container(pack_words(*prog), wave32=True, threads=case[4]))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode == 0, (label, run.stderr)
            stores = dict(interpret(run.stdout))
            if new_exec & 1:
                assert stores.get(0) == ACTIVE_SENTINEL, (label, stores)
            else:
                assert 0 not in stores, (label, stores)
            assert stores.get(1 * PLANE_BYTES) == dst_val, (label, hex(stores.get(PLANE_BYTES, -1)), hex(dst_val))
            assert stores.get(2 * PLANE_BYTES) == new_exec, (label, hex(stores.get(2 * PLANE_BYTES, -1)), hex(new_exec))
            assert stores.get(3 * PLANE_BYTES) == scc, (label, stores.get(3 * PLANE_BYTES), scc)

        for case in wave64_cases():
            label = case[0]
            prog, new_exec, dst_val, scc = build_wave64_program(case)
            path = work / f"{label}.cdx"
            path.write_bytes(compute_container(pack_words(*prog), wave32=False, threads=case[4]))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode == 0, (label, run.stderr)
            stores = dict(interpret(run.stdout))
            if new_exec[0] & 1:
                assert stores.get(0) == ACTIVE_SENTINEL, (label, stores)
            else:
                assert 0 not in stores, (label, stores)
            assert stores.get(1 * PLANE_BYTES) == dst_val[0], (label, stores, dst_val)
            assert stores.get(2 * PLANE_BYTES) == dst_val[1], (label, stores, dst_val)
            assert stores.get(3 * PLANE_BYTES) == scc, (label, stores, scc)
            assert stores.get(4 * PLANE_BYTES) == new_exec[0], (label, stores, new_exec)
            assert stores.get(5 * PLANE_BYTES) == new_exec[1], (label, stores, new_exec)

        for idx, (wave32, threads, bad_prog, err) in enumerate(refusal_programs()):
            path = work / f"refuse{idx}.cdx"
            path.write_bytes(compute_container(pack_words(*bad_prog), wave32=wave32, threads=threads))
            run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
            assert run.returncode != 0 and err in run.stderr, (idx, err, run.returncode, run.stderr)

    print(f"evaluated IR for {len(wave32_cases())} wave32 and {len(wave64_cases())} wave64 cases, "
          f"plus {len(refusal_programs())} refusals")
    return 0


def controls(cli, sdk, work, root):
    check_oracles()
    cache = work / "cache"
    compiled = 0
    for wave_size, builder, items in ((32, build_wave32_program, wave32_cases()),
                                      (64, build_wave64_program, wave64_cases())):
        for case in items:
            label = case[0]
            prog, _, _, _ = builder(case)
            marker = request(work, root, wave_size, case[4], prog)
            cso = marker.with_suffix(".cso")
            cso.unlink(missing_ok=True)
            res = subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", marker, cso],
                                 capture_output=True, text=True)
            assert res.returncode == 0 and cso.is_file(), (label, res.stdout, res.stderr)
            compiled += 1

    refusals = refusal_programs()
    for idx, (wave32, threads, bad_prog, err) in enumerate(refusals):
        marker = request(work, root, 32 if wave32 else 64, threads, bad_prog)
        cso = marker.with_suffix(".cso")
        cso.unlink(missing_ok=True)
        res = subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", marker, cso],
                             capture_output=True, text=True)
        assert res.returncode != 0 and err in res.stderr, (idx, err, res.stdout, res.stderr)
        assert not cso.exists(), cso

    print(f"compiled {compiled} wave32/wave64 cases cleanly and verified {len(refusals)} refusals")
    return 0


def hardware(replay, sdk, work, root):
    cache = work / "cache"
    for case in wave32_cases():
        label, _, _, _, threads, _, _, _, _, _ = case
        prog, new_exec, dst_val, scc = build_wave32_program(case)
        marker = request(work, root, 32, threads, prog)
        out = work / f"{label}.bin"
        run = subprocess.run([replay, sdk, marker, cache, "0", out, "--direct"])
        if run.returncode == 77:
            return 77
        run.check_returncode()
        actual = list(struct.unpack(f"<{OUTPUT_WORDS}I", out.read_bytes()))
        expected = [SENTINEL] * OUTPUT_WORDS
        for group in range(2):
            for lane in range(threads):
                pos = GUARD_WORDS + group * GROUP_WORDS + lane
                if new_exec & (1 << lane):
                    expected[pos] = ACTIVE_SENTINEL | lane
                expected[pos + 1 * PLANE_WORDS] = dst_val
                expected[pos + 2 * PLANE_WORDS] = new_exec
                expected[pos + 3 * PLANE_WORDS] = scc
        assert actual == expected, (label, [
            (i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected)) if a != b
        ][:10])

    for case in wave64_cases():
        label, _, _, _, threads, _, _, _, _ = case
        prog, new_exec, dst_val, scc = build_wave64_program(case)
        marker = request(work, root, 64, threads, prog)
        out = work / f"{label}.bin"
        run = subprocess.run([replay, sdk, marker, cache, "0", out, "--direct"])
        if run.returncode == 77:
            return 77
        run.check_returncode()
        actual = list(struct.unpack(f"<{OUTPUT_WORDS}I", out.read_bytes()))
        expected = [SENTINEL] * OUTPUT_WORDS
        for group in range(2):
            for local in range(threads):
                lane = local % 64
                pos = GUARD_WORDS + group * GROUP_WORDS + local
                if new_exec[lane // 32] & (1 << (lane % 32)):
                    expected[pos] = ACTIVE_SENTINEL | local
                expected[pos + 1 * PLANE_WORDS] = dst_val[0]
                expected[pos + 2 * PLANE_WORDS] = dst_val[1]
                expected[pos + 3 * PLANE_WORDS] = scc
                expected[pos + 4 * PLANE_WORDS] = new_exec[0]
                expected[pos + 5 * PLANE_WORDS] = new_exec[1]
        assert actual == expected, (label, [
            (i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected)) if a != b
        ][:10])

    print(f"replayed {len(wave32_cases())} wave32 and {len(wave64_cases())} wave64 cases on GPU with guards")
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
