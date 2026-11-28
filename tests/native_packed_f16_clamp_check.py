"""Direct DXIL VOP3P clamp on v_pk_add_f16, v_pk_mul_f16, and v_pk_fma_f16 in Compute and Pixel shaders."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from f16_packed_model import half_op
from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE, inline_word, vop3p
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

ALL_EXEC = 0xFFFFFFFF
DST_INIT = 0xDEADCAFE


def clamp_f16(bits):
    bits &= 0xFFFF
    if (bits & 0x7FFF) > 0x7C00 or (bits & 0x8000):
        return 0x0000
    return 0x3C00 if bits > 0x3BFF else bits


def eval_clamp_case(op_name, raw_words, op_sel, op_sel_hi, neg_lo, neg_hi, clamp):
    arity = 3 if op_name == 'fma' else 2

    def lane_eval(high):
        sel = op_sel_hi if high else op_sel
        neg = neg_hi if high else neg_lo
        ops = []
        for j in range(arity):
            half = (raw_words[j] >> (16 if ((sel >> j) & 1) else 0)) & 0xFFFF
            if (neg >> j) & 1:
                half ^= 0x8000
            ops.append(half)
        res = half_op(op_name, *ops)
        return clamp_f16(res) if clamp else res

    return lane_eval(False) | (lane_eval(True) << 16)


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    srcs: tuple
    vals: tuple
    op_sel: int = 0
    op_sel_hi: int = 7
    neg_lo: int = 0
    neg_hi: int = 0
    clamp: int = 1
    exec_mask: int = ALL_EXEC

    def raw_words(self):
        return tuple(v & 0xFFFFFFFF if (s < 128 or s >= 255) else inline_word(s)
                     for s, v in zip(self.srcs, self.vals))

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return DST_INIT
        return eval_clamp_case(self.op, self.raw_words(), self.op_sel, self.op_sel_hi,
                               self.neg_lo, self.neg_hi, self.clamp)


def all_cases():
    return [
        # 1. v_pk_add_f16: negative finite, -0.0, subnormals, interior (0,1), >= 1.0, +/-Inf, NaN, and unclamped contrast.
        Case('add_neg_and_over_one', 'add', (257, 258), ((0x4000 << 16) | 0xBC00, (0x3800 << 16) | 0xB800)),
        Case('add_neg_zero_and_lt_one', 'add', (257, 258), ((0x3BFF << 16) | 0x8000, (0x0000 << 16) | 0x8000)),
        Case('add_subnormals_exact_one', 'add', (257, 258), ((0x3800 << 16) | 0x0001, (0x3800 << 16) | 0x0000)),
        Case('add_pos_inf_and_nan', 'add', (257, 258), ((0x7C00 << 16) | 0x7E01, (0x3C00 << 16) | 0x3C00)),
        Case('add_neg_inf_and_inf_minus_inf', 'add', (257, 258), ((0xFC00 << 16) | 0x7C00, (0x3C00 << 16) | 0xFC00)),
        Case('add_unclamped_contrast', 'add', (257, 258), ((0x7C00 << 16) | 0xBC00, (0x3C00 << 16) | 0xB800), clamp=0),
        # 2. v_pk_mul_f16: subnormal normal-product, -0.0, interior, overflow/+Inf -> 1.0, -Inf/Inf*0 NaN -> +0.0.
        Case('mul_subnormal_and_neg_zero', 'mul', (257, 258), ((0x0400 << 16) | 0x8000, (0x3800 << 16) | 0x3C00)),
        Case('mul_overflow_and_interior', 'mul', (257, 258), ((0x7BFF << 16) | 0x3800, (0x4000 << 16) | 0x3800)),
        Case('mul_neg_inf_and_inf_times_zero', 'mul', (257, 258), ((0xFC00 << 16) | 0x7C00, (0x3C00 << 16) | 0x0000)),
        Case('mul_neg_finite_and_max_subnormal', 'mul', (257, 258), ((0xBC00 << 16) | 0x03FF, (0x3800 << 16) | 0x3C00)),
        # 3. v_pk_fma_f16: negative/-0, subnormals, exact 1.0, 0x3BFF edge, overflow avoidance, +/-Inf, Inf*0 NaN.
        Case('fma_neg_and_over_one', 'fma', (257, 258, 259),
             ((0x4000 << 16) | 0xBC00, (0x4000 << 16) | 0x3C00, (0x0000 << 16) | 0x3800)),
        Case('fma_subnormal_and_exact_one', 'fma', (257, 258, 259),
             ((0x3800 << 16) | 0x3800, (0x0400 << 16) | 0x3800, (0x0001 << 16) | 0x3A00)),
        Case('fma_near_one_and_overflow_avoid', 'fma', (257, 258, 259),
             ((0x7BFF << 16) | 0x3800, (0x4000 << 16) | 0x4000, (0xFBFF << 16) | 0x9000)),
        Case('fma_nan_and_inf_zero', 'fma', (257, 258, 259),
             ((0x7E01 << 16) | 0x7C00, (0x3C00 << 16) | 0x0000, (0x3C00 << 16) | 0x3C00)),
        Case('fma_neg_zero_and_neg_inf', 'fma', (257, 258, 259),
             ((0x8000 << 16) | 0xFC00, (0x3C00 << 16) | 0x3C00, (0x8000 << 16) | 0x0000)),
        # 4. VOP3P selectors (op_sel / op_sel_hi) + neg_lo / neg_hi combined with clamp.
        Case('add_opsel_neg_clamp', 'add', (257, 258),
             ((0x3800 << 16) | 0x4000, (0x3C00 << 16) | 0xBC00), op_sel=0b01, op_sel_hi=0b10, neg_lo=0b10, neg_hi=0b01),
        Case('mul_opsel_neg_clamp', 'mul', (257, 258),
             ((0x3800 << 16) | 0xC000, (0x4000 << 16) | 0x3C00), op_sel=0b10, op_sel_hi=0b01, neg_lo=0b01, neg_hi=0b10),
        Case('fma_opsel_neg_clamp', 'fma', (257, 258, 259),
             ((0x3800 << 16) | 0x4000, (0x4000 << 16) | 0x3800, (0xBC00 << 16) | 0x3400),
             op_sel=0b101, op_sel_hi=0b010, neg_lo=0b100, neg_hi=0b011),
        # 5. SGPR, literal (255), inline f16 float (242/243/240), inline int (193 NaN -> +0.0 under clamp).
        Case('add_sgpr_literal_clamp', 'add', (10, 255), ((0x4000 << 16) | 0xBC00, (0x3800 << 16) | 0x3800)),
        Case('mul_inline_242_243_clamp', 'mul', (257, 243), ((0x3800 << 16) | 0xBC00, 0),
             op_sel=0, op_sel_hi=0b101, neg_lo=0b10),
        Case('fma_inline_int_193_nan_clamp', 'fma', (257, 240, 193), ((0x4000 << 16) | 0x3C00, 0, 0),
             op_sel=0, op_sel_hi=0b101),
        # 6. Partial EXEC_LO mask: inactive lanes keep DST_INIT.
        Case('fma_partial_exec_clamp', 'fma', (257, 258, 259),
             ((0x4000 << 16) | 0xBC00, (0x3C00 << 16) | 0x3C00, (0x0000 << 16) | 0x0000),
             exec_mask=0xAAAAAAAA),
    ]


def emit_case(c: Case):
    code = F.mov(4, DST_INIT)
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 255), c.exec_mask]
    lit_tail = []
    for s, v in zip(c.srcs, c.vals):
        if s >= 256:
            code += F.mov(s - 256, v)
        elif s <= 105:
            code += F.smov(s, v)
        elif s == 255:
            lit_tail = [v]
    code += vop3p(c.op, 4, c.srcs, op_sel=c.op_sel, op_sel_hi=c.op_sel_hi,
                  neg_lo=c.neg_lo, neg_hi=c.neg_hi, clamp=c.clamp)
    code += lit_tail
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 193)]
    return code


def compute_program(cases):
    assert len(cases) <= 32, len(cases)
    code = list(PROLOGUE)
    for slot, c in enumerate(cases):
        code += emit_case(c) + F.store(slot * 4, 4)
    return code + [0xBF810000]


def refusal_programs():
    cond_mode = [0xBF840001, 0xBFA5000F]  # s_cbranch_scc0 +1; s_denorm_mode 15
    clamped_add = vop3p('add', 4, (257, 258), clamp=1) + [0xBF810000]
    add_bad_src2 = vop3p('add', 4, (257, 258), clamp=1)
    add_bad_src2[1] |= 259 << 18
    fmac_e64_clamp = [0xD53C8004, (258 << 9) | 257]  # VOP3 e64 v_pk_fmac_f16 with clamp=1
    return [
        # Unused third-operand fields and VOP3 e64 v_pk_fmac_f16 clamp refused.
        (add_bad_src2 + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
        (fmac_e64_clamp + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
        (vop3p('mul', 4, (257, 258), op_sel=4, clamp=1) + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
        (vop3p('add', 4, (257, 258), neg_lo=4, clamp=1) + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
        (vop3p('mul', 4, (257, 258), neg_hi=4, clamp=1) + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
        # Special scalar source refused.
        (vop3p('fma', 4, (257, 106, 259), clamp=1) + [0xBF810000], 'unsupported packed binary16 arithmetic source'),
        # Unsupported binary16 rounding/denormal modes and dynamic conditional MODE write.
        ([0xBFA40004] + clamped_add, 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50003] + clamped_add, 'unsupported binary16 rounding or denormal mode'),
        (cond_mode + clamped_add, 'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'packed-f16-clamp')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_packed_f16_clamp IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = all_cases()
    code = compute_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'fadd float' in dump and 'fmul float' in dump and '@dx.op.unaryBits.i32(i32 33,' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_packed_f16_clamp controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = all_cases()
    code = compute_program(items)
    marker = request(work, root, code)
    output = work / 'out.bin'
    run = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        return 77
    run.check_returncode()
    words_out = list(struct.unpack(f'<{F.WORDS}I', output.read_bytes()))
    expected = {}
    for group in range(F.GROUPS):
        for lane in range(32):
            head = F.BASE + group * 1024 + lane * 32
            for idx, c in enumerate(items):
                expected[head + idx] = c.expected(lane=lane)
    bad = []
    for i, got in enumerate(words_out):
        if i in expected:
            if got != expected[i]:
                bad.append((i, hex(got), hex(expected[i])))
        elif got != F.SENTINEL:
            bad.append((i, hex(got), hex(F.SENTINEL)))
    assert not bad, bad[:10]
    print(f'native_packed_f16_clamp compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
    return 0


def pixel_program(pair):
    code = []
    for idx, c in enumerate(pair):
        code += emit_case(c)
        lo_reg, hi_reg = 8 + 2 * idx, 9 + 2 * idx
        code += [vop2(0x1B, lo_reg, 255, 4), 65535, vop2(0x16, hi_reg, 144, 4)]
        # Each 16-bit half is exactly representable in binary32 as an integer in 0..65535.
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (lo_reg, hi_reg)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def pixel(cli, sdk, renderer, adapter, work):
    items = [c for c in all_cases() if c.exec_mask == ALL_EXEC]
    if len(items) % 2:
        items.append(items[-1])
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    pairs = [items[i:i + 2] for i in range(0, len(items), 2)]
    for idx, pair in enumerate(pairs):
        shader = work / f'packed-f16-clamp-{idx}.ps'
        shader.write_bytes(pixel_program(pair))
        cso = shader.with_suffix('.cso')
        build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                               capture_output=True, text=True, timeout=30)
        assert build.returncode == 0, ([c.name for c in pair], build.stdout, build.stderr)
        w0, w1 = pair[0].expected(), pair[1].expected()
        ref = work / 'expected.bin'
        ref.write_bytes(struct.pack('<4f', w0 & 0xFFFF, w0 >> 16, w1 & 0xFFFF, w1 >> 16))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', ref],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            return 77
        assert run.returncode == 0, (idx, [c.name for c in pair], run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'native_packed_f16_clamp pixel ({adapter}): {len(pairs)} pixel draws ({len(items)} cases) matched')
    return 0


def main(mode, tool, sdk=None, *args):
    if mode == 'ir':
        return ir(tool)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if mode == 'pixel':
            return pixel(tool, sdk, args[0], args[1], work)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'compute': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
