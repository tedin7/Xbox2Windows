"""Direct DXIL packed binary16 arithmetic and min/max with inline constant sources."""
from dataclasses import dataclass
from fractions import Fraction
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
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl


OPS_3SRC = {'fma': 0x0e}
OPS_2SRC = {'add': 0x0f, 'mul': 0x10, 'min_f16': 0x11, 'max_f16': 0x12, 'min_u16': 0x0d, 'max_i16': 0x07}
OPS = {**OPS_3SRC, **OPS_2SRC}

INLINE_F16_WORDS = {
    240: 0x3800, 241: 0xB800, 242: 0x3C00, 243: 0xBC00,
    244: 0x4000, 245: 0xC000, 246: 0x4400, 247: 0xC400,
    248: 0x3118,
}

GUARD = 0x69696969
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]


def inline_word(enc):
    if enc in INLINE_F16_WORDS:
        return INLINE_F16_WORDS[enc]
    if 128 <= enc <= 192:
        return enc - 128
    if 193 <= enc <= 208:
        return (192 - enc) & 0xFFFFFFFF
    raise ValueError(enc)


def vop3p(op_name, dst, srcs, *, op_sel=0, op_sel_hi=7, neg_lo=0, neg_hi=0, clamp=0):
    arity = 3 if op_name in OPS_3SRC else 2
    s0, s1 = srcs[0], srcs[1]
    s2 = srcs[2] if arity == 3 else 0
    # Default op_sel_hi[2]=1 for 2-src instructions (LLVM MC convention).
    hi2 = (op_sel_hi >> 2) & 1 if arity == 3 else 1
    w0 = (0xCC000000 | (OPS[op_name] << 16) | ((clamp & 1) << 15) |
          (hi2 << 14) | ((op_sel & 7) << 11) | ((neg_hi & 7) << 8) | (dst & 0xFF))
    w1 = ((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF)
    return [w0, w1]


def f16_key(bits):
    bits &= 0xFFFF
    return (bits ^ 0xFFFF) if (bits & 0x8000) else (bits ^ 0x8000)


def f16_minmax(a, b, is_max):
    a &= 0xFFFF
    b &= 0xFFFF
    a_nan = (a & 0x7FFF) > 0x7C00
    b_nan = (b & 0x7FFF) > 0x7C00
    if a_nan and b_nan:
        return 0x7E00
    if a_nan:
        return b
    if b_nan:
        return a
    ka, kb = f16_key(a), f16_key(b)
    return (a if ka > kb else b) if is_max else (a if ka < kb else b)


def int16_minmax(a, b, is_max, signed):
    a &= 0xFFFF
    b &= 0xFFFF
    if signed:
        sa = a - 0x10000 if (a & 0x8000) else a
        sb = b - 0x10000 if (b & 0x8000) else b
        return (a if sa > sb else b) if is_max else (a if sa < sb else b)
    return (a if a > b else b) if is_max else (a if a < b else b)


def eval_case(op_name, raw_words, op_sel, op_sel_hi, neg_lo, neg_hi):
    arity = 3 if op_name in OPS_3SRC else 2

    def lane_eval(high):
        sel = op_sel_hi if high else op_sel
        neg = neg_hi if high else neg_lo
        ops = []
        for j in range(arity):
            half = (raw_words[j] >> (16 if ((sel >> j) & 1) else 0)) & 0xFFFF
            if (neg >> j) & 1:
                half ^= 0x8000
            ops.append(half)
        if op_name in ('add', 'mul', 'fma'):
            return half_op(op_name, *ops)
        if op_name == 'min_f16':
            return f16_minmax(ops[0], ops[1], False)
        if op_name == 'max_f16':
            return f16_minmax(ops[0], ops[1], True)
        if op_name == 'min_u16':
            return int16_minmax(ops[0], ops[1], False, False)
        if op_name == 'max_i16':
            return int16_minmax(ops[0], ops[1], True, True)
        raise ValueError(op_name)

    return lane_eval(False) | (lane_eval(True) << 16)


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    srcs: tuple      # encoding per operand (>=256 for VGPR v[idx-256], <=105 for SGPR, 128..208/240..248 inline, 255 literal)
    vals: tuple      # 32-bit raw value per operand (ignored for inline; used for VGPR/SGPR/literal)
    op_sel: int = 0
    op_sel_hi: int = 7
    neg_lo: int = 0
    neg_hi: int = 0

    def raw_words(self):
        return tuple(inline_word(s) if 128 <= s < 255 else v for s, v in zip(self.srcs, self.vals))

    def expected(self):
        return eval_case(self.op, self.raw_words(), self.op_sel, self.op_sel_hi, self.neg_lo, self.neg_hi)


def all_cases():
    items = []
    # 1. All 9 inline binary16 float constants (240..248) broadcast to both lanes via op_sel=0, op_sel_hi=0.
    base_vec = (0x4200 << 16) | 0x3E00  # hi=3.0, lo=1.5
    for enc in range(240, 249):
        items.append(Case(f'mul_inline_{enc}_broadcast', 'mul', (257, enc), (base_vec, 0), op_sel=0, op_sel_hi=0b101))
    # 2. gfx1013 high-half zeroing on inline float constants (op_sel_hi selects bit 1 -> reads +0.0 = 0x0000).
    items.append(Case('add_inline_242_hi_zero', 'add', (257, 242), ((0x4000 << 16) | 0x3C00, 0), op_sel=0, op_sel_hi=0b111))
    items.append(Case('mul_inline_244_lo_hi_zero', 'mul', (244, 257), (0, (0x4200 << 16) | 0x3C00), op_sel=0b001, op_sel_hi=0b111))
    # 3. neg_lo and neg_hi on inline float constants (including negating zeroed high half -> -0.0 = 0x8000).
    items.append(Case('add_inline_242_neg_hi_zero', 'add', (257, 242), ((0x8000 << 16) | 0x3C00, 0),
                      op_sel=0, op_sel_hi=0b111, neg_lo=0b10, neg_hi=0b10))
    items.append(Case('mul_inline_240_neg_broadcast', 'mul', (257, 240), ((0x4400 << 16) | 0x4000, 0),
                      op_sel=0, op_sel_hi=0b101, neg_lo=0b10, neg_hi=0b10))
    # 4. Single-rounding FMA tie witness with inline float 240 (0.5) and inline int 129 (0x0001 = 2^-24).
    #    Low lane: (1 + 2^-10) * 0.5 + 2^-24 -> 0x3801 (double-rounded f32 product would give 0x3800).
    #    High lane: (1 + 3*2^-10) * 0.5 + 0 -> 0x3802.
    items.append(Case('fma_inline_tie_pos', 'fma', (257, 240, 129), ((0x3C03 << 16) | 0x3C01, 0, 0),
                      op_sel=0, op_sel_hi=0b101))
    #    Negative tie witness using neg_lo/neg_hi on inline 240 and inline 129 in src2 + SGPR src0.
    items.append(Case('fma_inline_src0_242_src2_241', 'fma', (242, 257, 241), (0, (0x4000 << 16) | 0x3C00, 0),
                      op_sel=0, op_sel_hi=0b010))
    # 5. Inline integer constants: 128 (0), 129 (1=0x00000001), 192 (64=0x00000040),
    #    193 (-1=0xffffffff -> NaN in both halves), 208 (-16=0xfffffff0 -> NaN hi, fff0=NaN lo).
    items.append(Case('add_inline_int_192', 'add', (257, 192), ((0x4000 << 16) | 0x3C00, 0), op_sel=0, op_sel_hi=0b111))
    items.append(Case('fma_inline_int_193_nan', 'fma', (257, 242, 193), ((0x4000 << 16) | 0x3C00, 0, 0),
                      op_sel=0, op_sel_hi=0b101))
    items.append(Case('add_inline_int_208_nan', 'add', (257, 208), ((0x4000 << 16) | 0x3C00, 0), op_sel=0, op_sel_hi=0b111))
    # 6. Packed f16 min/max with inline float constants (242, 243, 244, high-half zeroing, and 193 NaN selecting numeric).
    items.append(Case('min_f16_inline_242_broadcast', 'min_f16', (257, 242), ((0x4000 << 16) | 0x3800, 0), op_sel=0, op_sel_hi=0b101))
    items.append(Case('max_f16_inline_243_hi_zero', 'max_f16', (257, 243), ((0x8000 << 16) | 0xBC00, 0), op_sel=0, op_sel_hi=0b111))
    items.append(Case('min_f16_inline_193_nan', 'min_f16', (193, 257), (0, (0x4000 << 16) | 0x3C00), op_sel=0, op_sel_hi=0b111))
    items.append(Case('max_f16_inline_248_sgpr', 'max_f16', (10, 248), ((0x3400 << 16) | 0x3000, 0), op_sel=0, op_sel_hi=0b101))
    # 7. Packed integer min/max with inline integers (128, 129, 192, 193=-1, 208=-16).
    items.append(Case('min_u16_inline_192', 'min_u16', (257, 192), ((0x0010 << 16) | 0x0050, 0), op_sel=0, op_sel_hi=0b111))
    items.append(Case('max_i16_inline_193_208', 'max_i16', (193, 208), (0, 0), op_sel=0, op_sel_hi=0b111))
    return items


def emit_case(c: Case):
    code = []
    lit_tail = []
    for idx, (s, v) in enumerate(zip(c.srcs, c.vals)):
        if s >= 256:
            code += F.mov(s - 256, v)
        elif s <= 105:
            code += F.smov(s, v)
        elif s == 255:
            lit_tail = [v]
    code += vop3p(c.op, 4, c.srcs, op_sel=c.op_sel, op_sel_hi=c.op_sel_hi, neg_lo=c.neg_lo, neg_hi=c.neg_hi)
    code += lit_tail
    return code


def compute_program(cases):
    code = list(PROLOGUE)
    want = []
    for slot, c in enumerate(cases):
        code += emit_case(c) + F.store(slot * 4, 4)
        want.append(c.expected())
    return code + [0xBF810000], want


def refusal_programs():
    return [
        # Special scalar sources refused in packed_f16_alu
        (vop3p('add', 4, (257, 106)) + [0xBF810000], 'unsupported packed binary16 arithmetic source'),
        (vop3p('mul', 4, (124, 257)) + [0xBF810000], 'unsupported packed binary16 arithmetic source'),
        (vop3p('fma', 4, (257, 242, 126)) + [0xBF810000], 'unsupported packed binary16 arithmetic source'),
        (vop3p('fma', 4, (257, 242, 253)) + [0xBF810000], 'unsupported packed binary16 arithmetic source'),
        # Special scalar sources refused in packed_minmax
        (vop3p('min_f16', 4, (257, 106)) + [0xBF810000], 'unsupported packed 16-bit min/max source'),
        (vop3p('max_f16', 4, (124, 257)) + [0xBF810000], 'unsupported packed 16-bit min/max source'),
        (vop3p('min_f16', 4, (257, 253)) + [0xBF810000], 'unsupported packed 16-bit min/max source'),
        # Float inline constants refused on integer packed min/max
        (vop3p('min_u16', 4, (257, 242)) + [0xBF810000], 'unsupported packed 16-bit min/max source'),
        (vop3p('max_i16', 4, (240, 257)) + [0xBF810000], 'unsupported packed 16-bit min/max source'),
        # Unused third-source modifier refused on 2-source VOP3P add
        (vop3p('add', 4, (257, 242), neg_lo=4) + [0xBF810000], 'unsupported packed binary16 arithmetic encoding'),
    ]


def ir(dumper):
    items = all_cases()
    code, want = compute_program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'packed-f16-inline')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_packed_f16_inline IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = all_cases()
    code, want = compute_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'fmul float' in dump and 'fadd float' in dump and '@dx.op.unaryBits.i32(i32 33,' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_packed_f16_inline controls: compiled {len(want)} cases to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = all_cases()
    code, want = compute_program(items)
    marker = request(work, root, code)
    output = work / 'out.bin'
    run = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        return 77
    run.check_returncode()
    words_out = list(struct.unpack(f'<{F.WORDS}I', output.read_bytes()))
    expected = [F.SENTINEL] * F.WORDS
    for group in range(F.GROUPS):
        for lane in range(32):
            head = F.BASE + group * 1024 + lane * 32
            expected[head:head + len(want)] = want
    mismatch = next((i for i, (a, b) in enumerate(zip(words_out, expected, strict=True)) if a != b), None)
    assert mismatch is None, (mismatch, hex(words_out[mismatch]), hex(expected[mismatch]))
    print(f'native_packed_f16_inline compute hardware: {len(want)} cases matched across {F.GROUPS}x32 lanes and guards')
    return 0


def pixel_program(case_a: Case, case_b: Case):
    code = []
    for idx, c in enumerate((case_a, case_b)):
        code += emit_case(c)
        lo, hi = 8 + 2 * idx, 9 + 2 * idx
        code += [vop2(0x1B, lo, 255, 4), 65535, vop2(0x16, hi, 144, 4)]
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (lo, hi)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def pixel(cli, sdk, renderer, adapter, work):
    items = all_cases()
    if len(items) % 2:
        items.append(items[-1])
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    pairs = [(items[i], items[i + 1]) for i in range(0, len(items), 2)]
    for idx, (ca, cb) in enumerate(pairs):
        shader = work / f'inline-{idx}.ps'
        shader.write_bytes(pixel_program(ca, cb))
        cso = shader.with_suffix('.cso')
        build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                               capture_output=True, text=True, timeout=30)
        assert build.returncode == 0, (ca.name, cb.name, build.stdout, build.stderr)
        ea, eb = ca.expected(), cb.expected()
        ref = work / 'expected.bin'
        ref.write_bytes(struct.pack('<4f', ea & 0xFFFF, ea >> 16, eb & 0xFFFF, eb >> 16))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', ref],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            return 77
        assert run.returncode == 0, (idx, ca.name, cb.name, run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'native_packed_f16_inline pixel ({adapter}): {len(pairs)} paired draws ({len(all_cases())} cases) matched')
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
