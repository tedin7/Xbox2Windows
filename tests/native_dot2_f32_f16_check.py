"""Direct DXIL v_dot2_f32_f16 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import INLINE_F16_WORDS, PROLOGUE, inline_word
from native_scalar_lds_check import vmove
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OP_DOT2_F32_F16 = 0x13


def vop3p_dot2(dst, srcs, *, op_sel=0, op_sel_hi=0, neg_lo=0, neg_hi=0, clamp=0, op_sel_hi2=0):
    s0, s1, s2 = srcs
    w0 = (0xCC000000 | (OP_DOT2_F32_F16 << 16) | ((clamp & 1) << 15) | ((op_sel_hi2 & 1) << 14) |
          ((op_sel & 7) << 11) | ((neg_hi & 7) << 8) | (dst & 0xFF))
    w1 = ((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF)
    return [w0, w1]


def f32_to_bits(val):
    return struct.unpack('<I', struct.pack('<f', val))[0]


def bits_to_f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]


def widen_f16(half_bits):
    return struct.unpack('<e', struct.pack('<H', half_bits & 0xFFFF))[0]


def f32_inline_bits(enc):
    f32_inline = {
        240: 0x3F000000, 241: 0xBF000000, 242: 0x3F800000, 243: 0xBF800000,
        244: 0x40000000, 245: 0xC0000000, 246: 0x40800000, 247: 0xC0800000,
        248: 0x3E22F983,
    }
    if enc in f32_inline:
        return f32_inline[enc]
    if 128 <= enc <= 192:
        return enc - 128
    if 193 <= enc <= 208:
        return (192 - enc) & 0xFFFFFFFF
    raise ValueError(enc)


def is_nan_f32(bits):
    return (bits & 0x7FFFFFFF) > 0x7F800000


def same_f32(got, want):
    if is_nan_f32(want):
        return is_nan_f32(got)
    return got == want


def eval_dot2(raw_words, *, op_sel=0, op_sel_hi=0, neg_lo=0, neg_hi=0, clamp=0):
    def term(word, selector, neg):
        half = (word >> 16) if selector else (word & 0xFFFF)
        if neg:
            half ^= 0x8000
        return widen_f16(half)

    a0 = term(raw_words[0], op_sel & 1, bool(neg_lo & 1))
    b0 = term(raw_words[1], (op_sel >> 1) & 1, bool(neg_lo & 2))
    a1 = term(raw_words[0], op_sel_hi & 1, bool(neg_hi & 1))
    b1 = term(raw_words[1], (op_sel_hi >> 1) & 1, bool(neg_hi & 2))
    c_bits = raw_words[2] ^ (0x80000000 if (neg_lo & 4) else 0)
    c = bits_to_f32(c_bits)

    p0 = bits_to_f32(f32_to_bits(a0 * b0))
    p1 = bits_to_f32(f32_to_bits(a1 * b1))
    s01 = bits_to_f32(f32_to_bits(p0 + p1))
    res = bits_to_f32(f32_to_bits(s01 + c))
    if clamp:
        res = 0.0 if (math.isnan(res) or res <= 0.0) else (1.0 if res >= 1.0 else res)
    return f32_to_bits(res)


@dataclass(frozen=True)
class Case:
    name: str
    srcs: tuple
    vals: tuple
    op_sel: int = 0
    op_sel_hi: int = 3
    neg_lo: int = 0
    neg_hi: int = 0
    clamp: int = 0
    op_sel_hi2: int = 0
    pixel_safe: bool = True

    def raw_words(self):
        words = []
        for idx, (s, v) in enumerate(zip(self.srcs, self.vals)):
            if s < 128 or s >= 255:
                words.append(v & 0xFFFFFFFF)
            elif idx < 2:
                words.append(inline_word(s))
            else:
                words.append(f32_inline_bits(s))
        return tuple(words)

    def expected(self):
        return eval_dot2(self.raw_words(), op_sel=self.op_sel, op_sel_hi=self.op_sel_hi,
                         neg_lo=self.neg_lo, neg_hi=self.neg_hi, clamp=self.clamp)


def p16(lo, hi):
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def all_cases():
    return [
        # 1. Basic VGPR dot product with packed halves and binary32 accumulator.
        Case('basic_vgpr', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(0.5))),
        # 2. All op_sel / op_sel_hi cross-lane combinations + LLVM-style op_sel_hi[2]=1.
        Case('opsel_swap', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(1.25)),
             op_sel=0b10, op_sel_hi=0b01, op_sel_hi2=1),
        Case('opsel_broadcast_lo', (257, 258, 259), (p16(0x3C00, 0x7E00), p16(0x4200, 0x7E00), f32_to_bits(2.0)),
             op_sel=0b00, op_sel_hi=0b00),
        Case('opsel_broadcast_hi', (257, 258, 259), (p16(0x7E00, 0x3C00), p16(0x7E00, 0x4000), f32_to_bits(-0.5)),
             op_sel=0b11, op_sel_hi=0b11),
        # 3. Negate modifiers: neg_lo[0], neg_lo[1], neg_lo[2] (f32 src2), neg_hi[0], neg_hi[1].
        Case('neg_pair0', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(10.0)),
             neg_lo=0b001),
        Case('neg_pair0_both_and_c', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(10.0)),
             neg_lo=0b111),
        Case('neg_pair1_hi0_hi1', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(1.0)),
             neg_hi=0b10),
        Case('neg_all_five', (257, 258, 259), (p16(0x3C00, 0x4000), p16(0x4200, 0x4400), f32_to_bits(3.0)),
             neg_lo=0b111, neg_hi=0b11),
        # 4. Clamp / saturate: < 0 -> +0.0, interior (0, 1), > 1 -> 1.0, exact boundary 0.0 / 1.0.
        Case('clamp_negative', (257, 258, 259), (p16(0xBC00, 0x3C00), p16(0x4000, 0x3800), f32_to_bits(-2.0)),
             clamp=1),
        Case('clamp_above_one', (257, 258, 259), (p16(0x4000, 0x4000), p16(0x4000, 0x4000), f32_to_bits(0.5)),
             clamp=1),
        Case('clamp_interior', (257, 258, 259), (p16(0x3800, 0x3400), p16(0x3800, 0x3C00), f32_to_bits(0.125)),
             clamp=1),
        Case('clamp_nan_to_zero', (257, 258, 259), (p16(0x7E00, 0x3C00), p16(0x3C00, 0x3C00), f32_to_bits(0.5)),
             clamp=1, pixel_safe=False),
        # 5. Inline binary16 float constants on src0/src1 (including zero high half 242, 245, 248) and f32 src2 inline.
        Case('inline_f16_lo_242_hi_zero', (242, 258, 240), (0, p16(0x4000, 0x4400), 0)),
        Case('inline_f16_245_248', (245, 248, 259), (0, 0, f32_to_bits(1.0)), op_sel=0b00, op_sel_hi=0b11),
        Case('inline_src2_243_neg_c', (257, 258, 243), (p16(0x3C00, 0x3C00), p16(0x3C00, 0x4000), 0), neg_lo=0b100),
        # 6. Inline integer constants on src0/src1 (129=subnormal 2^-24, 192=64 subnormal) & src2 (128=+0, 129=2^-149).
        Case('inline_int_129_192_128', (129, 192, 128), (0, 0, 0)),
        Case('inline_int_src2_128_zero', (257, 258, 128), (p16(0x3C00, 0x3C00), p16(0x4000, 0x3800), 0)),
        Case('inline_int_src2_129', (257, 258, 129), (p16(0x0000, 0x0000), p16(0x0000, 0x0000), 0)),
        Case('vgpr_src2_subnormal_pos', (257, 258, 259), (p16(0x0000, 0x0000), p16(0x0000, 0x0000), 0x00000001)),
        Case('vgpr_src2_subnormal_neg', (257, 258, 259), (p16(0x0000, 0x0000), p16(0x0000, 0x0000), 0x80400000)),
        Case('vgpr_src2_subnormal_clamp_pos', (257, 258, 259), (p16(0x0000, 0x0000), p16(0x0000, 0x0000), 0x00000001), clamp=1),
        Case('vgpr_src2_subnormal_clamp_neg', (257, 258, 259), (p16(0x0000, 0x0000), p16(0x0000, 0x0000), 0x80400000), clamp=1),
        # 7. SGPR and 32-bit literal sources.
        Case('sgpr_sources', (10, 11, 12), (p16(0x3C00, 0xC000), p16(0x4200, 0x3800), f32_to_bits(0.75))),
        Case('literal_src0', (255, 258, 259), (p16(0x4000, 0x3C00), p16(0x3800, 0x4200), f32_to_bits(-1.0))),
        Case('literal_src2', (257, 258, 255), (p16(0x3C00, 0x3800), p16(0x3C00, 0x4000), f32_to_bits(3.25))),
        # 8. Binary16 subnormals, signed zeros, infinity, max finite, and NaN propagation.
        Case('subnormal_f16_exact', (257, 258, 259), (p16(0x0001, 0x03FF), p16(0x0001, 0x7BFF), f32_to_bits(0.0))),
        Case('signed_zero_keep', (257, 258, 259), (p16(0x8000, 0x8000), p16(0x3C00, 0x3C00), f32_to_bits(-0.0)),
             pixel_safe=False),
        Case('signed_zero_cancel_pos', (257, 258, 259), (p16(0x3C00, 0x3C00), p16(0xBC00, 0x3C00), f32_to_bits(-0.0)),
             pixel_safe=False),
        Case('inf_propagation', (257, 258, 259), (p16(0x7C00, 0x3C00), p16(0x3C00, 0x3C00), f32_to_bits(-4.0)),
             pixel_safe=False),
        Case('nan_propagation', (257, 258, 259), (p16(0x7E01, 0x3C00), p16(0x3C00, 0x3C00), f32_to_bits(1.0)),
             pixel_safe=False),
    ]


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
    code += vop3p_dot2(4, c.srcs, op_sel=c.op_sel, op_sel_hi=c.op_sel_hi,
                       neg_lo=c.neg_lo, neg_hi=c.neg_hi, clamp=c.clamp, op_sel_hi2=c.op_sel_hi2)
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
    # Conditional branch + s_denorm_mode makes half_mode_known == false for v_dot2_f32_f16.
    cond_mode = [0xBF840001, 0xBFA5000F, 0xBF810000]  # s_cbranch_scc0 +1; s_denorm_mode 15
    return [
        # Reserved OP_SEL[2] and NEG_HI[2] bits on v_dot2_f32_f16.
        (vop3p_dot2(4, (257, 258, 259), op_sel=0b100) + [0xBF810000], 'unsupported v_dot2_f32_f16 encoding'),
        (vop3p_dot2(4, (257, 258, 259), neg_hi=0b100) + [0xBF810000], 'unsupported v_dot2_f32_f16 encoding'),
        # Special scalar sources refused on src0, src1, src2.
        (vop3p_dot2(4, (106, 258, 259)) + [0xBF810000], 'unsupported v_dot2_f32_f16 source'),
        (vop3p_dot2(4, (257, 124, 259)) + [0xBF810000], 'unsupported v_dot2_f32_f16 source'),
        (vop3p_dot2(4, (257, 258, 126)) + [0xBF810000], 'unsupported v_dot2_f32_f16 source'),
        (vop3p_dot2(4, (257, 258, 253)) + [0xBF810000], 'unsupported v_dot2_f32_f16 source'),
        # Unsupported binary16 denormal flush (s_denorm_mode 3) or rounding mode (s_round_mode 4).
        ([0xBFA50003] + vop3p_dot2(4, (257, 258, 259)) + [0xBF810000],
         'unsupported binary16 rounding or denormal mode'),
        ([0xBFA40004] + vop3p_dot2(4, (257, 258, 259)) + [0xBF810000],
         'unsupported binary16 rounding or denormal mode'),
        # Unsupported binary32 denormal flush (s_denorm_mode 0xC) refused via mode gate.
        ([0xBFA5000C] + vop3p_dot2(4, (257, 258, 259)) + [0xBF810000],
         'unsupported compute control instruction s_denorm_mode'),
        # Dynamic conditional mode change makes half_mode_known false via half_binary tracking.
        (cond_mode[:-1] + vop3p_dot2(4, (257, 258, 259)) + [0xBF810000],
         'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = all_cases()
    code, want = compute_program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'dot2-f32-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if not same_f32(got, exp)]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_dot2_f32_f16 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
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
    assert 'legacyF16ToF32' in dump and 'fmul float' in dump and 'fadd float' in dump and 'unary.f32(i32 7,' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_dot2_f32_f16 controls: compiled {len(want)} cases to DXIL; verified {len(refusals)} refusals')
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
    expected = {}
    for group in range(F.GROUPS):
        for lane in range(32):
            head = F.BASE + group * 1024 + lane * 32
            for idx, w in enumerate(want):
                expected[head + idx] = w
    bad = []
    for i, got in enumerate(words_out):
        if i in expected:
            if not same_f32(got, expected[i]):
                bad.append((i, hex(got), hex(expected[i])))
        elif got != F.SENTINEL:
            bad.append((i, hex(got), hex(F.SENTINEL)))
    assert not bad, bad[:10]
    print(f'native_dot2_f32_f16 compute hardware: {len(want)} cases matched across {F.GROUPS}x32 lanes and guards')
    return 0


def pixel_program(quad):
    code = []
    for idx, c in enumerate(quad):
        code += emit_case(c)
        code.append(vmove(8 + idx, 256 + 4))
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def pixel(cli, sdk, renderer, adapter, work):
    items = [c for c in all_cases() if c.pixel_safe]
    while len(items) % 4:
        items.append(items[-1])
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    quads = [items[i:i + 4] for i in range(0, len(items), 4)]
    for idx, quad in enumerate(quads):
        shader = work / f'dot2-{idx}.ps'
        shader.write_bytes(pixel_program(quad))
        cso = shader.with_suffix('.cso')
        build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                               capture_output=True, text=True, timeout=30)
        assert build.returncode == 0, ([c.name for c in quad], build.stdout, build.stderr)
        ref = work / 'expected.bin'
        ref.write_bytes(struct.pack('<4I', *(c.expected() for c in quad)))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', ref],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            return 77
        assert run.returncode == 0, (idx, [c.name for c in quad], run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'native_dot2_f32_f16 pixel ({adapter}): {len(quads)} quad draws matched')
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
