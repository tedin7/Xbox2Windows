"""Direct DXIL VOP1/VOP3 v_fract_f16 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest import mock

from f16_packed_model import h2f, round_exact
from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE, inline_word
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OP_VOP1_FRACT_F16 = 0x5F
OP_VOP3_FRACT_F16 = 0x1DF
ALL_EXEC = 0xFFFFFFFF
DST_INIT = 0xDEADCAFE


def vop1_fract_f16(dst, src0):
    return [0x7E000000 | ((dst & 0xFF) << 17) | (OP_VOP1_FRACT_F16 << 9) | (src0 & 0x1FF)]


def vop3_fract_f16(dst, src0, *, abs_=0, neg=0, op_sel=0, clamp=0, omod=0, s1=0, s2=0):
    w0 = (0xD4000000 | (OP_VOP3_FRACT_F16 << 16) | ((clamp & 1) << 15) | ((op_sel & 15) << 11) |
          ((abs_ & 7) << 8) | (dst & 0xFF))
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def eval_fract_f16(raw_word, *, abs_=0, neg=0, op_sel=0, clamp=0, dst_init=DST_INIT):
    bits = (raw_word >> (16 if (op_sel & 1) else 0)) & 0xFFFF
    if abs_ & 1:
        bits &= 0x7FFF
    if neg & 1:
        bits ^= 0x8000
    if (bits & 0x7FFF) >= 0x7C00:
        res = 0x0000 if clamp else 0x7E00
    else:
        x = Fraction(h2f(bits))
        diff = x - math.floor(x)
        res = 0x0000 if diff == 0 else min(round_exact(diff), 0x3BFF)
    if op_sel & 8:
        return ((res & 0xFFFF) << 16) | (dst_init & 0xFFFF)
    return (dst_init & 0xFFFF0000) | (res & 0xFFFF)


@dataclass(frozen=True)
class Case:
    name: str
    src0: int
    val0: int
    wide: bool = False
    abs_: int = 0
    neg: int = 0
    op_sel: int = 0
    clamp: int = 0
    exec_mask: int = ALL_EXEC

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return DST_INIT
        raw = self.val0 & 0xFFFFFFFF if (self.src0 < 128 or self.src0 >= 255) else inline_word(self.src0)
        return eval_fract_f16(raw, abs_=self.abs_, neg=self.neg, op_sel=self.op_sel, clamp=self.clamp)


def all_cases():
    return [
        # 1. Positive fractions and exact positive integers (with nonzero unused high halves on op_sel[0]=0 VGPR reads).
        Case('pos_zero', 257, 0x0000),
        Case('pos_frac_mixed', 257, 0x7BFF3D00),      # 1.25 -> 0.25 (hi=65504 ignored)
        Case('pos_exact_one', 257, 0x3C00),           # 1.0 -> +0.0
        Case('pos_exact_max', 257, 0x7BFF),           # 65504 -> +0.0
        # 2. Negative fractions and exact negative integers (-0.0, -1.0, -65504).
        Case('neg_zero', 257, 0x8000),                # -0.0 -> +0.0
        Case('neg_quarter', 257, 0x7BFFB400),         # -0.25 -> 0.75 (hi=65504 ignored)
        Case('neg_1_75', 257, 0xBF00),                # -1.75 -> 0.25
        Case('neg_exact_one', 257, 0xBC00),           # -1.0 -> +0.0
        Case('neg_exact_max', 257, 0xFBFF),           # -65504 -> +0.0
        # 3. Tiny negatives: x - floor(x) near 1.0 (0x3bff clamp + RNE-vs-truncation discriminators).
        Case('tiny_neg_2_m11', 257, 0x9000),          # -2^-11 -> 0x3bff exact
        Case('tiny_neg_2_m12', 257, 0x8C00),          # -2^-12 (rounds to 1.0 without clamp) -> 0x3bff
        Case('tiny_neg_5_m12', 257, 0x9500),          # -5*2^-12 (halfway tie -> 0x3bfe under RNE vs 0x3bfd trunc)
        Case('tiny_neg_9_m13', 257, 0x9480),          # -9*2^-13 (rounds to 0x3bfe under RNE vs 0x3bfd trunc)
        Case('tiny_neg_min_subnormal', 257, 0x8001),  # -2^-24 -> 0x3bff
        # 4. Positive binary16 subnormal preserved as exact fraction.
        Case('pos_min_subnormal', 257, 0x0001),
        # 5. Non-finite inputs (+Inf, -Inf, NaN -> 0x7e00 or +0.0 with clamp) and finite clamp=1 preservation.
        Case('pos_inf', 257, 0x7C00),
        Case('neg_inf', 257, 0xFC00),
        Case('nan_payload', 257, 0x7E01),
        Case('vop3_clamp_inf_and_neg', 257, 0xFC00, wide=True, clamp=1),
        Case('vop3_clamp_nan', 257, 0x7E01, wide=True, clamp=1),
        Case('vop3_clamp_pos', 257, 0x7BFF3D00, wide=True, clamp=1),
        Case('vop3_clamp_neg', 257, 0x7BFFB400, wide=True, clamp=1),
        # 6. VOP3 modifiers: separate abs, neg, and high-half op_sel[0]/op_sel[3] preserving D[15:0].
        Case('vop3_abs', 257, 0x7BFFB400, wide=True, abs_=1),  # abs(-0.25) = 0.25 -> 0.25
        Case('vop3_neg', 257, 0x7BFF3400, wide=True, neg=1),   # neg(0.25) = -0.25 -> 0.75
        Case('vop3_opsel_hi_src', 257, 0x3D003C00, wide=True, op_sel=0b0001),
        Case('vop3_opsel_hi_dst', 257, 0x3C003D00, wide=True, op_sel=0b1000),
        Case('vop3_opsel_hi_rw', 257, 0x3D00BEEF, wide=True, op_sel=0b1001),
        # 7. SGPR, literal (255), inline binary16 float (248=0x3118) and inline integer subnormal (129=0x0001).
        Case('sgpr_src0', 10, 0xB400),
        Case('literal_src0', 255, 0x8C00),
        Case('inline_f16_248', 248, 0),
        Case('inline_int_129', 129, 0),
        # 8. Partial EXEC_LO mask: inactive lanes keep DST_INIT.
        Case('partial_exec', 257, 0x7BFF3D00, exec_mask=0xAAAAAAAA),
    ]


def emit_case(c: Case):
    code = F.mov(4, DST_INIT)
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 255), c.exec_mask]
    lit_tail = []
    if c.src0 >= 256:
        code += F.mov(c.src0 - 256, c.val0)
    elif c.src0 <= 105:
        code += F.smov(c.src0, c.val0)
    elif c.src0 == 255:
        lit_tail = [c.val0]
    if c.wide:
        code += vop3_fract_f16(4, c.src0, abs_=c.abs_, neg=c.neg, op_sel=c.op_sel, clamp=c.clamp)
    else:
        code += vop1_fract_f16(4, c.src0)
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
    plain = vop1_fract_f16(4, 257) + [0xBF810000]
    return [
        # VOP3 output modifier omod != 0.
        (vop3_fract_f16(4, 257, omod=1) + [0xBF810000], 'unsupported v_fract_f16 omod'),
        # Unused VOP3 operand modifier/op_sel bits.
        (vop3_fract_f16(4, 257, op_sel=0b0010) + [0xBF810000], 'unsupported v_fract_f16 operand modifiers'),
        (vop3_fract_f16(4, 257, s1=258) + [0xBF810000], 'unsupported v_fract_f16 operand modifiers'),
        # High-half op_sel on non-VGPR source.
        (vop3_fract_f16(4, 10, op_sel=0b0001) + [0xBF810000],
         'unsupported v_fract_f16 high-half op_sel on non-VGPR source'),
        # DPP16 and SDWA forms.
        (vop1_fract_f16(4, 250) + [0xFF00E401, 0xBF810000], 'unsupported DPP arithmetic'),
        (vop1_fract_f16(4, 249) + [0x06060601, 0xBF810000], 'unimplemented direct SDWA operand profile'),
        # Special scalar source (vcc_lo).
        (vop1_fract_f16(4, 106) + [0xBF810000], 'unsupported v_fract_f16 source'),
        # Unsupported binary16 rounding/denorm modes and dynamic conditional MODE write.
        ([0xBFA40004] + plain, 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50003] + plain, 'unsupported binary16 rounding or denormal mode'),
        (cond_mode + plain, 'unsupported binary16 rounding or denormal mode'),
    ]


def _exec_with_floor(orig_exec):
    def wrapped(line, values, memory, stores):
        if 'unary.f32(i32 27,' in line:
            name, rhs = [part.strip() for part in line.split('=', 1)]
            token = re.search(r'i32 27,\s*float ([^,\s\)]+)\)', rhs).group(1)
            if token.startswith('%'):
                kind, bits = values[token]
                assert kind == 'float', line
            else:
                bits = struct.unpack('<I', struct.pack('<f', float(token)))[0]
            val = struct.unpack('<f', struct.pack('<I', bits))[0]
            if not (math.isnan(val) or math.isinf(val) or val == 0.0):
                bits = struct.unpack('<I', struct.pack('<f', float(math.floor(val))))[0]
            values[name] = ('float', bits)
            return
        orig_exec(line, values, memory, stores)
    return wrapped


def ir(dumper):
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'fract-f16')
        assert run.returncode == 0, run.stderr
        with mock.patch.object(F, '_exec', _exec_with_floor(F._exec)):
            stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_fract_f16 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
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
    assert 'legacyF16ToF32' in dump and 'fsub float' in dump and 'unary.f32(i32 27,' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_fract_f16 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
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
    print(f'native_fract_f16 compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
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
        shader = work / f'fract-f16-{idx}.ps'
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
    print(f'native_fract_f16 pixel ({adapter}): {len(pairs)} pixel draws ({len(items)} cases) matched')
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
