"""Direct DXIL VOP3 v_cvt_pknorm_i16_f32 and v_cvt_pknorm_u16_f32 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OPS = {'i16': 0x368, 'u16': 0x369}
ALL_EXEC = 0xFFFFFFFF
DST_INIT = 0xDEADCAFE

INLINE_F32_WORDS = {
    240: 0x3F000000, 241: 0xBF000000, 242: 0x3F800000, 243: 0xBF800000,
    244: 0x40000000, 245: 0xC0000000, 246: 0x40800000, 247: 0xC0800000,
    248: 0x3E22F983,
}


def inline_f32_word(enc):
    if enc in INLINE_F32_WORDS:
        return INLINE_F32_WORDS[enc]
    if 128 <= enc <= 192:
        return enc - 128
    if 193 <= enc <= 208:
        return (192 - enc) & 0xFFFFFFFF
    raise ValueError(enc)


def vop3_pknorm(kind, dst, srcs, *, abs_=0, neg=0, op_sel=0, clamp=0, omod=0, s2=0):
    s0, s1 = srcs
    w0 = (0xD4000000 | (OPS[kind] << 16) | ((clamp & 1) << 15) | ((op_sel & 15) << 11) |
          ((abs_ & 7) << 8) | (dst & 0xFF))
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF)
    return [w0, w1]


def cvt_norm_f32_half(kind, bits):
    bits &= 0xFFFFFFFF
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return 0
    val = struct.unpack('<f', struct.pack('<I', bits))[0]
    if kind == 'u16':
        if math.isnan(val) or val <= 0.0:
            return 0
        if val >= 1.0:
            return 0xFFFF
        scaled = Fraction(val) * 65535
    else:
        if val == 0.0:
            return 0
        if val >= 1.0:
            return 32767
        if val <= -1.0:
            return (-32767) & 0xFFFF
        scaled = Fraction(val) * 32767
    neg_sign = scaled < 0
    mag = abs(scaled)
    q = int(mag)
    rem = mag - q
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (q & 1)):
        q += 1
    return ((-q) & 0xFFFF) if neg_sign else (q & 0xFFFF)


def eval_pknorm(kind, raw0, raw1, *, abs_=0, neg=0):
    words = []
    for j, w in enumerate((raw0, raw1)):
        bits = w & 0xFFFFFFFF
        if (abs_ >> j) & 1:
            bits &= 0x7FFFFFFF
        if (neg >> j) & 1:
            bits ^= 0x80000000
        words.append(cvt_norm_f32_half(kind, bits))
    return words[0] | (words[1] << 16)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    srcs: tuple
    vals: tuple
    abs_: int = 0
    neg: int = 0
    exec_mask: int = ALL_EXEC

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return DST_INIT
        raw = [v & 0xFFFFFFFF if (s < 128 or s >= 255) else inline_f32_word(s)
               for s, v in zip(self.srcs, self.vals)]
        return eval_pknorm(self.kind, raw[0], raw[1], abs_=self.abs_, neg=self.neg)


def all_cases():
    return [
        # 1. Zeros, subnormals, 2^-17 boundary, unity, saturation (>1, <-1, +/-Inf), and NaNs.
        Case('i16_signed_zeros', 'i16', (257, 258), (0x00000000, 0x80000000)),
        Case('u16_neg_and_pos_zero', 'u16', (257, 258), (0x80000000, 0x00000000)),
        Case('i16_subnormals', 'i16', (257, 258), (0x00000001, 0x807FFFFF)),
        Case('u16_2m17_boundary', 'u16', (257, 258), (0x37000000, 0x37000100)),  # 0.5-eps -> 0, 0.5+eps -> 1
        Case('i16_2m17_neg_subone', 'i16', (257, 258), (0xB7000000, 0x3F7FFFFF)),  # -> 0, 32767
        Case('u16_tiny_and_almost_one', 'u16', (257, 258), (0x34000000, 0x3F7FFFFF)),  # -> 0, 65535
        Case('i16_unity_sat', 'i16', (257, 258), (0x3F800000, 0xBF800000)),       # -> +32767, -32767
        Case('u16_unity_neg_sat', 'u16', (257, 258), (0x3F800000, 0xBF800000)),   # -> 65535, 0
        Case('i16_over_under_sat', 'i16', (257, 258), (0x40000000, 0xC0000000)),  # 2.0, -2.0 -> 32767, -32767
        Case('u16_over_under_sat', 'u16', (257, 258), (0x60000000, 0xC0000000)),  # large, -2.0 -> 65535, 0
        Case('i16_infs', 'i16', (257, 258), (0x7F800000, 0xFF800000)),
        Case('u16_infs', 'u16', (257, 258), (0x7F800000, 0xFF800000)),
        Case('i16_nans', 'i16', (257, 258), (0x7FC00001, 0xFF800001)),
        Case('u16_nans', 'u16', (257, 258), (0x7F800001, 0xFFC00001)),
        # 2. Exact halfway RNE ties and p_lo != 0 carry-over tie-break witnesses.
        Case('i16_half_ties', 'i16', (257, 258), (0x3F000000, 0xBF000000)),       # 0.5 -> 16384, -0.5 -> -16384
        Case('u16_half_and_quarter', 'u16', (257, 258), (0x3F000000, 0x3E800000)),  # 0.5 -> 32768, 0.25 -> 16384
        Case('i16_neg_quarter', 'i16', (257, 258), (0xBE800000, 0x3E800000)),     # -0.25 -> -8192, 0.25 -> 8192
        Case('u16_p_lo_tie_witness', 'u16', (257, 258), (0x3F000101, 0xBF000101)),  # 32768.50389 -> 32769 (kills drop p_lo)
        Case('i16_p_lo_tie_witness', 'i16', (257, 258), (0x3F000201, 0xBF000201)),  # +-16384.50192 -> +-16385
        # Single-rounding exact product vs binary32-multiply-then-RNE witnesses:
        Case('i16_single_round_sep', 'i16', (257, 258), (0x3F000400, 0x38400180)),  # 16385 vs 16386, 1 vs 2
        Case('u16_single_round_sep', 'u16', (257, 258), (0x3F000200, 0x37C000C0)),  # 32769 vs 32770, 1 vs 2
        # 3. VOP3 abs[1:0] and neg[1:0] across both operands.
        Case('i16_abs_both', 'i16', (257, 258), (0xBF000000, 0xBE800000), abs_=0b11),
        Case('i16_neg_both', 'i16', (257, 258), (0x3F000000, 0x3F800000), neg=0b11),
        Case('u16_abs_neg_mix', 'u16', (257, 258), (0xBF000000, 0xBF000000), abs_=0b01, neg=0b10),
        # 4. SGPR, literal (255), inline f32 (240/241/242/248), inline int (128), and partial EXEC_LO.
        Case('u16_sgpr_literal', 'u16', (10, 255), (0x3F000000, 0x3F000101)),
        Case('i16_inline_f32', 'i16', (240, 243), (0, 0)),
        Case('u16_inline_248_128', 'u16', (248, 128), (0, 0)),
        Case('i16_partial_exec', 'i16', (257, 258), (0x3F800000, 0xBF000000), exec_mask=0xAAAAAAAA),
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
    code += vop3_pknorm(c.kind, 4, c.srcs, abs_=c.abs_, neg=c.neg)
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
    return [
        # clamp and omod refused.
        (vop3_pknorm('i16', 4, (257, 258), clamp=1) + [0xBF810000], 'unsupported v_cvt_pknorm_i16_f32 clamp or omod'),
        (vop3_pknorm('u16', 4, (257, 258), omod=1) + [0xBF810000], 'unsupported v_cvt_pknorm_u16_f32 clamp or omod'),
        # op_sel, third-operand abs/neg, and nonzero src2 refused.
        (vop3_pknorm('u16', 4, (257, 258), op_sel=1) + [0xBF810000], 'unsupported v_cvt_pknorm_u16_f32 operand modifiers'),
        (vop3_pknorm('i16', 4, (257, 258), op_sel=8) + [0xBF810000], 'unsupported v_cvt_pknorm_i16_f32 operand modifiers'),
        (vop3_pknorm('i16', 4, (257, 258), abs_=4) + [0xBF810000], 'unsupported v_cvt_pknorm_i16_f32 operand modifiers'),
        (vop3_pknorm('u16', 4, (257, 258), neg=4) + [0xBF810000], 'unsupported v_cvt_pknorm_u16_f32 operand modifiers'),
        (vop3_pknorm('u16', 4, (257, 258), s2=259) + [0xBF810000], 'unsupported v_cvt_pknorm_u16_f32 operand modifiers'),
        # Unsupported special operand encoding (e.g. 254).
        (vop3_pknorm('i16', 4, (254, 258)) + [0xBF810000], 'direct DXIL operand encoding'),
    ]


def cvt_norm_f32_half_mul_f32_oracle(kind, bits):
    """Alternative binary32-multiply-then-RNE oracle (separated by *_single_round_sep)."""
    bits &= 0xFFFFFFFF
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return 0
    val = struct.unpack('<f', struct.pack('<I', bits))[0]
    if kind == 'u16':
        if math.isnan(val) or val <= 0.0:
            return 0
        if val >= 1.0:
            return 0xFFFF
        scaled = Fraction(struct.unpack('<f', struct.pack('<f', val * 65535.0))[0])
    else:
        if val == 0.0:
            return 0
        if val >= 1.0:
            return 32767
        if val <= -1.0:
            return (-32767) & 0xFFFF
        scaled = Fraction(struct.unpack('<f', struct.pack('<f', val * 32767.0))[0])
    neg_sign = scaled < 0
    mag = abs(scaled)
    q = int(mag)
    rem = mag - q
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (q & 1)):
        q += 1
    return ((-q) & 0xFFFF) if neg_sign else (q & 0xFFFF)


def check_oracle_separation():
    assert cvt_norm_f32_half('i16', 0x3F000400) == 16385 and cvt_norm_f32_half_mul_f32_oracle('i16', 0x3F000400) == 16386
    assert cvt_norm_f32_half('i16', 0x38400180) == 1 and cvt_norm_f32_half_mul_f32_oracle('i16', 0x38400180) == 2
    assert cvt_norm_f32_half('u16', 0x3F000200) == 32769 and cvt_norm_f32_half_mul_f32_oracle('u16', 0x3F000200) == 32770
    assert cvt_norm_f32_half('u16', 0x37C000C0) == 1 and cvt_norm_f32_half_mul_f32_oracle('u16', 0x37C000C0) == 2


def ir(dumper):
    check_oracle_separation()
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'cvt-pknorm-f32')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_cvt_pknorm_f32 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    check_oracle_separation()
    items = all_cases()
    code = compute_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert '32767' in dump and '65535' in dump and 'lshr i32' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_cvt_pknorm_f32 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
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
    print(f'native_cvt_pknorm_f32 compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
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
        shader = work / f'cvt-pknorm-f32-{idx}.ps'
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
    print(f'native_cvt_pknorm_f32 pixel ({adapter}): {len(pairs)} pixel draws ({len(items)} cases) matched')
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
