"""Direct DXIL VOP1/VOP3 v_cvt_norm_i16_f16 and v_cvt_norm_u16_f16 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE, inline_word
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

VOP1_OPS = {'i16': 0x63, 'u16': 0x64}
VOP3_OPS = {'i16': 0x1E3, 'u16': 0x1E4}
ALL_EXEC = 0xFFFFFFFF
DST_INIT = 0xDEADCAFE


def vop1_cvt_norm(kind, dst, src0):
    return [0x7E000000 | ((dst & 0xFF) << 17) | (VOP1_OPS[kind] << 9) | (src0 & 0x1FF)]


def vop3_cvt_norm(kind, dst, src0, *, abs_=0, neg=0, op_sel=0, clamp=0, omod=0, s1=0, s2=0):
    w0 = (0xD4000000 | (VOP3_OPS[kind] << 16) | ((clamp & 1) << 15) | ((op_sel & 15) << 11) |
          ((abs_ & 7) << 8) | (dst & 0xFF))
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def eval_cvt_norm(kind, raw_word, *, abs_=0, neg=0, op_sel=0, dst_init=DST_INIT):
    signed = kind == 'i16'
    bits = (raw_word >> (16 if (op_sel & 1) else 0)) & 0xFFFF
    if abs_ & 1:
        bits &= 0x7FFF
    if neg & 1:
        bits ^= 0x8000
    sign = bits & 0x8000
    mag = bits & 0x7FFF
    if mag > 0x7C00 or mag == 0:
        res = 0
    elif not signed:
        if sign:
            res = 0
        elif mag >= 0x3C00:
            res = 0xFFFF
        else:
            exp = mag >> 10
            mant = mag if exp == 0 else (0x400 | (mag & 0x3FF))
            shift = 24 if exp == 0 else (25 - exp)
            prod = mant * 65535
            q = prod >> shift
            mid = 1 << (shift - 1)
            rem = prod & ((1 << shift) - 1)
            if rem > mid or (rem == mid and (q & 1)):
                q += 1
            res = q
    else:
        if mag >= 0x3C00:
            q = 32767
        else:
            exp = mag >> 10
            mant = mag if exp == 0 else (0x400 | (mag & 0x3FF))
            shift = 24 if exp == 0 else (25 - exp)
            prod = mant * 32767
            q = prod >> shift
            mid = 1 << (shift - 1)
            rem = prod & ((1 << shift) - 1)
            if rem > mid or (rem == mid and (q & 1)):
                q += 1
        res = ((-q) & 0xFFFF) if sign else q
    if op_sel & 8:
        return ((res & 0xFFFF) << 16) | (dst_init & 0xFFFF)
    return (dst_init & 0xFFFF0000) | (res & 0xFFFF)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    src0: int
    val0: int
    wide: bool = False
    abs_: int = 0
    neg: int = 0
    op_sel: int = 0
    exec_mask: int = ALL_EXEC

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return DST_INIT
        raw = self.val0 & 0xFFFFFFFF if (self.src0 < 128 or self.src0 >= 255) else inline_word(self.src0)
        return eval_cvt_norm(self.kind, raw, abs_=self.abs_, neg=self.neg, op_sel=self.op_sel)


def all_cases():
    return [
        # 1. Signed zero, unity, saturation (>1 / <-1 / +/-Inf), and NaN -> 0 for i16 & u16.
        Case('i16_neg_zero', 'i16', 257, 0x8000),
        Case('u16_neg_zero', 'u16', 257, 0x8000),
        Case('i16_pos_one_and_over', 'i16', 257, 0x7BFF3C00),     # 1.0 -> 32767 (hi=65504 ignored)
        Case('i16_neg_two_sat', 'i16', 257, 0xC000),              # -2.0 -> -32767 (0x8001)
        Case('u16_neg_clamp_zero', 'u16', 257, 0xBC00),           # -1.0 -> 0
        Case('u16_pos_inf_sat', 'u16', 257, 0x7C00),              # +Inf -> 65535 (0xffff)
        Case('i16_neg_inf_sat', 'i16', 257, 0xFC00),              # -Inf -> -32767 (0x8001)
        Case('u16_neg_inf_zero', 'u16', 257, 0xFC00),             # -Inf -> 0
        Case('i16_nan_zero', 'i16', 257, 0x7E01),                 # NaN -> 0
        Case('u16_neg_nan_zero', 'u16', 257, 0xFC01),             # -NaN -> 0
        # 2. Exact halfway ties and binary32 double-rounding / subnormal discriminators.
        Case('i16_half_tie_even', 'i16', 257, 0x3800),            # 0.5 * 32767 = 16383.5 -> 16384 (even)
        Case('u16_half_tie_even', 'u16', 257, 0x3800),            # 0.5 * 65535 = 32767.5 -> 32768 (even)
        Case('i16_neg_half_tie_even', 'i16', 257, 0xB800),        # -0.5 -> -16384 (0xc000)
        Case('i16_dbl_round_3801', 'i16', 257, 0x3801),           # 1025/2048 * 32767 = 16399.4997 -> 16399
        Case('u16_dbl_round_3801', 'u16', 257, 0x3801),           # 1025/2048 * 65535 = 32799.4995 -> 32799
        Case('i16_dbl_round_b802', 'i16', 257, 0xB802),           # -1026/2048 * 32767 -> -16415
        Case('u16_max_subnormal', 'u16', 257, 0x03FF),            # 0x03ff * 65535 / 2^24 -> 4
        Case('i16_neg_max_subnormal', 'i16', 257, 0x83FF),        # -0x03ff * 32767 / 2^24 -> -2 (0xfffe)
        Case('u16_lt_one_3bff', 'u16', 257, 0x3BFF),              # (1 - 2^-11) * 65535 = 65503.0005 -> 65503
        # 3. VOP3 modifiers: separate abs, neg, and high-half op_sel[0]/op_sel[3].
        Case('i16_vop3_abs', 'i16', 257, 0x7BFFB800, wide=True, abs_=1),
        Case('u16_vop3_neg', 'u16', 257, 0x7BFFB800, wide=True, neg=1),
        Case('i16_vop3_opsel_hi_src', 'i16', 257, 0x38000000, wide=True, op_sel=0b0001),
        Case('u16_vop3_opsel_hi_dst', 'u16', 257, 0x00003800, wide=True, op_sel=0b1000),
        Case('i16_vop3_opsel_hi_rw', 'i16', 257, 0xB800BEEF, wide=True, op_sel=0b1001),
        # 4. SGPR, literal (255), inline f16 float (240/248), inline int (129/193), and partial EXEC_LO.
        Case('u16_sgpr_src0', 'u16', 10, 0x3800),
        Case('i16_literal_src0', 'i16', 255, 0x3802),
        Case('u16_inline_f16_248', 'u16', 248, 0),
        Case('i16_inline_int_129', 'i16', 129, 0),
        Case('i16_inline_int_193_nan', 'i16', 193, 0),
        Case('u16_partial_exec', 'u16', 257, 0x7BFF3800, exec_mask=0xAAAAAAAA),
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
        code += vop3_cvt_norm(c.kind, 4, c.src0, abs_=c.abs_, neg=c.neg, op_sel=c.op_sel)
    else:
        code += vop1_cvt_norm(c.kind, 4, c.src0)
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
    plain = vop1_cvt_norm('i16', 4, 257) + [0xBF810000]
    return [
        # Clamp and omod refused on both forms.
        (vop3_cvt_norm('i16', 4, 257, clamp=1) + [0xBF810000], 'unsupported v_cvt_norm_i16_f16 clamp or omod'),
        (vop3_cvt_norm('u16', 4, 257, omod=1) + [0xBF810000], 'unsupported v_cvt_norm_u16_f16 clamp or omod'),
        # Unused VOP3 operand modifier / op_sel bits.
        (vop3_cvt_norm('u16', 4, 257, op_sel=0b0010) + [0xBF810000], 'unsupported v_cvt_norm_u16_f16 operand modifiers'),
        (vop3_cvt_norm('i16', 4, 257, s1=258) + [0xBF810000], 'unsupported v_cvt_norm_i16_f16 operand modifiers'),
        # High-half op_sel on non-VGPR source.
        (vop3_cvt_norm('u16', 4, 10, op_sel=0b0001) + [0xBF810000],
         'unsupported v_cvt_norm_u16_f16 high-half op_sel on non-VGPR source'),
        # DPP16 and SDWA forms.
        (vop1_cvt_norm('i16', 4, 250) + [0xFF00E401, 0xBF810000], 'unsupported DPP arithmetic'),
        (vop1_cvt_norm('u16', 4, 249) + [0x06060601, 0xBF810000], 'unimplemented direct SDWA operand profile'),
        # Special scalar source (vcc_lo).
        (vop1_cvt_norm('i16', 4, 106) + [0xBF810000], 'unsupported v_cvt_norm_i16_f16 source'),
        # Unsupported binary16 rounding/denorm modes and dynamic conditional MODE write.
        ([0xBFA40004] + plain, 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50003] + plain, 'unsupported binary16 rounding or denormal mode'),
        (cond_mode + plain, 'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'cvt-norm-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_cvt_norm_f16 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
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
    assert '32767' in dump and '65535' in dump and 'lshr i32' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_cvt_norm_f16 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
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
    print(f'native_cvt_norm_f16 compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
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
        shader = work / f'cvt-norm-f16-{idx}.ps'
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
    print(f'native_cvt_norm_f16 pixel ({adapter}): {len(pairs)} pixel draws ({len(items)} cases) matched')
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
