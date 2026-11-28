"""Direct DXIL VOP2 v_dot2c_f32_f16 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dot2_f32_f16_check import bits_to_f32, f32_to_bits, p16, same_f32, widen_f16
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE, inline_word
from native_scalar_lds_check import sop1, vmove
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OP_DOT2C_F32_F16 = 0x02
ALL_EXEC = 0xFFFFFFFF


def vop2_dot2c(dst, src0, vsrc1):
    assert 256 <= vsrc1 < 512, vsrc1
    return ((OP_DOT2C_F32_F16 << 25) | ((dst & 0xFF) << 17) | (((vsrc1 - 256) & 0xFF) << 9) | (src0 & 0x1FF))


def vop3_dot2c_e64(dst, src0, vsrc1, *, clamp=0):
    return [
        0xD5020000 | ((clamp & 1) << 15) | (dst & 0xFF),
        ((vsrc1 & 0x1FF) << 9) | (src0 & 0x1FF),
    ]


def eval_dot2c(s0_word, s1_word, acc_bits):
    a0 = widen_f16(s0_word & 0xFFFF)
    b0 = widen_f16(s1_word & 0xFFFF)
    a1 = widen_f16((s0_word >> 16) & 0xFFFF)
    b1 = widen_f16((s1_word >> 16) & 0xFFFF)
    p0 = bits_to_f32(f32_to_bits(a0 * b0))
    p1 = bits_to_f32(f32_to_bits(a1 * b1))
    s01 = bits_to_f32(f32_to_bits(p0 + p1))
    c = bits_to_f32(acc_bits)
    return f32_to_bits(s01 + c)


@dataclass(frozen=True)
class Case:
    name: str
    steps: tuple
    acc: int
    exec_mask: int = ALL_EXEC
    alias: str = 'none'
    pixel_safe: bool = True

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return self.acc & 0xFFFFFFFF
        cur = self.acc & 0xFFFFFFFF
        for s0, v0, v1 in self.steps:
            if self.alias == 'src0':
                w0 = cur
            elif s0 < 128 or s0 >= 255:
                w0 = v0 & 0xFFFFFFFF
            else:
                w0 = inline_word(s0)
            w1 = cur if self.alias == 'src1' else (v1 & 0xFFFFFFFF)
            cur = eval_dot2c(w0, w1, cur)
        return cur


def all_cases():
    return [
        # 1. Basic VGPR accumulate with distinct low/high products.
        Case('vgpr_basic', ((262, p16(0x3C00, 0x4000), p16(0x4200, 0x4400)),), f32_to_bits(0.5)),
        # 2. Opposite products cancel to +0.0, preserving binary32 accumulator.
        Case('vgpr_cancel_to_acc', ((262, p16(0x3C00, 0x4000), p16(0xC000, 0x3C00)),), f32_to_bits(1.25)),
        # 3. Back-to-back accumulation in the same destination VGPR.
        Case('multi_accumulate',
             ((262, p16(0x3C00, 0x3800), p16(0x4000, 0x4000)),
              (262, p16(0xBC00, 0x4000), p16(0x3C00, 0x4200))),
             f32_to_bits(0.25)),
        # 4. Destination VGPR aliased with src0 or vsrc1 (read both as f16x2 input and f32 acc).
        Case('vgpr_dst_alias_src0', ((260, 0, p16(0x4000, 0x4200)),), p16(0x0000, 0x3C00), alias='src0'),
        Case('vgpr_dst_alias_src1', ((262, p16(0x4200, 0x3C00), 0),), p16(0x3800, 0x4000), alias='src1'),
        # 5. SGPR and 32-bit literal src0 encodings.
        Case('sgpr_src0', ((10, p16(0x3C00, 0xC000), p16(0x4200, 0x3800)),), f32_to_bits(0.75)),
        Case('literal_src0', ((255, p16(0x4000, 0x3C00), p16(0x3800, 0x4200)),), f32_to_bits(-1.0)),
        # 6. Inline binary16 float constants (0.5, 1.0, -1.0) and inline integer 0.
        Case('inline_f16_240', ((240, 0, p16(0x4000, 0x4400)),), f32_to_bits(1.5)),
        Case('inline_f16_242', ((242, 0, p16(0x4000, 0x4400)),), f32_to_bits(2.0)),
        Case('inline_f16_243', ((243, 0, p16(0x3C00, 0x4000)),), f32_to_bits(3.0)),
        Case('inline_int_128', ((128, 0, p16(0x4000, 0x4400)),), f32_to_bits(-2.5)),
        # 7. Binary16 subnormal operands (2^-24 and max subnormal).
        Case('half_subnormal_src0', ((262, p16(0x0001, 0x03FF), p16(0x0001, 0x7BFF)),), f32_to_bits(0.0)),
        # 8. Binary32 subnormal accumulators: preserved when sum == 0, flushed/rounded under nonzero sum.
        Case('f32_subnormal_acc_pos', ((262, p16(0x0000, 0x0000), p16(0x0000, 0x0000)),), 0x00000001),
        Case('f32_subnormal_acc_neg', ((262, p16(0x0000, 0x0000), p16(0x0000, 0x0000)),), 0x80400000),
        Case('f32_subnormal_acc_nonzero_dot', ((262, p16(0x3C00, 0x0000), p16(0x3C00, 0x0000)),), 0x00000001),
        # 9. Partial EXEC mask: odd lanes update vdst, even lanes preserve pre-existing vdst accumulator.
        Case('partial_exec_odd', ((262, p16(0x3C00, 0x4000), p16(0x4000, 0x4200)),),
             0x40A00000, exec_mask=0xAAAAAAAA, pixel_safe=False),
        # 10. Signed zero, infinity, and NaN propagation.
        Case('signed_zero_keep', ((262, p16(0x8000, 0x8000), p16(0x3C00, 0x3C00)),),
             f32_to_bits(-0.0), pixel_safe=False),
        Case('inf_propagation', ((262, p16(0x7C00, 0x3C00), p16(0x3C00, 0x3C00)),),
             f32_to_bits(-4.0), pixel_safe=False),
        Case('nan_propagation', ((262, p16(0x7E01, 0x3C00), p16(0x3C00, 0x3C00)),),
             f32_to_bits(1.0), pixel_safe=False),
    ]


def emit_case(c: Case):
    code = F.mov(4, c.acc)
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 255), c.exec_mask]
    for s0, v0, v1 in c.steps:
        lit_tail = []
        vsrc1 = 260 if c.alias == 'src1' else 261
        if c.alias != 'src1':
            code += F.mov(vsrc1 - 256, v1)
        src0 = 260 if c.alias == 'src0' else s0
        if c.alias != 'src0':
            if src0 >= 256:
                code += F.mov(src0 - 256, v0)
            elif src0 <= 105:
                code += F.smov(src0, v0)
            elif src0 == 255:
                lit_tail = [v0]
        code.append(vop2_dot2c(4, src0, vsrc1))
        code += lit_tail
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 193)]
    return code


def compute_program(cases):
    code = list(PROLOGUE)
    for slot, c in enumerate(cases):
        code += emit_case(c) + F.store(slot * 4, 4)
    return code + [0xBF810000]


def refusal_programs():
    cond_mode = [0xBF840001, 0xBFA5000F]  # s_cbranch_scc0 +1; s_denorm_mode 15
    plain = [vop2_dot2c(4, 262, 261), 0xBF810000]
    return [
        # VOP3 _e64 form of v_dot2c_f32_f16.
        (vop3_dot2c_e64(4, 262, 261) + [0xBF810000], 'unsupported v_dot2c_f32_f16 encoding'),
        # DPP16, DPP8, and SDWA forms.
        ([vop2_dot2c(4, 250, 261), 0xFF00E406, 0xBF810000], 'unsupported DPP arithmetic'),
        ([vop2_dot2c(4, 233, 261), 0xC6886806, 0xBF810000], 'unsupported DPP arithmetic'),
        ([vop2_dot2c(4, 249, 261), 0x06060606, 0xBF810000], 'unimplemented direct SDWA operand profile'),
        # Special scalar sources refused on src0 (vcc_lo).
        ([vop2_dot2c(4, 106, 261), 0xBF810000], 'unsupported v_dot2c_f32_f16 source'),
        # Unsupported binary16 rounding/denorm, binary32 denorm flush, and conditional MODE write.
        ([0xBFA40004] + plain, 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50003] + plain, 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA5000C] + plain, 'unsupported compute control instruction s_denorm_mode'),
        (cond_mode + plain, 'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'dot2c-f32-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if not same_f32(got, exp)]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_dot2c_f32_f16 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
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
    assert 'legacyF16ToF32' in dump and 'fmul float' in dump and 'fadd float' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_dot2c_f32_f16 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
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
            if not same_f32(got, expected[i]):
                bad.append((i, hex(got), hex(expected[i])))
        elif got != F.SENTINEL:
            bad.append((i, hex(got), hex(F.SENTINEL)))
    assert not bad, bad[:10]
    print(f'native_dot2c_f32_f16 compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
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
        shader = work / f'dot2c-{idx}.ps'
        shader.write_bytes(pixel_program(quad))
        cso = shader.with_suffix('.cso')
        build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                               capture_output=True, text=True, timeout=30)
        assert build.returncode == 0, ([c.name for c in quad], build.stdout, build.stderr)
        ref = work / 'expected.bin'
        ref.write_bytes(struct.pack('<4I', *(c.expected(lane=0) for c in quad)))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', ref],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            return 77
        assert run.returncode == 0, (idx, [c.name for c in quad], run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'native_dot2c_f32_f16 pixel ({adapter}): {len(quads)} quad draws matched')
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
