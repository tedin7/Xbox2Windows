"""v_rcp_iflag_f32 (VOP1, VOP3, SDWA, DPP16) in direct Compute and Pixel DXIL."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE
from native_scalar_lds_check import vmove
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fval(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def vop1(op, dst, src0):
    return [0x7E000000 | ((dst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)]


def vop3(op, dst, src0, *, abs_bit=0, neg_bit=0, clamp=0, omod=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) | ((abs_bit & 1) << 8) | (dst & 0xFF)
    w1 = ((neg_bit & 1) << 29) | ((omod & 3) << 27) | (src0 & 0x1FF)
    return [w0, w1]


def sdwa(op, dst, src_reg, *, src_scalar=0, abs_bit=0, neg_bit=0, clamp=0, omod=0,
         src_sel=6, dst_sel=6, dst_unused=0):
    w0 = 0x7E000000 | ((dst & 0xFF) << 17) | ((op & 0xFF) << 9) | 249
    ext = (((src_scalar & 1) << 23) | ((abs_bit & 1) << 21) | ((neg_bit & 1) << 20) |
           ((src_sel & 7) << 16) | ((omod & 3) << 14) | ((clamp & 1) << 13) |
           ((dst_unused & 3) << 11) | ((dst_sel & 7) << 8) | (src_reg & 0xFF))
    return [w0, ext]


def dpp(op, dst, src_vgpr, *, pattern=0xB1, row_mask=0xF, bank_mask=0xF):
    w0 = 0x7E000000 | ((dst & 0xFF) << 17) | ((op & 0xFF) << 9) | 250
    ctrl = ((row_mask & 0xF) << 28) | ((bank_mask & 0xF) << 24) | ((pattern & 0x1FF) << 8) | (src_vgpr & 0xFF)
    return [w0, ctrl]


def f32_inline(enc):
    return {240: 0x3F000000, 241: 0xBF000000, 242: 0x3F800000, 243: 0xBF800000,
            244: 0x40000000, 245: 0xC0000000, 246: 0x40800000, 247: 0xC0800000}[enc]


@dataclass(frozen=True)
class Case:
    name: str
    form: str
    src: int
    val: int
    abs_bit: int = 0
    neg_bit: int = 0
    clamp: int = 0
    omod: int = 0
    pixel_safe: bool = True

    def expected(self):
        raw = f32_inline(self.src) if 240 <= self.src <= 248 else self.val
        if self.abs_bit:
            raw &= 0x7FFFFFFF
        if self.neg_bit:
            raw ^= 0x80000000
        r = 1.0 / fval(raw)
        if self.omod == 1:
            r *= 2.0
        elif self.omod == 2:
            r *= 4.0
        elif self.omod == 3:
            r *= 0.5
        if self.clamp:
            r = 0.0 if r <= 0.0 else (1.0 if r >= 1.0 else r)
        return fbits(r)


def all_cases():
    return [
        # VOP1: VGPR, SGPR, inline floats, literal.
        Case('vop1_vgpr_two', 'vop1', 257, fbits(2.0)),
        Case('vop1_vgpr_neg_four', 'vop1', 257, fbits(-4.0)),
        Case('vop1_sgpr_eight', 'vop1', 10, fbits(8.0)),
        Case('vop1_inline_240', 'vop1', 240, 0),
        Case('vop1_inline_245', 'vop1', 245, 0),
        Case('vop1_literal_eighth', 'vop1', 255, fbits(0.125)),
        # VOP3: abs, neg, clamp, omod (*2, *4, *0.5).
        Case('vop3_abs', 'vop3', 257, fbits(-2.0), abs_bit=1),
        Case('vop3_neg', 'vop3', 257, fbits(4.0), neg_bit=1),
        Case('vop3_clamp_hi', 'vop3', 257, fbits(0.25), clamp=1),
        Case('vop3_clamp_lo', 'vop3', 257, fbits(-2.0), clamp=1),
        Case('vop3_omod_2x', 'vop3', 257, fbits(4.0), omod=1),
        Case('vop3_omod_4x', 'vop3', 257, fbits(8.0), omod=2),
        Case('vop3_omod_half', 'vop3', 257, fbits(0.5), omod=3),
        # SDWA: VGPR & SGPR DWORD with abs/neg/clamp.
        Case('sdwa_vgpr_abs_neg', 'sdwa', 257, fbits(2.0), abs_bit=1, neg_bit=1),
        Case('sdwa_sgpr_clamp', 'sdwa', 10, fbits(0.5), clamp=1),
        # DPP16: quad_perm:[1,0,3,2] (lane 0 reads lane 1 = 0.25 -> 4.0) across compute lanes.
        Case('dpp_quad_perm', 'dpp', 257, fbits(0.25), pixel_safe=False),
    ]


def emit_case(c: Case, *, use_plain_rcp=False):
    op_vop1 = 0x2A if use_plain_rcp else 0x2B
    op_vop3 = 0x1AA if use_plain_rcp else 0x1AB
    code = []
    if c.form == 'dpp':
        code += F.mov(1, c.val)
        code += dpp(op_vop1, 4, 1)
        return code
    if c.src >= 256:
        code += F.mov(c.src - 256, c.val)
    elif c.src <= 105:
        code += F.smov(c.src, c.val)
    if c.form == 'vop1':
        code += vop1(op_vop1, 4, c.src)
        if c.src == 255:
            code.append(c.val)
    elif c.form == 'vop3':
        code += vop3(op_vop3, 4, c.src, abs_bit=c.abs_bit, neg_bit=c.neg_bit, clamp=c.clamp, omod=c.omod)
    elif c.form == 'sdwa':
        scalar = 1 if c.src <= 105 else 0
        reg = c.src if scalar else (c.src - 256)
        code += sdwa(op_vop1, 4, reg, src_scalar=scalar, abs_bit=c.abs_bit, neg_bit=c.neg_bit, clamp=c.clamp)
    else:
        raise ValueError(c.form)
    return code


def compute_program(cases, *, use_plain_rcp=False):
    code = list(PROLOGUE)
    want = []
    for slot, c in enumerate(cases):
        code += emit_case(c, use_plain_rcp=use_plain_rcp) + F.store(slot * 4, 4)
        want.append(c.expected())
    return code + [0xBF810000], want


def refusal_programs():
    return [
        # SDWA invalid dst_select=7 refused by SDWA profile validation.
        (sdwa(0x2B, 4, 1, dst_sel=7) + [0xBF810000], 'unsupported SDWA destination modifier'),
        # DPP16 partial row mask refused.
        (dpp(0x2B, 4, 1, row_mask=0x5) + [0xBF810000], 'DPP requires a full-row, full-bank mask'),
    ]


def controls(cli, sdk, work, root):
    items = all_cases()
    code, want = compute_program(items)
    ctrl_code, _ = compute_program(items, use_plain_rcp=True)
    m_iflag = request(work, root, code)
    m_plain = request(work, root, ctrl_code)
    o_iflag = work / 'iflag.cso'
    o_plain = work / 'plain.cso'
    for m, o in ((m_iflag, o_iflag), (m_plain, o_plain)):
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode == 0 and o.is_file(), (r.stdout, r.stderr)
    d_iflag = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', o_iflag], capture_output=True, text=True, check=True).stdout
    d_plain = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', o_plain], capture_output=True, text=True, check=True).stdout
    assert 'fdiv float 1.000000e+00,' in d_iflag and d_iflag == d_plain, 'v_rcp_iflag_f32 DXIL mismatch vs v_rcp_f32'

    # Also verify Pixel stage compilation equivalence.
    ps_items = [c for c in items if c.pixel_safe][:4]
    ps_iflag = work / 'ps_iflag.ps'
    ps_plain = work / 'ps_plain.ps'
    ps_iflag.write_bytes(pixel_program(ps_items, use_plain_rcp=False))
    ps_plain.write_bytes(pixel_program(ps_items, use_plain_rcp=True))
    for src, cso in ((ps_iflag, ps_iflag.with_suffix('.cso')), (ps_plain, ps_plain.with_suffix('.cso'))):
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', src, cso], capture_output=True, text=True)
        assert r.returncode == 0 and cso.is_file(), (r.stdout, r.stderr)
    dp_iflag = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', ps_iflag.with_suffix('.cso')],
                              capture_output=True, text=True, check=True).stdout
    dp_plain = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', ps_plain.with_suffix('.cso')],
                              capture_output=True, text=True, check=True).stdout
    assert dp_iflag == dp_plain, 'Pixel DXIL mismatch vs v_rcp_f32'

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = work / f'refused-{idx}.cso'
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_rcp_iflag controls: {len(want)} Compute + {len(ps_items)} Pixel cases matched v_rcp_f32 DXIL; {len(refusals)} refusals matched')
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
    print(f'native_rcp_iflag compute hardware: {len(want)} cases matched across {F.GROUPS}x32 lanes and guards')
    return 0


def pixel_program(quad, *, use_plain_rcp=False):
    code = []
    for idx, c in enumerate(quad):
        code += emit_case(c, use_plain_rcp=use_plain_rcp)
        code.append(vmove(8 + idx, 256 + 4))
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def pixel(cli, sdk, renderer, adapter, work):
    items = [c for c in all_cases() if c.pixel_safe]
    while len(items) % 4:
        items.append(items[-1])
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    quads = [items[i:i + 4] for i in range(0, len(items), 4)]
    for idx, quad in enumerate(quads):
        shader = work / f'rcp-{idx}.ps'
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
    print(f'native_rcp_iflag pixel ({adapter}): {len(quads)} quad draws matched')
    return 0


def main(mode, tool, sdk=None, *args):
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
