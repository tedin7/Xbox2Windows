"""v_madmk_f32 and v_madak_f32 in the direct C++ Compute and Pixel DXIL paths."""
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
from native_packed_f16_inline_check import PROLOGUE
from native_scalar_lds_check import vmove
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OP_CODE = {'madmk': 0x20, 'madak': 0x21}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fval(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def ftz(b):
    b &= 0xFFFFFFFF
    return b & 0x80000000 if (b & 0x7FFFFFFF) < 0x00800000 else b


def mad_ftz(a, b, c, *, fused=False, flush=True):
    if flush:
        a, b, c = ftz(a), ftz(b), ftz(c)
    x, y, z = fval(a), fval(b), fval(c)
    if (math.isinf(x) and y == 0.0) or (math.isinf(y) and x == 0.0) or math.isnan(x) or math.isnan(y):
        product = math.nan
    else:
        product = x * y
    if fused:
        total = product + z if not math.isnan(product) else math.nan
        out = 0x7FC00000 if math.isnan(total) else fbits(total)
    else:
        p = 0x7FC00000 if math.isnan(product) else (ftz(fbits(product)) if flush else fbits(product))
        q = fval(p)
        total = q + z if not math.isnan(q) else math.nan
        out = 0x7FC00000 if math.isnan(total) else fbits(total)
    return ftz(out) if flush else out


def is_nan_f32(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def same_f32(got, want):
    if is_nan_f32(want):
        return is_nan_f32(got)
    return got == want


def vop2_k(op, dst, src0, vsrc1, lit):
    w0 = (OP_CODE[op] << 25) | ((dst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    return [w0, lit & 0xFFFFFFFF]


def f32_inline(enc):
    inline_f = {
        240: 0x3F000000, 241: 0xBF000000, 242: 0x3F800000, 243: 0xBF800000,
        244: 0x40000000, 245: 0xC0000000, 246: 0x40800000, 247: 0xC0800000,
    }
    if enc in inline_f:
        return inline_f[enc]
    if 128 <= enc <= 192:
        return enc - 128
    raise ValueError(enc)


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    src0_enc: int
    src0_val: int
    vsrc1_val: int
    k_val: int
    pixel_safe: bool = True

    def operands(self):
        a = f32_inline(self.src0_enc) if 128 <= self.src0_enc < 255 else self.src0_val
        if self.op == 'madmk':
            return a, self.k_val, self.vsrc1_val
        return a, self.vsrc1_val, self.k_val

    def expected(self, **mutation):
        return mad_ftz(*self.operands(), **mutation)


def all_cases():
    return [
        # 1. Distinct K vs S1 argument positions.
        Case('madmk_asymmetric', 'madmk', 257, fbits(3.0), fbits(1.25), fbits(-2.5)),
        Case('madak_asymmetric', 'madak', 257, fbits(3.0), fbits(-2.5), fbits(1.25)),
        # 2. Unfused product rounding ties: fused FMA differs by 1 ULP.
        Case('madmk_unfused_tie_sub', 'madmk', 257, 0x3F800800, fbits(-1.0), 0x3F800800),
        Case('madak_unfused_tie_add', 'madak', 257, 0x3F800001, 0x3F800001, 0x3F800003),
        # 3. Denormal input flushes (src0, literal K multiplier in madmk, literal K addend in madak).
        Case('madmk_ftz_src0', 'madmk', 257, 0x00000001, fbits(1.0), 0x7E800000),
        Case('madmk_ftz_lit_k', 'madmk', 257, 0x7E800000, fbits(2.0), 0x80400000),
        Case('madak_ftz_lit_k', 'madak', 257, 0x00800000, fbits(1.0), 0x00400000),
        # 4. Denormal product flush and denormal sum flush.
        Case('madmk_ftz_product', 'madmk', 257, 0x00800000, 0x00800000, fbits(0.5)),
        Case('madak_ftz_sum', 'madak', 257, fbits(1.5 * 2.0 ** -63), fbits(2.0 ** -63), 0x80800000, pixel_safe=False),
        # 5. SGPR src0 and inline float src0 (242=1.0, 240=0.5).
        Case('madmk_sgpr_src0', 'madmk', 10, fbits(2.5), fbits(-0.75), fbits(4.0)),
        Case('madak_inline_242', 'madak', 242, 0, fbits(3.5), fbits(-1.5)),
        Case('madmk_inline_240', 'madmk', 240, 0, fbits(0.25), fbits(5.0)),
        # 6. Inf/zero -> NaN and negative-zero preservation.
        Case('madmk_zero_inf_nan', 'madmk', 257, fbits(0.0), fbits(2.0), 0x7F800000, pixel_safe=False),
        Case('madak_neg_zero', 'madak', 257, 0x80000000, fbits(1.0), 0x80000000, pixel_safe=False),
    ]


def emit_case(c: Case):
    code = []
    if c.src0_enc >= 256:
        code += F.mov(c.src0_enc - 256, c.src0_val)
    elif c.src0_enc <= 105:
        code += F.smov(c.src0_enc, c.src0_val)
    code += F.mov(2, c.vsrc1_val)
    code += vop2_k(c.op, 4, c.src0_enc, 2, c.k_val)
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
        # Non-VOP2 (VOP3 e64) forms of v_madmk_f32 / v_madak_f32 refused.
        ([0xD5200004, 0x00020501, 0xBF810000], 'unsupported v_madmk_f32 encoding'),
        ([0xD5210004, 0x00020501, 0xBF810000], 'unsupported v_madak_f32 encoding'),
        # SDWA and DPP forms remain refused.
        ([0x400804F9, 0x06060601, 0xBF810000], 'unimplemented direct SDWA operand profile'),
        ([0x420804FA, 0xFF001B01, 0xBF810000], 'SDWA or DPP vector operands'),
    ]


def check_reference_mutations(items):
    baseline = [c.expected() for c in items]
    for label, mutation in (('fused', {'fused': True}), ('no_flush', {'flush': False})):
        mutated = [c.expected(**mutation) for c in items]
        assert mutated != baseline, label
    swapped = [mad_ftz(c.operands()[0], c.operands()[2], c.operands()[1]) for c in items]
    assert swapped != baseline, 'swapped_k_s1'


def ir(dumper):
    items = all_cases()
    check_reference_mutations(items)
    code, want = compute_program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'mad32-k')
        assert run.returncode == 0, run.stderr
        assert '@dx.op.tertiary.f32(i32 46,' not in run.stdout, run.stdout
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if not same_f32(got, exp)]
        assert not bad, bad[:8]
        for idx, (prog, why) in enumerate(refusals):
            r = F.dump(dumper, work, prog, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_mad32_k IR: {len(want)} cases matched unfused FTZ model; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = all_cases()
    check_reference_mutations(items)
    code, want = compute_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert dump.count('fmul') >= len(items) and dump.count('fadd') >= len(items), dump
    assert '@dx.op.tertiary.f32(i32 46,' not in dump, dump
    assert ' fast ' not in dump and 'contract' not in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_mad32_k controls: compiled {len(want)} cases with separate fmul/fadd; verified {len(refusals)} refusals')
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
    print(f'native_mad32_k compute hardware: {len(want)} cases matched across {F.GROUPS}x32 lanes and guards')
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
        shader = work / f'madk-{idx}.ps'
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
    print(f'native_mad32_k pixel ({adapter}): {len(quads)} quad draws matched')
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
