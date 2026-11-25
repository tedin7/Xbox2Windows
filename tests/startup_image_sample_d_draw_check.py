"""image_sample_d drawn on D3D12 (WARP and hardware), GPU readback against a CPU reference (Part of #1723).

  (no arguments)                       build the programs, translate them, check the CPU reference
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw against the
                                       4x4 two-slice, three-mip R32G32B32A32_FLOAT fixture of startup_image_3d.cpp

One pixel stage samples four times with explicit gradients. The gradients, not the coordinates, pick the mip level:
rho = largest gradient component * 4 texels, so 0.25 -> mip 0, 0.5 -> mip 1, 1.0 -> mip 2 (exact powers of two, no
rounding ambiguity). Probes use ddx only, ddy only, a gradient on the second axis and the second array slice.
The expected colours come from the fixture formula red = (1 + x + 4y + 16s + 32m) / 128 evaluated here with
exact fractions. Mutants: a sampler capped to MAX_LOD 0 and SampleGrad replaced by an implicit Sample must both
return mip 0 on every probe, which the same draw must report as different from the expected colours.
"""
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import (  # noqa: E402
    FULLSCREEN_VS,
    VS_WORDS,
    CompileQueue,
    compile_hlsl,
    root_blob,
)
from startup_image_sample_c_border_check import (  # noqa: E402
    DIM_2D_ARRAY,
    EXP_MRT0_V4_V7,
    INLINE_FLOATS,
    S_ENDPGM,
    S_LOAD_DWORDX8,
    S_WAIT_LGKM,
    S_WAIT_VM,
    f32_hex,
    manifest_sampler,
    mimg_seq,
    s_mov_b32,
    v_mov_inline,
    v_mov_lit,
)

IMG_SAMPLE_D = 0x22
SIZE = 4
LEVELS = 3
# Point filters, mip filter point (word 2 bits 27:26 = 1), MIN_LOD 0 and MAX_LOD 2.0 (u4.8, word 1 bits 23:12).
SAMPLER_WORDS = (0, 2 * 256 << 12, 0x04000000, 0)
CAPPED_WORDS = (0, 0, 0x04000000, 0)
ADDR = 10  # v10..v16: ddx, ddy, coordinates, slice; v4..v7 receive the four probes

# (u, v, slice, ddx, ddy)
PROBES = (
    (Fraction(5, 8), Fraction(3, 8), 0, (Fraction(1, 4), 0), (0, Fraction(1, 4))),
    (Fraction(5, 8), Fraction(3, 8), 0, (Fraction(1, 2), 0), (0, 0)),
    (Fraction(1, 8), Fraction(7, 8), 0, (0, 0), (0, 1)),
    (Fraction(7, 8), Fraction(5, 8), 1, (0, Fraction(1, 2)), (0, 0)),
)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def mov_value(vdst, value):
    raw = bits(value)
    code = 0x80 if raw == 0 else INLINE_FLOATS.get(raw)
    asm = 'v_mov_b32 v%d, %s' % (vdst, '0' if raw == 0 else '%#x' % raw)
    return asm, (v_mov_inline(vdst, code) if code is not None else v_mov_lit(vdst, raw))


def build_program(words):
    lines = ['s_load_dwordx8 s[16:23], s[8:9], 0x0']
    code = bytearray(S_LOAD_DWORDX8)
    for reg, val in zip((24, 25, 26, 27), words):
        lines.append('s_mov_b32 s%d, %s' % (reg, '0' if val == 0 else '%#x' % val))
        code += s_mov_b32(reg, val)
    lines.append('s_waitcnt lgkmcnt(0)')
    code += S_WAIT_LGKM
    for i, (u, v, slice_, ddx, ddy) in enumerate(PROBES):
        for k, value in enumerate((*ddx, *ddy, u, v, slice_)):
            asm, raw = mov_value(ADDR + k, value)
            lines.append(asm)
            code += raw
        lines.append('image_sample_d v%d, v[%d:%d], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY'
                     % (4 + i, ADDR, ADDR + 6))
        code += mimg_seq(IMG_SAMPLE_D, 4 + i, ADDR, dmask=0x1, dim=DIM_2D_ARRAY)
    lines += ['s_waitcnt vmcnt(0)', 'exp mrt0 v4, v5, v6, v7 done vm', 's_endpgm']
    code += S_WAIT_VM + EXP_MRT0_V4_V7 + S_ENDPGM
    return '\n'.join(lines) + '\n', bytes(code)


def texel_red(x, y, s, m):
    return Fraction(1 + x + 4 * y + 16 * s + 32 * m, 128)


def reference(max_lod=LEVELS - 1):
    """Red channel of every probe: the level follows the largest gradient component, texel = floor(coord * size)."""
    out = []
    for u, v, s, ddx, ddy in PROBES:
        rho = max(abs(Fraction(c)) for c in (*ddx, *ddy)) * SIZE
        m = 0
        while Fraction(2) ** (m + 1) <= rho and m < max_lod:
            m += 1
        dim = SIZE >> m
        out.append(texel_red(int(u * dim), int(v * dim), s, m))
    return out


def pipeline(code):
    return G.Pipeline(VS_WORDS, code, root_blob())


def check_disassembly(code):
    off, decoded = 0, []
    while off < len(code):
        size, mnemonic, asm, _ = D.decode_instruction(code, off)
        assert size > 0 and 'unknown' not in mnemonic, (off, mnemonic, asm)
        decoded.append(asm)
        off += size
    assert off == len(code) and decoded[-1] == 's_endpgm'
    assert sum(a.startswith('image_sample_d') for a in decoded) == len(PROBES), decoded


def controls():
    _src, code = build_program(SAMPLER_WORDS)
    check_disassembly(code)
    hlsl = pipeline(code).hlsl_ps()
    assert hlsl.count('SampleGrad(') == len(PROBES), hlsl
    expected = reference()
    # Mip 1 and 2 are reached, and every probe differs from its mip 0 value: the draw can tell the levels apart.
    assert expected == [Fraction(7, 128), Fraction(34, 128), Fraction(65, 128), Fraction(54, 128)], expected
    assert all(a != b for a, b in list(zip(expected, reference(0)))[1:]) and expected[0] == reference(0)[0]
    assert 'smpl ' in pipeline(code).manifest()
    print('four SampleGrad probes translate; reference reaches mips 0, 1, 2 and slice 1')


def run_draw(renderer, adapter, vs, ps, expected, sampler):
    cmd = [renderer, adapter, '--startup-image-2d-array-mips', str(vs), str(ps), '0', expected, sampler]
    return subprocess.run(cmd, capture_output=True, text=True)


def drop_gradients(hlsl):
    """SampleGrad(smp, uv, ddx, ddy) -> Sample(smp, uv): split the arguments at top-level commas."""
    out, pos = [], 0
    while True:
        at = hlsl.find('.SampleGrad(', pos)
        if at < 0:
            out.append(hlsl[pos:])
            return ''.join(out)
        out.append(hlsl[pos:at])
        i = at + len('.SampleGrad(')
        depth, args, cur = 0, [], ''
        while True:
            ch = hlsl[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth == 0:
                    break
                depth -= 1
            if ch == ',' and depth == 0:
                args.append(cur.strip())
                cur = ''
            else:
                cur += ch
            i += 1
        args.append(cur.strip())
        assert len(args) == 4, args
        out.append('.Sample(%s, %s' % (args[0], args[1]))
        pos = i


def check_draw(dxc, renderer, adapter, fxc=None):
    _src, code = build_program(SAMPLER_WORDS)
    pipe = pipeline(code)
    sampler = manifest_sampler(pipe)
    capped = manifest_sampler(pipeline(build_program(CAPPED_WORDS)[1]))
    hlsl = pipe.hlsl_ps()
    implicit = drop_gradients(hlsl)
    assert implicit != hlsl and '.SampleGrad(' not in implicit and implicit.count('.Sample(') == len(PROBES)
    want = ','.join(f32_hex(float(r)) for r in reference())
    mip0 = ','.join(f32_hex(float(r)) for r in reference(0))
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp, CompileQueue() as queue:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = queue.call(compile_hlsl, compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            grad = queue.call(compile_hlsl, compiler, kind, hlsl, 'grad_%s' % kind, ps_profile, directory)
            run = run_draw(renderer, adapter, vs, grad, want, sampler)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                sys.exit(77)
            if run.returncode != 0:
                raise AssertionError('%s image_sample_d draw failed:\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s image_sample_d: %s' % (kind, want), flush=True)

            capped_run = run_draw(renderer, adapter, vs, grad, want, capped)
            if capped_run.returncode == 0 or ('actual=' + mip0) not in capped_run.stderr:
                raise AssertionError('%s capped-sampler mutant did not produce mip 0:\n%s\n%s'
                                     % (kind, capped_run.stdout, capped_run.stderr))
            print('%s mutant MAX_LOD 0 caught' % kind, flush=True)

            flat = queue.call(compile_hlsl, compiler, kind, implicit, 'flat_%s' % kind, ps_profile, directory)
            flat_run = run_draw(renderer, adapter, vs, flat, want, sampler)
            if flat_run.returncode == 0 or ('actual=' + mip0) not in flat_run.stderr:
                raise AssertionError('%s implicit-Sample mutant did not produce mip 0:\n%s\n%s'
                                     % (kind, flat_run.stdout, flat_run.stderr))
            print('%s mutant implicit Sample caught' % kind, flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
