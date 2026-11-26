"""image_get_lod past a branch or an EXEC change in tools/startup/gfx2hlsl.py (implicit derivatives of the quad).

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given), draw each accepted
                                       fixture and compare the readback with the CPU reference; each mutant must differ

Semantics. image_get_lod returns the LOD a sample would use, clamped, and the LOD before the clamp, from the same quad
derivatives as image_sample_b; the lowering is CalculateLevelOfDetail / CalculateLevelOfDetailUnclamped and the
control-flow proof is Stage.uniform_cf with the quad vote of EXEC branches (see tests/startup_image_sample_b_flow_check.py
for the model: a lane EXEC masked off offers the address it last wrote).

Fixtures (synthetic gfx1013 wave64 programs, no game data). v0 is the packed pixel position, x = v0 & 0xffff. Every
program starts with `s_wqm_b64 exec, exec`, sets u = 0.125 + 0.5 * x and v = 0.25 and queries the LOD of the 4x4,
three-mip 2D texture of tests/startup_image_3d.cpp (--startup-image-2d-view 0/3: one covered pixel at (0, 0), three
helper lanes) through a sampler with mip filter point, MIN_LOD 0 and MAX_LOD 2. The unclamped LOD is
log2(4 * |u(x=1) - u(x=0)|); the clamped one is that in [0, 2]. The export is (clamped, unclamped, 0, 0):
  join_pixel   lanes with x == 0 branch into u -= 0.5, the quad queries at the join: |du| = 1, LOD 2
  join_helper  the helper column x == 1 branches into u += 1.5: |du| = 2, unclamped 3, clamped 2
  cmpx         v_cmpx keeps x == 0, u -= 0.5 there only, the query runs with that EXEC: stale helper u, LOD 2
  scc          a branch on root constant 1 (the same in every lane, not taken) before the query: LOD 1
  inside       the query inside the s_cbranch_execz block of x == 0 after u -= 0.5 there (EXEC branches are decided
               per quad, see tests/startup_image_sample_b_flow_check.py): stale helper u, LOD 2
  kill         v_cmpx keeps x == 0, s_cbranch_execnz past an s_endpgm, u -= 0.5, query: LOD 2
Refused: the query inside a block a VCC branch skips.
"""
import math
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402
from startup_image_sample_b_flow_check import PS_INPUTS, assemble  # noqa: E402

PROLOGUE = """\
s_wqm_b64 exec, exec
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0x200000
s_mov_b32 s26, 0x4000000
s_mov_b32 s27, 0
v_and_b32_e32 v8, 0xffff, v0
v_cvt_f32_u32_e32 v9, v8
v_mul_f32_e32 v9, 0.5, v9
v_add_f32_e32 v2, 0x3e000000, v9
v_mov_b32_e32 v3, 0x3e800000
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
"""
QUERY = """\
s_waitcnt lgkmcnt(0)
image_get_lod v[10:11], v[2:3], s[16:23], s[24:27] dmask:0x3 dim:SQ_RSRC_IMG_2D
"""
EXPORT = """\
s_waitcnt vmcnt(0)
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
"""


def branch_on_x(x, delta):
    return ('v_cmp_eq_u32_e32 vcc, %d, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n'
            'v_add_f32_e32 v2, %s, v2\nBB1:\ns_or_b64 exec, exec, s[0:1]\n' % (x, delta))


# name: (source, u at x = 0 and x = 1 when the quad queries)
FIXTURES = {
    'join_pixel': (PROLOGUE + branch_on_x(0, '-0.5') + QUERY + EXPORT, (Fraction(-3, 8), Fraction(5, 8))),
    'join_helper': (PROLOGUE + branch_on_x(1, '0x3fc00000') + QUERY + EXPORT, (Fraction(1, 8), Fraction(17, 8))),
    'cmpx': (PROLOGUE + 'v_cmpx_eq_u32_e32 0, v8\nv_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
             (Fraction(-3, 8), Fraction(5, 8))),
    'scc': (PROLOGUE + 's_cmp_lg_u32 s9, 0\ns_cbranch_scc1 BB1\n' + QUERY + 'BB1:\n' + EXPORT,
            (Fraction(1, 8), Fraction(5, 8))),
    'inside': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n'
               'v_add_f32_e32 v2, -0.5, v2\n' + QUERY + 'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + EXPORT,
               (Fraction(-3, 8), Fraction(5, 8))),
    'kill': (PROLOGUE + 'v_cmpx_eq_u32_e32 0, v8\ns_cbranch_execnz BB1\ns_endpgm\nBB1:\nv_add_f32_e32 v2, -0.5, v2\n'
             + QUERY + EXPORT, (Fraction(-3, 8), Fraction(5, 8))),
    # u is the same in every lane and v = 0.25 + 0.5 * y, so the LOD comes from the vertical derivative alone; only the row
    # y == 0 enters the block. (u0, u1) is (u of the drawn pixel, u + the footprint along y): 4 * 0.5 = 2, LOD 1 - 1 = 0.
    'inside_y': (PROLOGUE + 'v_mov_b32_e32 v2, 0x3e000000\nv_lshrrev_b32_e32 v7, 16, v0\nv_cvt_f32_u32_e32 v6, v7\nv_mul_f32_e32 v6, 0.5, v6\nv_add_f32_e32 v3, 0x3e800000, v6\nv_cmp_eq_u32_e32 vcc, 0, v7\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\nv_mov_b32_e32 v14, 0\n'
                 + QUERY + 'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + EXPORT, (Fraction(1, 8), Fraction(5, 8))),
    # s_wqm exec, exec on a narrowed EXEC turns the whole quad back on (see tests/startup_image_sample_b_flow_check.py).
    'wqm': (PROLOGUE + 'v_cmpx_eq_u32_e32 0, v8\ns_wqm_b64 exec, exec\nv_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
            (Fraction(-3, 8), Fraction(1, 8))),
}
REFUSED = {
    # The plain s_mov_b64 exec, s[N] restore of a mask saved before s_wqm is a lane-exec break too (main accepts the form).
    'restore_mov': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_mov_b64 exec, s[12:13]\nv_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
                    'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A constant 32-bit EXEC write is no more exact than a mask one: lanes 32..63 keep their old bits in wave64.
    'exec_lo_const': (PROLOGUE + 's_mov_b32 exec_lo, -1\nv_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
                      'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A mask saved before s_wqm exec, exec has the guest's helper lanes off, while every thread here starts
    # with exec = 1: writing it back is a lane-exec break (review of ef786354).
    'restore': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_and_b64 exec, exec, s[12:13]\n' + QUERY + EXPORT,
                   'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # The same restore, then u -= 0.5 on the lanes it leaves on: the guest's helper keeps its stale u, the
    # translation's helper takes the new one (measured by the reviewer on WARP and an RX 9070 XT).
    'restore_helper': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_and_b64 exec, exec, s[12:13]\n'
                      'v_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
                   'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A kill export under v_cmpx before the op: the discard is a lane-exec break (it fails without it).
    'discard_break': (PROLOGUE + 'v_cmpx_eq_u32_e32 0, v8\nexp null off, off, off, off vm\n' + QUERY + EXPORT,
                   'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A 32-bit EXEC write keeps lanes 32..63 of the old mask in wave64: only the 64-bit forms are exact.
    'exec_lo_mask': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_mov_b32 exec_lo, vcc_lo\n'
                         'v_add_f32_e32 v2, -0.5, v2\n' + QUERY + EXPORT,
                   'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'vcc_inside': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_cbranch_vccz BB1\n' + QUERY + 'BB1:\n' + EXPORT,
                   'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A scalar write inside the block an EXEC branch can skip happens once for the wave in the guest; a quad that
    # votes to skip it would miss it (review of the first version): the branch is not voted, the op is refused.
    'scalar_in_block': (PROLOGUE + 's_mov_b32 s28, -1.0\nv_cmp_eq_u32_e32 vcc, 2, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n'
                        's_mov_b32 s28, 0\nBB1:\ns_or_b64 exec, exec, s[0:1]\nv_mov_b32_e32 v1, s28\n'
                        + QUERY + EXPORT, 'image_get_lod needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
}
# llvm-mc-19 -arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize64,-wavefrontsize32 of the label-resolved sources
WORDS_HEX = {
    'restore_mov': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff000008'
        '0d127ef0121210ff1204060000003eff02067e0000803e8002187e80021a7e0c04febef10404067fc08cbf080380f1020a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'inside_y': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003eff02067e0000803e8002187e80021a7eff02047e0000003e90000e2c070d0c7ef00c0c10ff0c'
        '06060000803e800e847d6a2480be040088bf80021c7e7fc08cbf080380f1020ac4007e00fe88703f8cbf0f1800f80a0b0c'
        '0d000081bf'
    ),
    'exec_lo_const': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003eff02067e0000803e8002187e80021a7ec103febef10404067fc08cbf080380f1020ac400703f'
        '8cbf0f1800f80a0b0c0d000081bf'
    ),
    'restore_helper': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff000008'
        '0d127ef0121210ff1204060000003eff02067e0000803e8002187e80021a7e7e0cfe87f10404067fc08cbf080380f1020a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'discard_break': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003eff02067e0000803e8002187e80021a7e8010a47d901000f8000000007fc08cbf080380f1020a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'exec_lo_mask': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003eff02067e0000803e8002187e80021a7e8010847d6a03febef10404067fc08cbf080380f1020a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'join_pixel': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010847d6a2480be010088bff10404067e00fe88'
        '7fc08cbf080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'join_helper': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8110847d6a2480be020088bfff0404060000c03f'
        '7e00fe887fc08cbf080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'cmpx': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010a47df10404067fc08cbf080380f1020ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'scc': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e098007bf030085bf7fc08cbf080380f1020ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'inside': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010847d6a2480be040088bff10404067fc08cbf'
        '080380f1020ac4007e00fe88703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'kill': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010a47d010089bf000081bff10404067fc08cbf'
        '080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'vcc_inside': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010847d030086bf7fc08cbf080380f1020ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'scalar_in_block': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003eff02067e0000803e8002187e80021a7ef3039cbe8210847d6a2480be010088bf80039cbe7e00fe881c02027e7fc08cbf080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'restore': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003eff02067e0000803e8002187e80021a7e7e0cfe877fc08cbf080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'wqm': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003eff02067e0000803e8002187e80021a7e8010a47d7e0afebef10404067fc08cbf080380f1020ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
}
CALL = ('t = float4(r0.CalculateLevelOfDetail(q0, float2(F(v2), F(v3))), '
        'r0.CalculateLevelOfDetailUnclamped(q0, float2(F(v2), F(v3))), 0.0, 0.0);')


def all_sources():
    return dict({k: v[0] for k, v in FIXTURES.items()}, **{k: v[0] for k, v in REFUSED.items()})


def pipeline(name):
    return G.Pipeline(VS_WORDS, bytes.fromhex(WORDS_HEX[name]), root_blob(), ps_hw_inputs=PS_INPUTS)


def refused(name, text):
    try:
        pipeline(name).hlsl_ps()
    except G.Unsupported as error:
        assert text in str(error), (name, text, str(error))
    else:
        raise AssertionError('%s accepted, expected %r' % (name, text))


def reference(u):
    """(clamped, unclamped) LOD of the drawn pixel: u is the same down a column, v is constant."""
    lod = math.log2(4 * abs(u[1] - u[0]))
    assert lod == int(lod), u
    return min(max(lod, 0.0), 2.0), lod


def controls():
    for name in FIXTURES:
        text = pipeline(name).hlsl_ps()
        assert text.count(CALL) == 1, (name, text)
    for name, (_source, text) in REFUSED.items():
        refused(name, text)
    print('image_get_lod after a join, under a v_cmpx EXEC, after a uniform branch, inside an EXEC branch and '
          'past a kill branch: CalculateLevelOfDetail; '
          'references %s' % ', '.join('%s=%s' % (n, reference(f[1])) for n, f in FIXTURES.items()))
    print('refused: inside a block a VCC branch skips')


def check_binary(llvm_mc):
    for name, source in all_sources().items():
        built = assemble(llvm_mc, source)
        if built.hex() != WORDS_HEX[name]:
            raise AssertionError('%s: assembled %s, recorded %s' % (name, built.hex(), WORDS_HEX[name]))
    print('every fixture reassembles to its recorded words with %s' % llvm_mc)


def mutants(name, hlsl):
    """HLSL variants that must give another LOD: the branch write dropped, or the EXEC guard dropped."""
    write = re.search(r'\n\s*if \(exec & 1\) v2 = U\((-0\.5|F\(0x3fc00000u\)) \+ F\(v2\)\);', hlsl)
    if name.startswith('join'):
        return [('without the branch write', hlsl.replace(write.group(0), ''))]
    if name in ('cmpx', 'inside', 'kill'):
        return [('every lane writes u', hlsl.replace(write.group(0), write.group(0).replace('if (exec & 1) ', '')))]
    if name == 'wqm':
        return [('s_wqm as a no-op', hlsl.replace('exec = x2w_quad_any(exec) ? 1u : 0u;', ''))]
    return []


def read_lod(renderer, adapter, vs, ps, sampler):
    run = subprocess.run([renderer, adapter, '--startup-image-2d-view', str(vs), str(ps), sampler, '0/3'],
                         capture_output=True, text=True, timeout=60)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    match = re.search(r'view_rgba=((?:[0-9a-f]{8},){3}[0-9a-f]{8})', run.stdout)
    if run.returncode or not match:
        raise AssertionError(run.stdout + run.stderr)
    return struct.unpack('>4f', bytes.fromhex(match.group(1).replace(',', '')))


def close(got, want):
    # One D3D12 LOD fractional step (1/256), as tests/startup_image_get_lod_view_check.py allows.
    return all(abs(g - w) <= 1 / 256 for g, w in zip(got, want))


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for name, (_source, u) in FIXTURES.items():
                pipe = pipeline(name)
                hlsl = pipe.hlsl_ps()
                sampler, = (line for line in pipe.manifest().splitlines() if line.startswith('smpl '))
                want = reference(u) + (0.0, 0.0)
                ps = compile_hlsl(compiler, kind, hlsl, '%s_%s' % (name, kind), ps_profile, directory)
                got = read_lod(renderer, adapter, vs, ps, sampler)
                if not close(got, want):
                    raise AssertionError('%s %s %s: GPU %s, CPU reference %s' % (adapter, kind, name, got, want))
                seen = []
                for i, (label, text) in enumerate(mutants(name, hlsl)):
                    ps = compile_hlsl(compiler, kind, text, '%s_m%d_%s' % (name, i, kind), ps_profile, directory)
                    other = read_lod(renderer, adapter, vs, ps, sampler)
                    if close(other, want):
                        raise AssertionError('%s %s %s: mutant "%s" matched the reference %s'
                                             % (adapter, kind, name, label, other))
                    seen.append('%s -> %s' % (label, other[:2]))
                print('%s %s %s: GPU %s == CPU reference %s; %s'
                      % (adapter, kind, name, got[:2], want[:2], '; '.join(seen) or 'no mutant'), flush=True)


def main():
    if len(sys.argv) == 1 or sys.argv[1] == 'controls':
        controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    elif sys.argv[1] == 'draw' and len(sys.argv) in (5, 6):
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
