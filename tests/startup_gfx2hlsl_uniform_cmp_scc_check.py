"""SCC of a uniform compare mask ANDed with EXEC in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_uniform_cmp_scc_check.py
       startup_gfx2hlsl_uniform_cmp_scc_check.py draw <dxc> <offscreen_check>
       startup_gfx2hlsl_uniform_cmp_scc_check.py binary <llvm-mc>

Compilers test a wave-uniform condition with a vector compare into an SGPR pair whose sources are scalar (a
constant-buffer word against a literal), then AND that mask with EXEC (`s_and_b64 s[N:N+1], s[N:N+1], exec`),
branch on SCC, and read SCC with src_scc past the join. Every active lane compares the same two values and gets the same bit b, so the AND is b ? EXEC : 0
and SCC is b whenever EXEC is not zero. The single-lane model computes SCC from this thread's bit alone, which is b
only on threads whose lane is active; the src_scc read past the join was refused as a non-uniform SCC.

The SCC is now taken as wave-uniform when every thread that reaches the AND is an active lane: either EXEC is the
entry mask at the compare and at the AND, or the compare and the AND lie in one stretch past a per-lane
`s_cbranch_execz` (a thread whose lane is idle takes it) that nothing in the stretch writes EXEC in and no branch
enters from outside. The 32-bit form with exec_lo also needs a proven wave32 program. Everything else stays
refused: no execz guard, a compare of a lane value, a mask ANDed after EXEC was restored, a stretch entered by a
branch that skips the execz test, and exec_lo in a program with 64-bit EXEC forms.

`draw` translates three vertex stages (entry EXEC; a narrowed stretch with the 64-bit and with the 32-bit AND),
compiles them with DXC and draws two triangles on WARP. Vertices 0-2 run the stretch and export the two saved SCC
values (17 == 17 and 17 == 23) as red and green; vertices 3-5 skip it and keep 0.5. The readback must match the
CPU reference below, and a mutant that inverts the first src_scc read must change the left half.

The words are recorded per source line from llvm-mc (gfx1013); `binary` reassembles every line.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import PS_WORDS, assemble  # noqa: E402
from startup_image_3d_check import compile_hlsl, root_blob  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


WORDS = {
    'v_mov_b32 v6, 0.5': 'f0020c7e',
    'v_mov_b32 v7, 0.5': 'f0020e7e',
    'v_cmp_gt_u32 vcc_lo, 3, v0': '8300887d',
    'v_cmpx_gt_u32 3, v0': '8300a87d',
    's_and_saveexec_b32 s12, vcc_lo': '6a3c8cbe',
    's_bitcmp1_b32 s4, 3': '04830dbf',
    's_cbranch_scc1 1': '010085bf',
    's_cbranch_execz 2': '020088bf',
    's_cbranch_execz 14': '0e0088bf',
    's_nop 0': '000080bf',
    'v_cmp_eq_u32_e64 s8, 17, 17': '0800c2d491220100',
    'v_cmp_eq_u32_e64 s8, 17, v0': '0800c2d491000200',
    's_and_b64 s[8:9], s[8:9], exec': '087e8887',
    's_and_b32 s8, s8, exec_lo': '087e0887',
    's_cbranch_scc0 1': '010084bf',
    'v_mov_b32 v5, 0': '80020a7e',
    's_mov_b32 s20, src_scc': 'fd0394be',
    'v_cmp_eq_u32_e64 s10, 17, 23': '0a00c2d4912e0100',
    's_and_b64 s[10:11], s[10:11], exec': '0a7e8a87',
    's_and_b32 s10, s10, exec_lo': '0a7e0a87',
    's_mov_b32 s21, src_scc': 'fd0395be',
    'v_cvt_f32_u32 v6, s20': '140c0c7e',
    'v_cvt_f32_u32 v7, s21': '150c0e7e',
    's_or_b32 exec_lo, exec_lo, s12': '7e0c7e88',
    'v_mov_b32 v1, 0': '8002027e',
    'v_cmp_eq_u32 vcc_lo, 0, v0': '8000847d',
    'v_cndmask_b32 v1, v1, -4.0, vcc_lo': '010001d501efa901',
    'v_cmp_eq_u32 vcc_lo, 4, v0': '8400847d',
    'v_cndmask_b32 v1, v1, 4.0, vcc_lo': '010001d501eda901',
    'v_cmp_eq_u32 vcc_lo, 2, v0': '8200847d',
    'v_cndmask_b32 v2, -4.0, 4.0, vcc_lo': '020001d5f7eca901',
    'v_cmp_eq_u32 vcc_lo, 5, v0': '8500847d',
    'v_cndmask_b32 v2, v2, 4.0, vcc_lo': '020001d502eda901',
    'v_mov_b32 v3, 0': '8002067e',
    'v_mov_b32 v4, 1.0': 'f202087e',
    'exp pos0 v1, v2, v3, v4 done': 'cf0800f801020304',
    'exp param0 v6, v7, v3, v4': '0f0200f806070304',
    's_endpgm': '000081bf',
}

# Two oversized triangles: vertices 0-2 cover the left half of a 4x4 target, vertices 3-5 the right half.
TAIL = """\
v_mov_b32 v1, 0
v_cmp_eq_u32 vcc_lo, 0, v0
v_cndmask_b32 v1, v1, -4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 4, v0
v_cndmask_b32 v1, v1, 4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32 v2, -4.0, 4.0, vcc_lo
v_cmp_eq_u32 vcc_lo, 5, v0
v_cndmask_b32 v2, v2, 4.0, vcc_lo
v_mov_b32 v3, 0
v_mov_b32 v4, 1.0
exp pos0 v1, v2, v3, v4 done
exp param0 v6, v7, v3, v4
s_endpgm
"""

# EXEC is the entry mask at the compare and at the AND (every vertex saves 17 == 17).
ENTRY = """\
v_mov_b32 v7, 0.5
v_cmp_eq_u32_e64 s8, 17, 17
s_and_b64 s[8:9], s[8:9], exec
s_cbranch_scc0 1
v_mov_b32 v5, 0
s_mov_b32 s20, src_scc
v_cvt_f32_u32 v6, s20
""" + TAIL

# The narrowed stretch (offsets in the comments): +0x10 s_cbranch_execz to +0x4c, compares at +0x14 and +0x2c,
# ANDs at +0x1c and +0x34, each SCC read past a join (+0x28, +0x40).
REGION = """\
v_mov_b32 v6, 0.5
v_mov_b32 v7, 0.5
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_saveexec_b32 s12, vcc_lo
s_cbranch_execz 14
v_cmp_eq_u32_e64 s8, 17, 17
s_and_b64 s[8:9], s[8:9], exec
s_cbranch_scc0 1
v_mov_b32 v5, 0
s_mov_b32 s20, src_scc
v_cmp_eq_u32_e64 s10, 17, 23
s_and_b64 s[10:11], s[10:11], exec
s_cbranch_scc0 1
v_mov_b32 v5, 0
s_mov_b32 s21, src_scc
v_cvt_f32_u32 v6, s20
v_cvt_f32_u32 v7, s21
s_or_b32 exec_lo, exec_lo, s12
""" + TAIL
REGION_B32 = (REGION.replace('s_and_b64 s[8:9], s[8:9], exec', 's_and_b32 s8, s8, exec_lo')
              .replace('s_and_b64 s[10:11], s[10:11], exec', 's_and_b32 s10, s10, exec_lo'))

ACCEPTED = {'entry': ENTRY, 'region_b64': REGION, 'region_b32': REGION_B32}

# The mask is ANDed after EXEC is restored: vertices 3-5 were idle at the compare and hold 0, the others b.
RESTORED = """\
v_mov_b32 v6, 0.5
v_mov_b32 v7, 0.5
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_saveexec_b32 s12, vcc_lo
v_cmp_eq_u32_e64 s8, 17, 17
s_or_b32 exec_lo, exec_lo, s12
s_and_b64 s[8:9], s[8:9], exec
s_cbranch_scc0 1
v_mov_b32 v5, 0
s_mov_b32 s20, src_scc
v_cvt_f32_u32 v6, s20
""" + TAIL
# A uniform branch at +0xc jumps past the execz test into the stretch, with EXEC narrowed by v_cmpx.
ENTERED = """\
v_mov_b32 v6, 0.5
v_cmpx_gt_u32 3, v0
s_bitcmp1_b32 s4, 3
s_cbranch_scc1 1
""" + REGION.split('s_and_saveexec_b32 s12, vcc_lo\n', 1)[1]

UNVERIFIED = 'src_scc with unverified SCC [operand read: s_mov_b32, destination sgpr] at +0x%x: '
REFUSED = {
    'no s_cbranch_execz guard': (REGION.replace('s_cbranch_execz 14', 's_nop 0'), 0x28),
    'compare of a lane value': (REGION.replace('v_cmp_eq_u32_e64 s8, 17, 17', 'v_cmp_eq_u32_e64 s8, 17, v0'), 0x28),
    'mask ANDed after EXEC is restored': (RESTORED, 0x28),
    'stretch entered past the execz test': (ENTERED, 0x28),
    'exec_lo AND with 64-bit EXEC forms': (REGION.replace('s_and_b64 s[10:11], s[10:11], exec',
                                                          's_and_b32 s10, s10, exec_lo'), 0x40),
}


def words(source):
    return b''.join(bytes.fromhex(WORDS[line]) for line in source.splitlines())


def translate(source):
    pipe = G.Pipeline(words(source), PS_WORDS, root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def controls():
    shaders = {}
    for name, source in ACCEPTED.items():
        try:
            shaders[name] = translate(source)
        except G.Unsupported as error:
            check(name + ' translates', False, str(error))
        else:
            check(name + ' translates', True)
    for name, (source, read) in REFUSED.items():
        try:
            translate(source)
        except G.Unsupported as error:
            check(name + ' is refused', str(error).startswith(UNVERIFIED % read), str(error))
        else:
            check(name + ' is refused', False, 'translated')
    return shaders


def reference(name):
    """The guest's result: vertices 0-2 run the stretch (all six in `entry`), the rest keep 0.5."""
    def color(vertex):
        if name == 'entry':
            return float(17 == 17), 0.5
        if vertex < 3:
            return float(17 == 17), float(17 == 23)
        return 0.5, 0.5
    # Flat attributes come from the first vertex of each triangle: 0 on the left, 3 on the right.
    return b''.join(struct.pack('<4f', *color(0 if x < 2 else 3), 0.0, 1.0) for _ in range(4) for x in range(4))


FIRST_READ = 's20 = (scc ? 1u : 0u);'


def draw(dxc, runner, shaders):
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        runs = []
        for name, (vs, ps) in shaders.items():
            check(name + ' reads SCC once into s20', vs.count(FIRST_READ) == 1)
            runs.append((name, name, vs, ps, 0))
        vs, ps = shaders['region_b64']
        runs.append(('region_b64_mutant', 'region_b64', vs.replace(FIRST_READ, 's20 = (scc ? 0u : 1u);'), ps, 8))
        for label, name, vs, ps, mismatches in runs:
            pixel = compile_hlsl(dxc, 'dxc', ps, label + '_ps', 'ps_6_0', directory)
            vertex = compile_hlsl(dxc, 'dxc', vs, label + '_vs', 'vs_6_0', directory)
            expected = directory / (label + '.bin')
            expected.write_bytes(reference(name))
            run = subprocess.run([runner, 'warp', '--raster-reference', str(vertex), str(pixel),
                                  '4', '4', str(expected), 'none', 'cw'], capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(1 if failures else 77)
            report = run.stdout + run.stderr
            want = f'{mismatches} of 16 pixels differ' if mismatches else 'all 16 pixels matched'
            check(label + ' WARP readback', run.returncode == int(bool(mismatches)) and want in report.splitlines(),
                  report)


if len(sys.argv) == 4 and sys.argv[1] == 'draw':
    shaders = controls()
    if not failures:
        draw(sys.argv[2], str(Path(sys.argv[3]).resolve()), shaders)
elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
    for line, hex_words in WORDS.items():
        check('encoding of ' + line, assemble(sys.argv[2], line + '\n').hex() == hex_words)
elif len(sys.argv) == 1:
    controls()
else:
    sys.exit(__doc__)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
