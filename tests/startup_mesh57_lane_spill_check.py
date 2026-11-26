"""Constant-lane v_writelane_b32 / v_readlane_b32 in translate_mesh57 (#2413 T7).

usage: startup_mesh57_lane_spill_check.py
       startup_mesh57_lane_spill_check.py binary <llvm-mc>
       startup_mesh57_lane_spill_check.py draw <dxc> <offscreen_check> <warp|hardware>

The fixture extends the accepted slot 57 group-memory shape with the #2127 spill idiom:
park ordinary scalars in fixed wave64 lanes of a spill-only VGPR under EXEC-independent
writelane/readlane, clobber the SGPRs, reload them, and use the colour reload as vertex red.
Raw EXEC sources stay refused. Ordinary VGPR use of the spill register, an exchange VGPR, an
unwritten slot, a register-selected lane, and a mutant that zeroes the spilled colour assignment
(not the multi-decl) must fail by name or change pixels.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_mesh57_skeleton_check import (  # noqa: E402
    MESH, PIXEL, PIXEL_ASM, POST_COUNT, assemble, buffer, code, expect, FAILED, geometry, rasterise, root,
)

SIDE = 32


def writelane(vd, src, lane):
    """Two little-endian words for `v_writelane_b32 vD, src, lane` (src is an SGPR index or special)."""
    return ['%08x' % (0xd7610000 | vd), '%08x' % (src | ((128 + lane) << 9))]


def readlane(sd, vv, lane):
    """Two little-endian words for `v_readlane_b32 sD, vV, lane` (VGPR encoded as 256+V)."""
    return ['%08x' % (0xd7600000 | sd), '%08x' % ((256 + vv) | ((128 + lane) << 9))]


# s_mov_b32 s40, 1.0; writelane v30 lane 3; clobber; park/reload a second ordinary scalar in lane 63
# (wave64); v_mov_b32 v7, s40 — vertex red comes from the colour reload. Raw EXEC spills are refused.
SPILL_ASM = """s_mov_b32 s40, 1.0
v_writelane_b32 v30, s40, 3
s_mov_b32 s40, 0
s_mov_b32 s41, 0x11
v_writelane_b32 v30, s41, 63
s_mov_b32 s41, 0
v_readlane_b32 s41, v30, 63
v_readlane_b32 s40, v30, 3
v_mov_b32 v7, s40"""
SPILL_WORDS = (
    ['bea803f2'] + writelane(30, 40, 3) + ['bea80380']
    + ['bea90391'] + writelane(30, 41, 63) + ['bea90380'] + readlane(41, 30, 63)
    + readlane(40, 30, 3) + ['7e0e0228']
)
# After the colour load's waitcnt, overwrite v20 (red) with the reloaded spill in v7.
_WAIT = MESH.index('bf8c3f70')
SPILL_MESH = (
    MESH[:POST_COUNT] + SPILL_WORDS + MESH[POST_COUNT:_WAIT + 1]
    + ['7e280307'] + MESH[_WAIT + 1:]
)
# Mutant: drop the colour writelane so the reload of lane 3 is unwritten at the host, or after a
# successful translation zero the spill assignment in HLSL so the draw oracle fails.
MUTANT_NO_WRITE = (
    MESH[:POST_COUNT] + (
        ['bea803f2', 'bea80380'] + ['bea90391'] + writelane(30, 41, 63) + ['bea90380']
        + readlane(41, 30, 63) + readlane(40, 30, 3) + ['7e0e0228']
    ) + MESH[POST_COUNT:_WAIT + 1] + ['7e280307'] + MESH[_WAIT + 1:]
)


def request(directory, mesh, pixel=PIXEL):
    ms, ps, blob = code(mesh), code(pixel), root()
    key = G.mesh57_key(ms, ps, blob)
    for ext, data in (('.root', blob), ('.ps57', ps), ('.m57', ms)):
        (Path(directory) / ('req_%016x%s' % (key, ext))).write_bytes(data)
    return key, G.translate_request(str(directory), key)


def controls():
    os.environ['X2W_FXC'] = ''
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, SPILL_MESH)
        expect('spill fixture translates', result, 'ok')
        if result == 'ok':
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('spill slots are declared', 'sp30_3' in text and 'sp30_63' in text, True)
            expect('colour spill is assigned', 'sp30_3 =' in text, True)
            expect('wave64 lane 63 parks the second scalar', 'sp30_63 =' in text and 's41 = sp30_63' in text, True)
            expect('reload feeds the SGPR', 's40 = sp30_3' in text, True)
            expect('spill VGPR is not a vector temporary', not re.search(r'\bv30\b', text), True)

    # Slot 57 mesh is always wave64: lanes 3 and 35 of one VGPR are two slots there (the k / k + 32 pair rule of the
    # pixel and vertex stages, whose wave size is unknown, must not refuse them).
    with tempfile.TemporaryDirectory() as tmp:
        pair = (MESH[:POST_COUNT] + ['bea803f2'] + writelane(30, 40, 3) + ['bea80380']
                + ['bea90391'] + writelane(30, 41, 35) + ['bea90380'] + readlane(41, 30, 35)
                + readlane(40, 30, 3) + ['7e0e0228'] + MESH[POST_COUNT:])
        key, result = request(tmp, pair)
        expect('mesh lanes 3 and 35 of one VGPR are two slots', result, 'ok')
        if result == 'ok':
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('both slots are declared', 'sp30_3' in text and 'sp30_35' in text, True)

    refusals = (
        ('unwritten spill slot',
         MESH[:POST_COUNT] + ['bea803f2'] + writelane(30, 40, 3) + ['bea80380']
         + readlane(40, 30, 4) + ['7e0e0228'] + MESH[POST_COUNT:],
         'v_readlane_b32 reads v30 lane 4 before any v_writelane_b32 wrote it'),
        ('register-selected lane',
         MESH[:POST_COUNT] + ['bea803f2', 'bea80380']
         + ['%08x' % (0xd761001e), '%08x' % (40 | (40 << 9))]  # v_writelane v30, s40, s40
         + readlane(40, 30, 3) + ['7e0e0228'] + MESH[POST_COUNT:],
         'v_writelane_b32 requires a constant lane, got s40'),
        ('lane 64 outside wave64',
         MESH[:POST_COUNT] + ['bea803f2'] + writelane(30, 40, 64) + MESH[POST_COUNT:],
         'v_writelane_b32 lane 64 outside 0..63'),
        ('mixed ordinary use of the spill VGPR',
         MESH[:POST_COUNT] + ['bea803f2'] + writelane(30, 40, 3) + ['bea80380']
         + ['7e3c0300']  # v_mov_b32 v30, v0 — ordinary vector write of the spill VGPR
         + readlane(40, 30, 3) + ['7e0e0228'] + MESH[POST_COUNT:],
         'v_writelane_b32: v30 is also used as a vector (v_mov_b32_e32)'),
        ('spill into the group-index exchange VGPR',
         MESH[:POST_COUNT] + ['bea803f2'] + writelane(5, 40, 3) + ['bea80380']
         + readlane(40, 5, 3) + ['7e0e0228'] + MESH[POST_COUNT:],
         'v_writelane_b32: v5 is also used as a vector (v_lshrrev_b32_e32)'),
        ('raw exec_hi spill',
         MESH[:POST_COUNT] + writelane(30, 127, 63) + MESH[POST_COUNT:],
         'v_writelane_b32 source exec_hi is a lane mask or m0'),
        ('raw exec_lo spill',
         MESH[:POST_COUNT] + writelane(30, 126, 3) + MESH[POST_COUNT:],
         'v_writelane_b32 source exec_lo is a lane mask or m0'),
        ('read before any write',
         MESH[:POST_COUNT] + readlane(40, 30, 3) + ['7e0e0228'] + MESH[POST_COUNT:],
         'v_readlane_b32 reads v30 lane 3 before any v_writelane_b32 wrote it'),
        ('missing colour write refuses at reload', MUTANT_NO_WRITE,
         'v_readlane_b32 reads v30 lane 3 before any v_writelane_b32 wrote it'),
    )
    for what, mesh, reason in refusals:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, mesh)
            expect(what, result, 'unsupported: ' + reason)
            expect(what + ': only the refusal is published',
                   sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')),
                   ['%016x.fail' % key])


def binary(mc):
    mesh = (HERE / 'startup_mesh57_skeleton.s').read_text()
    insert = mesh.replace('v_cmpx_gt_i32 s14, v0', SPILL_ASM + '\nv_cmpx_gt_i32 s14, v0')
    insert = insert.replace('exp pos0', 'v_mov_b32 v20, v7\nexp pos0')
    expect('spill mesh words match llvm-mc', assemble(mc, insert), SPILL_MESH)
    expect('pixel words match llvm-mc', assemble(mc, PIXEL_ASM), PIXEL)


def draw(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, SPILL_MESH)
        expect('spill fixture translates and compiles', result, 'ok')
        if result != 'ok':
            return 0
        name = Path(tmp) / ('%016x' % key)
        pipe = G.MeshPipeline(code(SPILL_MESH), code(PIXEL), root())
        constants = bytearray(1024)
        for slot, count in enumerate((4, 12)):
            struct.pack_into('<I', constants, 4 * pipe.uds.index(('rootconst', 0, slot)), count)
        words = [a | b << 10 | c << 20 for a, b, c in indices]
        # Reloaded s40 is 1.0 (0x3f800000); v_mov v7, s40 so every vertex red channel is 255.
        red = [(255, g, b, a) for (_, g, b, a) in colours]
        image = rasterise(points, red, indices, 12, 4)
        head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
        path = Path(tmp) / 'spill.bin'
        path.write_bytes(head + b''.join(map(bytes, image)))
        args = [renderer, adapter, '--mesh-indexed', str(name) + '.ms57.cso', str(name) + '.ps57.cso', str(path)]
        run = subprocess.run(args, capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        expect('draw of spill fixture', (run.returncode, 'pixels=1024 mismatches=0' in run.stdout), (0, True))

        # Mutant: zero the spilled colour word in HLSL so the reload yields 0 and red becomes 0.
        # Match a standalone assignment only (not the multi-decl `uint sp30_3 = 0, sp30_63 = 0;`).
        source = name.with_suffix('.ms57.hlsl').read_text()
        expect('fixture writes the colour spill slot',
               re.search(r'(?m)^\s*sp30_3 = ', source) is not None, True)
        mutated = Path(tmp) / 'zero-spill.hlsl'
        mutated_text, nsub = re.subn(r'(?m)^(\s*)sp30_3 = [^;]+;', r'\1sp30_3 = 0u;', source, count=1)
        expect('mutant rewrites exactly one colour spill assignment', nsub, 1)
        expect('mutant still declares lane 63', 'sp30_63' in mutated_text, True)
        mutated.write_text(mutated_text)
        mutant = Path(tmp) / 'zero-spill.cso'
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutated], check=True)
        zero_red = [(0, g, b, a) for (_, g, b, a) in colours]
        wrong = rasterise(points, zero_red, indices, 12, 4)
        different = sum(a != b for a, b in zip(image, wrong))
        expect('zeroing the spill changes pixels', different > 0, True)
        path.write_bytes(head + b''.join(map(bytes, image)))
        run = subprocess.run(
            [renderer, adapter, '--mesh-indexed', str(mutant), str(name) + '.ps57.cso', str(path)],
            capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        expect('mutant fails the correct reference',
               (run.returncode != 0 and 'mismatches=' in run.stdout), True)
        path.write_bytes(head + b''.join(map(bytes, wrong)))
        run = subprocess.run(
            [renderer, adapter, '--mesh-indexed', str(mutant), str(name) + '.ps57.cso', str(path)],
            capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        expect('mutant matches the zeroed reference',
               (run.returncode, 'pixels=1024 mismatches=0' in run.stdout), (0, True))
        print('spill draw: %d pixels differ when the spilled word is zeroed' % different)
    return 0


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif len(argv) == 1:
        controls()
    else:
        print(__doc__)
        return 2
    if FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
