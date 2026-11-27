"""Slot 57 mask selects, with a DispatchMesh WARP render-target oracle."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

from startup_mesh57_skeleton_check import (
    AFTER_COLOUR_LOAD, G, MESH, PIXEL, buffer, code, geometry, rasterise, request, root,
)

# Reuse the count exchanges and export plumbing. The inserted program makes
# red depend on a sparse VCC, green on a numeric EXEC word, and blue on saved EXEC.
PROGRAM = [
    'bf068408',  # s_cmp_eq_u32 s8, 4
    '7d880086',  # v_cmp_gt_u32 vcc, 6, v0
    '856a6a7e',  # s_cselect_b32 vcc_lo, exec_lo, vcc_lo
    '7e3a02f2',  # v_mov_b32 v29, 1.0
    '020e3a80',  # v_cndmask_b32 v7, 0, v29, vcc
    '857f80c1',  # s_cselect_b32 exec_hi, -1, 0
    'bf068080',  # s_cmp_eq_u32 0, 0
    '8528807f',  # s_cselect_b32 s40, exec_hi, 0
    'befe04c1',  # s_mov_b64 exec, -1
    'bf06c128',  # s_cmp_eq_u32 s40, -1
    '856a80c1',  # s_cselect_b32 vcc_lo, -1, 0
    'bea90380',  # s_mov_b32 s41, 0
    'd501001a', '00a23a80',  # v_cndmask_b32 v26, 0, v29, s[40:41]
    '02363480',  # v_cndmask_b32 v27, 0, v26, vcc
    '7d880086',  # v_cmp_gt_u32 vcc, 6, v0
    'beaa246a',  # s_and_saveexec_b64 s[42:43], vcc
    'befe04c1',  # s_mov_b64 exec, -1
    'bf068408',  # s_cmp_eq_u32 s8, 4
    '856a802b',  # s_cselect_b32 vcc_lo, s43, 0
    '02383a80',  # v_cndmask_b32 v28, 0, v29, vcc
]
MUTATIONS = {'red': 2, 'green': 5, 'blue': 19}
EXPORT_COLOUR = ['7e280307', '7e2a031b', '7e2c031c']  # v20..22 = v7, v27, v28


def mesh(program):
    start = MESH.index('7d28000e')
    after_wait = AFTER_COLOUR_LOAD + 1
    return MESH[:start] + program + MESH[start:after_wait] + EXPORT_COLOUR + MESH[after_wait:]


def mutant(index):
    program = PROGRAM.copy()
    word = int(program[index], 16)
    program[index] = '%08x' % ((word & ~0xffff) | ((word & 0xff) << 8) | ((word >> 8) & 0xff))
    return program


def pipeline(program):
    return G.MeshPipeline(code(mesh(program)), code(PIXEL), root())


def reference(points, indices, prims, mutation=None):
    scc = prims == 4
    colours = []
    for lane in range(len(points)):
        red = (scc ^ (mutation == 'red')) or lane < 6
        green = scc ^ (mutation == 'green')
        blue = scc ^ (mutation == 'blue')
        colours.append(tuple(255 * bit for bit in (red, green, blue, True)))
    return rasterise(points, colours, indices, len(points), prims)


def host():
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'X2W_FXC': ''}):
        key, result = request(tmp, mesh(PROGRAM))
        assert result == 'ok', result
        hlsl = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        invalid = PROGRAM.copy()
        invalid[2] = '856a6a81'  # numeric literal 1 in place of EXEC_LO
        _, result = request(tmp, mesh(invalid))
        assert result == 'unsupported: slot 57 mesh: unproven mask select source 1', result
    assert hlsl.count(' = mesh_ballot(tid, pc == ') == 5
    assert 's40 = mesh_mask;' in hlsl
    for channel, index in MUTATIONS.items():
        assert pipeline(mutant(index)).hlsl_ms() != hlsl, channel
    points, _, indices = geometry()
    for prims in (3, 4):
        want = reference(points, indices, prims)
        for channel in MUTATIONS:
            assert reference(points, indices, prims, channel) != want, (prims, channel)
    print('slot 57 mask selects and CPU mutant references: ok')


def draw(dxc, renderer):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    packed = [a | b << 10 | c << 20 for a, b, c in indices]
    uds = pipeline(PROGRAM).uds
    with tempfile.TemporaryDirectory() as tmp:
        shaders = {}
        for channel, program in [(None, PROGRAM), *[(c, mutant(i)) for c, i in MUTATIONS.items()]]:
            key, result = request(tmp, mesh(program))
            if result != 'ok':
                raise AssertionError((channel, result))
            shaders[channel] = Path(tmp) / ('%016x' % key)
        path = Path(tmp) / 'draw.bin'
        for prims in (3, 4):
            constants = bytearray(1024)
            for offset, value in enumerate((prims, len(points))):
                struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, offset)), value)
            expected = reference(points, indices, prims)
            path.write_bytes(bytes(constants) + buffer(points, colours, packed) + b'\xcd' * 80
                             + b''.join(bytes(pixel) for pixel in expected))
            for channel, shader in shaders.items():
                args = [renderer, 'warp', '--mesh-indexed', str(shader) + '.ms57.cso',
                        str(shader) + '.ps57.cso', str(path)]
                result = subprocess.run(args, capture_output=True, text=True)
                print('SCC=%d mutant=%s\n%s%s' % (prims == 4, channel, result.stdout, result.stderr))
                if result.returncode == 77:
                    return 77
                if channel is None:
                    assert result.returncode == 0 and 'pixels=1024 mismatches=0' in result.stdout.splitlines()
                else:
                    changed = reference(points, indices, prims, channel)
                    mismatches = sum(a != b for a, b in zip(expected, changed))
                    assert mismatches > 0
                    assert result.returncode == 1 and 'Draw reference mismatch' in result.stderr
                    assert ('pixels=1024 mismatches=%d' % mismatches) in result.stdout.splitlines()
    return 0


if __name__ == '__main__':
    # Keep fixture products under the repository's ignored output directory.
    output = Path(__file__).resolve().parent.parent / 'out'
    output.mkdir(exist_ok=True)
    tempfile.tempdir = str(output)
    if sys.argv[1:2] == ['draw'] and len(sys.argv) == 4:
        sys.exit(draw(*sys.argv[2:]))
    if len(sys.argv) != 1:
        sys.exit('usage: startup_gfx2hlsl_mesh_cselect_check.py [draw <dxc> <offscreen_check>]')
    host()
