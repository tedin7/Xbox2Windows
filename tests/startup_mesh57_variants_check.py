"""Synthetic slot 57 exchange variants and a bounded runtime loop.

The programs edit the source-built indexed fixture in startup_mesh57_loop.s.
Binary mode verifies the edits against an assembler. Draw mode exercises the
normal request path and compares DispatchMesh readback against the CPU image.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import startup_mesh57_loop_check as fixture


def replace(words, old, new):
    old = old.split()
    starts = [i for i in range(len(words)) if words[i:i + len(old)] == old]
    assert len(starts) == 1, old
    i = starts[0]
    return words[:i] + new.split() + words[i + len(old):]


def variants():
    base = fixture.MESH
    loop = replace(base, 'bea80380 80288128 bf0a8428', '87298708 bea80380 80288128 bf0a2928')
    group = replace(base, 'befe0481 be84047e', 'befe0481')
    for branch in ('bf880007', 'bf880009'):
        group = replace(group, 'befe0404 ' + branch, 'befe0481 ' + branch)
    group = replace(group, '2c0a0a87 7e0c0280', '7e0c0280 2c0a0a87')
    count = replace(base, '7e06020e d8340000 00000102', 'd8340000 00000102 7e06020e')
    hoisted = replace(base, 'be8e0309 bf8a0000 bf008006 bf840010', 'be8e0309 7e040280 bf8a0000 bf008006 bf84000e')
    hoisted = replace(hoisted, 'bf880007 7e02020d 7e040280 7e06020e d8340000 00000102 d8340004 00000302',
                      '7e06020e 7e02020d d8340004 00000302 d8340000 00000102')
    return {
        'loop': (loop, replace(loop, 'bf85fffd', 'bf800000')),
        'group_index': (group, replace(group, 'be8d0308', 'be8d0383')),
        'count': (count, replace(count, 'be8e0309', 'be8e038b')),
        'count_hoisted': (hoisted, replace(hoisted, 'be8e0309', 'be8e038b')),
    }


def binary(mc):
    src = (fixture.HERE / 'startup_mesh57_loop.s').read_text()
    loop = src.replace('s_mov_b32 s40, 0', 's_and_b32 s41, s8, 7\ns_mov_b32 s40, 0')
    loop = loop.replace('s_cmp_lt_u32 s40, 4', 's_cmp_lt_u32 s40, s41')
    group = src.replace('s_mov_b64 s[4:5], exec\n', '').replace('s_mov_b64 exec, s[4:5]', 's_mov_b64 exec, 1')
    group = group.replace('v_lshrrev_b32 v5, 7, v5\nv_mov_b32 v6, 0', 'v_mov_b32 v6, 0\nv_lshrrev_b32 v5, 7, v5')
    count = src.replace('v_mov_b32 v3, s14\nds_write_b32 v2, v1', 'ds_write_b32 v2, v1\nv_mov_b32 v3, s14')
    hoisted = src.replace('s_mov_b32 s14, s9\ns_barrier', 's_mov_b32 s14, s9\nv_mov_b32 v2, 0\ns_barrier')
    hoisted = hoisted.replace('s_cbranch_scc0 16', 's_cbranch_scc0 14')
    hoisted = re.sub(r's_cbranch_execz 7[^\n]*\nv_mov_b32 v1, s13\nv_mov_b32 v2, 0\n'
                     r'v_mov_b32 v3, s14\nds_write_b32 v2, v1\nds_write_b32 v2, v3 offset:4',
                     'v_mov_b32 v3, s14\nv_mov_b32 v1, s13\nds_write_b32 v2, v3 offset:4\nds_write_b32 v2, v1',
                     hoisted)
    sources = {
        'loop': (loop, loop.replace('s_cbranch_scc1 -3\n', 's_nop 0\n')),
        'group_index': (group, group.replace('s_mov_b32 s13, s8', 's_mov_b32 s13, 3')),
        'count': (count, count.replace('s_mov_b32 s14, s9', 's_mov_b32 s14, 11')),
        'count_hoisted': (hoisted, hoisted.replace('s_mov_b32 s14, s9', 's_mov_b32 s14, 11')),
    }
    for name, programs in variants().items():
        for words, asm in zip(programs, sources[name]):
            fixture.expect(name + ' assembly', fixture.assemble(mc, asm), words)


def controls():
    os.environ['X2W_FXC'] = ''
    for reason, family in (
            (fixture.LOOP_REASON, 'mesh57-loop'),
            ('slot 57 mesh group-index exchange: instruction 2 of the accepted shape (s_mov_b64) is absent',
             'mesh57-group-index'),
            ('slot 57 mesh count exchange: instruction 8 of the accepted shape (v_mov_b32_e32) is absent',
             'mesh57-count'),
            ('slot 57 mesh count exchange: instruction 5 of the accepted shape (s_cbranch_execz) is absent',
             'mesh57-count')):
        fixture.expect('stale refusal is prioritized for retry', fixture.G.lifted_family(reason), family)
    with tempfile.TemporaryDirectory() as tmp:
        for name, programs in variants().items():
            for i, words in enumerate(programs):
                key, result = fixture.request(tmp, words)
                fixture.expect(f'{name} program {i} translates', result, 'ok')
                if result == 'ok':
                    hlsl = (Path(tmp) / f'{key:016x}.ms57.hlsl').read_text()
                    fixture.expect(f'{name} program {i} runtime loop', '[loop] do {' in hlsl, name == 'loop' and i == 0)
        loop, _ = variants()['loop']
        for old, new in [('87298708', '8729ff08 ffffffff'), ('80288128', '80288228')]:
            _, result = fixture.request(tmp, replace(loop, old, new))
            fixture.expect('unproven runtime recurrence refuses', result, 'unsupported: ' + fixture.LOOP_REASON)
        count, _ = variants()['count']
        _, result = fixture.request(tmp, replace(count, '7e06020e', '7e02020e'))
        fixture.expect('count store of the wrong register refuses', result.startswith(
            'unsupported: slot 57 mesh count exchange:'), True)
        _, result = fixture.request(tmp, replace(count, 'bf840010', 'bf840011'))
        fixture.expect('count branch into the reader refuses', result,
                       'unsupported: slot 57 mesh count exchange: a branch leaves the accepted shape')
        hoisted, _ = variants()['count_hoisted']
        _, result = fixture.request(tmp, replace(hoisted, 'be8e0309 7e040280', 'be8e0309 7e040281'))
        fixture.expect('hoisted nonzero store address refuses', result.startswith(
            'unsupported: slot 57 mesh count exchange:'), True)


def draw(name, dxc, renderer):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    programs = variants()[name]
    points, colours, indices = fixture.geometry()
    packed_indices = [a | b << 10 | c << 20 for a, b, c in indices]
    prims, verts = len(indices), len(points)
    red = 255 if max(1, prims & 7) == 4 else 0
    image = fixture.rasterise(points, [(red, *colour[1:]) for colour in colours], indices, verts, prims)
    with tempfile.TemporaryDirectory() as tmp:
        mesh_code = fixture.code(programs[0])
        pipeline = fixture.G.MeshPipeline(mesh_code, fixture.code(fixture.PIXEL), fixture.root())
        constants = bytearray(1024)
        for slot, value in enumerate((prims, verts)):
            struct.pack_into('<I', constants, 4 * pipeline.uds.index(('rootconst', 0, slot)), value)
        path = Path(tmp) / 'draw.bin'
        path.write_bytes(bytes(constants) + fixture.buffer(points, colours, packed_indices) + b'\xcd' * 80
                         + b''.join(bytes(pixel) for pixel in image))
        for mutant, words in enumerate(programs):
            key, result = fixture.request(tmp, words)
            fixture.expect(f'{name} program {mutant} compiles', result, 'ok')
            if result != 'ok':
                return 1
            stem = Path(tmp) / f'{key:016x}'
            run = subprocess.run(
                [renderer, 'warp', '--mesh-indexed', str(stem.with_suffix('.ms57.cso')),
                 str(stem.with_suffix('.ps57.cso')), str(path)], capture_output=True, text=True)
            print(run.stdout + run.stderr)
            if run.returncode == 77:
                return 77
            if mutant:
                fixture.expect(name + ' mutant fails the CPU image',
                               (run.returncode == 1 and 'Draw reference mismatch' in run.stderr
                                and re.search(r'pixels=1024 mismatches=[1-9]\d*', run.stdout) is not None), True)
            else:
                fixture.expect(name + ' matches the CPU image',
                               (run.returncode, 'pixels=1024 mismatches=0' in run.stdout), (0, True))
    return int(bool(fixture.FAILED))


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['draw'] and len(args) == 4:
        raise SystemExit(draw(*args[1:]))
    if args[:1] == ['binary'] and len(args) == 2:
        binary(args[1])
    elif not args:
        controls()
    else:
        raise SystemExit('usage: startup_mesh57_variants_check.py [binary <llvm-mc> | draw <variant> <dxc> <renderer>]')
    if not fixture.FAILED:
        print('startup_mesh57_variants: ok')
    raise SystemExit(int(bool(fixture.FAILED)))
