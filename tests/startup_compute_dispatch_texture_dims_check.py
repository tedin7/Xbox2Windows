"""Translated compute programs reading 1D, 2D-array and 3D texture views through the stand-in's Dispatch (slot 14).

usage: startup_compute_dispatch_texture_dims_check.py <startup_stream_consumer> <dxc> <warp|hardware>

Case A of startup_gfx2hlsl_compute_check.py with one image_load (for 1D an image_sample_lz at the texel's
center, because gfx2hlsl lowers no 1D image_load) spliced before its store: the texel read is XORed into every
stored word. The program's manifest names the dimension of its texture input (`srv 1 tex1d`,
`tex2da` or `tex3d`, table byte 64). The consumer (dispatch-texdim-<1d|2da|3d>) uploads a described R32_FLOAT
texture, texel (x + 4 y + 16 slice) = 0.25 * (index + 1), subresource by subresource, and creates its view at table byte 64
(device slot 18):

- 1d: eight texels, the program reads x = 5 (1.5);
- 2da: 4x4 with six slices, the view starts at slice 1, the program reads x = 1, y = 2, relative slice 2, which
  is slice 3 of the texture (14.5);
- 3d: 4x4x4, the program reads x = 3, y = 1, z = 2 (10.0).

Per dimension: before translation the dispatch is refused with "no translation" and writes nothing; after
gfx2hlsl.py --pending with DXC all 320 words equal case A's words XORed with the texel's bits on the 288 written
words (the rest keep the sentinel). Controls, each refused with one logged reason and an untouched output:
the title's view is a 2D view of a 2D texture (-mismatch), the view has no description (-undescribed), the view
has a ResourceMinLODClamp of 1.0, which does not fit as address_view_fits reads views (-clamped; a view outside the
texture would remove the device when the title creates it), and the published manifest line rewritten to tex2d (and, for
a second mutant, to another non-2D dimension): the runtime must read the manifest's dimension, not the title's.
Exit 77 when the adapter is not available.
"""
from pathlib import Path
import struct
import sys
import tempfile

import startup_compute_dispatch_texture_check as T
import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

LOAD = 'image_load v12, %s, s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_%s'
# MIMG: dword 0 is 0xf0000000 | dmask << 8 | dim << 3 (1D 0, 3D 2, 2D array 5); dword 1 holds vaddr 8, vdata 12 and
# srsrc 24 / 4 in its three low bytes. The decoder below must read each word back as the text of its dictionary key.
DIM_FIELD = {'1D': 0, '3D': 2, '2D_ARRAY': 5}

# name: (manifest word, image dim, texel coordinates, expected texel value, refusal word for the view, manifest line)
KINDS = {
    '1d': ('tex1d', '1D', (5,), 1.5, '1D', 'srv 1 tex1d table 1 64'),
    '2da': ('tex2da', '2D_ARRAY', (1, 2, 2), 14.5, '2D array', 'srv 1 tex2da table 1 64'),
    '3d': ('tex3d', '3D', (3, 1, 2), 10.0, '3D', 'srv 1 tex3d table 1 64'),
}
# What the consumer holds: texel (x, y, slice) is 0.25 * (x + 4 y + 16 slice + 1); the 2D-array view starts at slice 1.
FIRST_SLICE = {'1d': 0, '2da': 1, '3d': 0}


def texel(kind, coordinates):
    x, y, z = (*coordinates, 0, 0)[:3]
    return 0.25 * (x + 4 * y + 16 * (z + FIRST_SLICE[kind]) + 1)


def vmov(register, value):
    return bytes([0x80 + value, 0x02, register * 2, 0x7e]).hex()  # v_mov_b32_e32 v<register>, <inline constant>


def program(kind):
    _, dim, coordinates, _, _, _ = KINDS[kind]
    C.SPLICE.update({T.XOR: '05190a3a', 's_waitcnt vmcnt(0)': '703f8cbf'})
    if dim == '1D':
        # gfx2hlsl lowers a 1D image only as image_sample, _lz or _l (no image_load), so the texel is read by a
        # level-zero sample at its center under the all-zero (wrap, point) sampler of the sample test.
        center = struct.unpack('<I', struct.pack('<f', (coordinates[0] + 0.5) / 8))[0]
        move = 'v_mov_b32 v8, 0x%08x' % center
        load = 'image_sample_lz v12, v8, s[24:31], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_1D'
        C.SPLICE[move] = 'ff02107e' + struct.pack('<I', center).hex()
        C.SPLICE[load] = '00019cf0' '080ca600'  # image_sample_lz: dword 0 as the 2D word of the sample test, dim 0
        steps = ['s_mov_b64 s[20:21], 0', 's_mov_b64 s[22:23], 0', 's_load_dwordx8 s[24:31], s[10:11], 0x40', move,
                 's_waitcnt lgkmcnt(0)', load, 's_waitcnt vmcnt(0)', T.XOR]
    else:
        registers = 'v[8:%d]' % (7 + len(coordinates))
        load = LOAD % (registers, dim)
        movs = ['v_mov_b32 v%d, %d' % (8 + i, value) for i, value in enumerate(coordinates)]
        C.SPLICE.update({'v_mov_b32 v%d, %d' % (8 + i, value): vmov(8 + i, value) for i, value in enumerate(coordinates)})
        C.SPLICE[load] = bytes([DIM_FIELD[dim] << 3, 1, 0, 0xf0]).hex() + '080c0600'
        steps = ['s_load_dwordx8 s[24:31], s[10:11], 0x40', *movs, 's_waitcnt lgkmcnt(0)', load, 's_waitcnt vmcnt(0)',
                 T.XOR]
    code = C.splice(*steps)
    decoded = [i[3].replace('_e32', '') for i in G.decode(code) if i[0] >= C.STORE_A][:len(steps)]
    assert decoded == [step.replace('_e32', '') for step in steps], decoded
    return code


def check(directory, consumer, dxc, kind, adapter):
    name, dim, coordinates, value, refusal, line = KINDS[kind]
    assert texel(kind, coordinates) == value, (kind, texel(kind, coordinates))
    cache, base = prepare(directory)
    env = dict(base, X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, _, rsrc2, group, groups = C.CASES['A']
    assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
    code = program(kind)
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    written = 24 * 3 * 2 * 2  # threads per group times groups; the words past them keep the sentinel
    untouched, plain = [C.SENTINEL] * C.WORDS, C.expected('A')
    assert plain[written:] == untouched[written:]
    expected = [word ^ bits for word in plain[:written]] + plain[written:]
    layout = '<%dI' % C.WORDS
    mode = 'dispatch-texdim-' + kind

    got, result = T.run(directory, consumer, env, mode, layout)
    assert got == untouched, 'a Dispatch without a translation wrote the output'
    assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    T.translate(cache, dxc, key)
    manifest = cache / f'{key:016x}.txt'
    original = manifest.read_text()
    assert line in original.splitlines() and 'srv 0 buf table 1 0' in original.splitlines(), original

    got, result = T.run(directory, consumer, env, mode, layout)
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    assert not wrong, '%s: word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        kind, wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    layer = T.clean(result)
    assert sum(a != b for a, b in zip(got, plain)) == written, 'the texel must change every written word'

    def refused(label, control_mode, reason):
        got, result = T.run(directory, consumer, env, control_mode, layout)
        logged = lines(result, 'refused: ')
        assert got == untouched and len(logged) == 1 and logged[0].endswith('refused: ' + reason), (
            label, logged, result.stderr[-3000:])
        assert not lines(result, 'Dispatch(3, 2, 2) translated'), label
        errors = [l for l in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in l or ' sev=1 ' in l]
        assert not errors, (label, errors[0])

    refused('view dimension', mode + '-mismatch', f'an input texture view is not {refusal}')
    refused('no description', mode + '-undescribed', f'an input {refusal} texture has no recorded view description')
    refused('view does not fit', mode + '-clamped', 'an input texture view does not fit its texture')

    # Mutants: the dimension is the manifest's. The title's view is of this kind, so another name must be refused.
    for other in ('tex2d', 'tex1d' if kind != '1d' else 'tex3d'):
        reason = {'tex1d': 'an input texture view is not 1D', 'tex2d': 'an input texture view is not 2D',
                  'tex3d': 'an input texture view is not 3D'}[other]
        assert original.count(line) == 1
        manifest.write_bytes(original.replace(line, line.replace(name, other)).encode())
        refused('manifest ' + other, mode, reason)
    manifest.write_bytes(original.encode())
    got, result = T.run(directory, consumer, env, mode, layout)
    assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    print(f'{adapter}: {kind}: {C.WORDS} words match the texel {value} ({written} written words changed); '
          f'view of another dimension, no description, a view that does not fit and a manifest line rewritten to tex2d or '
          f'another dimension are each refused with the output untouched; debug layer: {layer}', flush=True)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        for kind in KINDS:
            check(Path(temporary) / kind, consumer, dxc, kind, adapter)
    print(f'startup_compute_dispatch_texture_dims: PASS on {adapter}')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
