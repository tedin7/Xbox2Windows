"""Translated compute programs with 2D texture views, dispatched through the stand-in's command list (slot 14).

usage: startup_compute_dispatch_texture_check.py <startup_stream_consumer> <dxc> <warp|hardware> [two-mips]

Texture output: the consumer creates the image store program of startup_compute_image_store_d16_check.py as a
device slot 67 pipeline, fills a 32x8 RGBA16_FLOAT texture with sentinel halves, creates an unordered access view of
it at table byte 64 (device slot 19), records Dispatch(2, 1, 1) and copies the texture to a readback buffer. Four
runs share one cache:

1. before translation: refused with "no translation", every half keeps its sentinel;
2. after gfx2hlsl.py --pending with DXC: the 1024 halves equal mip 0 of the fixture's CPU reference (64 changed);
3. texture-store-none (no Dispatch recorded): every half keeps its sentinel;
4. texture-store-denied (a texture without unordered access, whose view the stand-in does not create): refused
   with one logged reason, every half keeps its sentinel.
A fifth run binds the same program under a manifest edited to declare no entry values.
The 1D and 3D float texture cases check mip-one readback, a mismatched view, and an invalid 3D W range.

Texture input: case A of startup_gfx2hlsl_compute_check.py with a level-zero sample of a 2D texture XORed into the
stored word (a static sampler, a texture view at table byte 64). The consumer adds a 1x1 RGBA8 texture whose red
is 1.0 and a shader resource view of it (device slot 18). After translation all 320 words equal the case's
expectation XOR the bits of 1.0f; dispatch-sampled-unbound (no view created) is refused with one logged reason and
every word keeps the sentinel.

Filtered input: the texture input case with a constant wrap, linear sampler that the translator writes as `smpl`
and a 2x1 texture (dispatch-sampled-2x1): red 0.5 at the left edge. The same program with the manifest's sampler
changed to clamp reads red 0 (the control); an `smpheap` line is refused with its reason.

Compared input: case A with a level-zero comparison sample (image_sample_c_lz) under a constant point, clamp, LESS
sampler that the translator writes as `smpc`, and a 2x1 R32_FLOAT texture of depths 1.0 and 2.5
(dispatch-sampled-depth). The reference is the thread's z id and u its x id, so the 288 written words hold both
outcomes; all 320 words equal a CPU model of the comparison. The same program with the manifest's function changed
to GREATER matches the GREATER model (the control).

The two-mips mode writes two different patterns through separate UAVs of the same texture, then swaps the views.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_compute_image_store_d16_check as S
import startup_gfx2hlsl_compute_check as C
import startup_gfx2hlsl_image_store_3d1d_check as V
from startup_compute_dispatch_check import lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G
from startup_image_load_mip_check import DIMENSIONS, FLOAT_BITS, IDENTIFIERS, debug_errors, typed_mutant
from startup_stream_draw_check import vop1, vop2


def run(directory, consumer, env, mode, layout):
    out = directory / 'out.bin'
    out.unlink(missing_ok=True)
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(out)), capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    return list(struct.unpack(layout, out.read_bytes())), result


def translate(cache, dxc, key, *, integer_srvs=False):
    out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                         text=True, env=dict(os.environ, X2W_DXC=dxc, X2W_INTEGER_TEXTURE_SRVS='1' if integer_srvs else '0')).stdout
    assert f'{key:016x}: ok' in out, out


def clean(result):
    assert not lines(result, 'refused'), result.stderr[-3000:]
    errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    assert not errors, errors[0]
    layer = lines(result, 'x2w-d3d12 debug layer: ')
    return 'enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'


def store(directory, consumer, dxc, env):
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(3, table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    assert S.fixture.GROUP == (4, 2, 1) and S.fixture.GROUPS == 2, 'the consumer hard-codes Dispatch(2, 1, 1)'
    (directory / 'program').write_bytes(S.fixture.WORDS)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(S.fixture.GROUP, S.RSRC2))
    reference = S.fixture.reference()
    halves = 32 * 8 * 4  # mip 0 comes first in both halves of the reference
    untouched, expected = reference[:halves], reference[len(reference) // 2:][:halves]
    assert sum(a != b for a, b in zip(untouched, expected)) == 64
    layout = '<%dH' % halves

    got, result = run(directory, consumer, env, 'texture-store', layout)
    print(result.stdout, end='')
    assert got == untouched, 'a Dispatch without a translation wrote the texture'
    assert len(lines(result, 'Dispatch(2, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)

    got, result = run(directory, consumer, env, 'texture-store', layout)
    wrong = [i for i in range(halves) if got[i] != expected[i]]
    assert not wrong, 'half=%d got=%04x expected=%04x (%d halves differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]
    layer = clean(result)

    got, result = run(directory, consumer, env, 'texture-store-none', layout)
    assert got == untouched and not lines(result, 'Dispatch('), 'the texture changed without a Dispatch'

    got, result = run(directory, consumer, env, 'texture-store-denied', layout)
    assert got == untouched, 'a refused Dispatch wrote the texture'
    refused = lines(result, 'refused: ')
    assert len(refused) == 1 and 'an output texture view was not created on the desktop' in refused[0], result.stderr[-3000:]

    # A refused creation over a descriptor that holds a live view keeps that view: the Dispatch still writes the texture.
    got, result = run(directory, consumer, env, 'texture-store-occupied', layout)
    wrong = [i for i in range(halves) if got[i] != expected[i]]
    assert not wrong, 'occupied slot: half=%d (%d halves differ)\n%s' % (wrong[0], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]

    # The same when the live token reached the slot by a memory copy in title memory instead of a creation there.
    got, result = run(directory, consumer, env, 'texture-store-copied', layout)
    wrong = [i for i in range(halves) if got[i] != expected[i]]
    assert not wrong, 'copied token: half=%d (%d halves differ)\n%s' % (wrong[0], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]

    # Refusals the runtime already had, each with its one logged reason and an untouched texture: an integer view
    # format, another manifest version, another stage, and (with a manifest line added here for a view the consumer
    # creates) the output texture named as an input too.
    manifest = cache / f'{key:016x}.txt'
    original = manifest.read_text()
    for mode, old, new, reason in (
            ('texture-store-uint', '', '', 'an output texture format is not a float or normalized one'),
            ('texture-store', 'x2w-manifest 2', 'x2w-manifest 3', 'manifest version'),
            ('texture-store', 'stage cs', 'stage ps', 'not a compute manifest')):
        assert not old or original.count(old) == 1, original
        manifest.write_bytes(original.replace(old, new).encode() if old else original.encode())
        got, result = run(directory, consumer, env, mode, layout)
        refused = lines(result, 'refused: ')
        assert got == untouched and len(refused) == 1 and refused[0].endswith('refused: ' + reason), (
            mode, reason, refused, result.stderr[-3000:])
    manifest.write_bytes(original.encode())

    # The output texture is also named as an input (a manifest line added here for a view the consumer creates at table
    # byte 0). The dispatch used to be refused; now the input is bound from a copy made before the dispatch, so the
    # output texture is written exactly as without the input and the debug layer sees no SRV/UAV state conflict.
    assert original.count('uav 0 tex2d') == 1, original
    manifest.write_bytes(original.replace('uav 0 tex2d', 'srv 0 tex2d table 1 0' + chr(10) + 'uav 0 tex2d').encode())
    got, result = run(directory, consumer, env, 'texture-store-self', layout)
    wrong = [i for i in range(halves) if got[i] != expected[i]]
    assert not wrong, 'self input: half=%d (%d halves differ)\n%s' % (wrong[0], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]
    assert len(lines(result, 'compute input texture copied before its own output is written')) == 1, result.stderr[-3000:]
    clean(result)
    # Without the input in the manifest no copy is made.
    manifest.write_bytes(original.encode())
    got, result = run(directory, consumer, env, 'texture-store', layout)
    assert got == expected and not lines(result, 'copied before its own output'), result.stderr[-3000:]

    # The same compiled program under a manifest without entry values (the program reads none of its three): the
    # runtime binds no constants. The manifest is edited here; the translator writes `ud 0` for a root without them.
    manifest = cache / f'{key:016x}.txt'
    kept = [line for line in manifest.read_text().splitlines() if not line.startswith('udsrc ')]
    assert len(kept) == 5 and kept[3] == 'ud 3', kept
    manifest.write_bytes(('\n'.join(kept).replace('ud 3', 'ud 0') + '\n').encode())
    got, result = run(directory, consumer, env, 'texture-store', layout)
    assert got == expected and lines(result, 'translated: 0 entry values, 0 inputs, 1 outputs'), result.stderr[-3000:]
    clean(result)
    print(f'texture output: all {halves} halves match after translation (64 changed) and without entry values; '
          f'untouched before translation, without a Dispatch and for a refused view; debug layer: {layer}')


# A second descriptor, at table byte 128, and a different color in v14.
SECOND_STORE = bytes.fromhex(
    '06080cf4 800000fa '  # s_load_dwordx8 s[32:39], s[12:13], 0x80
    '7fc08cbf '           # s_waitcnt lgkmcnt(0)
    'ff021c7e 00bc0044 ' # v_mov_b32_e32 v14, 0x4400bc00
    '080f20f0 080e0880'  # image_store v[14:15], v[8:9], s[32:39] dmask:0xf dim:SQ_RSRC_IMG_2D d16
)
MIPS = ((32, 8), (16, 4))


def mip_reference(colors):
    halves = []
    for mip, ((width, height), color) in enumerate(zip(MIPS, colors, strict=True)):
        for y in range(height):
            for x in range(width):
                sentinel = 0x5000 + 0x400 * mip + 4 * (y * width + x)
                if color is None or x >= 8 or y >= 2:
                    halves.extend(range(sentinel, sentinel + 4))
                else:
                    tag = 8 * y + x
                    rg = (0x3c00 + tag, 0xc000 + tag) if color == 0 else (0xbc00, 0x4400)
                    halves.extend((*rg, tag, 0x8000 + tag))
    return halves


def check_mips(label, got, expected):
    start = 0
    for mip, (width, height) in enumerate(MIPS):
        for index in range(width * height * 4):
            actual, wanted = got[start + index], expected[start + index]
            pixel, channel = divmod(index, 4)
            y, x = divmod(pixel, width)
            assert actual == wanted, (f'{label}: mip={mip} x={x} y={y} channel={channel} '
                                      f'expected={wanted:04x} actual={actual:04x}')
        start += width * height * 4


def store_two_mips(directory, consumer, dxc, env):
    cache, base = prepare(directory)
    root = C.root_blob(3, table_range=True)
    (directory / 'compute_root').write_bytes(root)
    env = dict(base, **env, X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
    code = S.PREFIX + S.STORE + SECOND_STORE + S.END
    pipeline = G.ComputePipeline(code, root, S.RSRC2, S.fixture.GROUP)
    outputs = [line for line in pipeline.manifest().splitlines() if line.startswith('uav ')]
    assert outputs == [
        'uav 0 tex2d float table 1 64 channels 4',
        'uav 1 tex2d float table 1 128 channels 4',
    ], outputs
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(S.fixture.GROUP, S.RSRC2))
    untouched = mip_reference((None, None))
    layout = '<%dH' % len(untouched)

    got, result = run(directory, consumer, env, 'texture-store-mips', layout)
    print(result.stdout, end='')
    check_mips('missing translation', got, untouched)
    assert len(lines(result, 'Dispatch(2, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)

    expected = mip_reference((0, 1))
    got, result = run(directory, consumer, env, 'texture-store-mips', layout)
    check_mips('ordered views', got, expected)
    assert len(lines(result, 'translated: 3 entry values, 0 inputs, 2 outputs')) == 1, result.stderr[-3000:]
    layer = clean(result)

    got, result = run(directory, consumer, env, 'texture-store-mips-swapped', layout)
    check_mips('swapped views', got, mip_reference((1, 0)))
    clean(result)
    changed = sum(a != b for a, b in zip(got, expected, strict=True))
    assert changed == 64, changed
    print(f'two mip outputs: {len(got)} halves match; swapping views changes {changed} halves; debug layer: {layer}')


def store_3d1d(directory, consumer, dxc, env, form):
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(3, table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    (directory / 'program').write_bytes(V.program(form))
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(V.group(form), V.A.RSRC2))

    volume = form != 'line'
    mode = 'texture-volume' if volume else 'texture-line'
    width, height, depth = 32, 8 if volume else 1, 6 if volume else 1
    plane_words = width * height * 4
    initial = [0xbf000000 | (z << 12) | i for z in range(depth) for i in range(plane_words)]
    expected = initial.copy()
    for y in range(V.group(form)[1]):
        # The 3D view starts at W slice 1 and the shader writes relative z=1.
        start = (2 * plane_words if volume else 0) + y * width * 4
        expected[start:start + 32] = V.VALUES * 8
    layout = '<%dI' % len(initial)
    got, result = run(directory, consumer, env, mode, layout)
    assert got == initial and len(lines(result, 'Dispatch(2, 1, 1) refused: no translation')) == 1
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)
    manifest = (cache / f'{key:016x}.txt').read_text().splitlines()
    dim = '3d' if volume else '1d'
    assert f'uav 0 tex{dim} float table 1 64 channels 4' in manifest, manifest
    got, result = run(directory, consumer, env, mode, layout)
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]
    assert got == expected, (form, sum(a != b for a, b in zip(got, expected, strict=True)))
    clean(result)

    mismatch = initial[:32 * 8 * 4] if volume else [0xbf000000 | i for i in range(32 * 8 * 4)]
    cases = [('texture-array-2d', mismatch, f'an output texture view or resource is not {dim.upper()}')]
    if volume:
        # Manual mutant: skipping the mip-relative W-range check must fail the required refusal.
        cases.append(('texture-volume-bad-w', initial, 'an output 3D texture W range is outside the mip'))
    else:
        # Manual mutant: dropping the 1D mip check before `created` must fail this required refusal.
        cases.append(('texture-line-bad-mip', initial, 'an output 1D texture mip is outside the resource'))
    for bad_mode, untouched, reason in cases:
        got, result = run(directory, consumer, env, bad_mode, '<%dI' % len(untouched))
        refusal = lines(result, 'Dispatch(2, 1, 1) refused:')
        assert len(refusal) == 1 and refusal[0].endswith(reason), result.stderr[-3000:]
        assert got == untouched and not lines(result, 'Dispatch(2, 1, 1) translated')
    print(f'{form}: {len(expected)} mip-one words match; incompatible views refused with sentinels intact')


# v_xor_b32 v5, v5, v12: the assembled `v_xor_b32 v5, v5, v6` of case A with its second source register changed
# (not assembled by llvm-mc here; the translation below is checked to read it as written).
XOR = 'v_xor_b32 v5, v5, v12'
SAMPLED = ('s_mov_b64 s[20:21], 0', 's_mov_b64 s[22:23], 0', 's_load_dwordx8 s[24:31], s[10:11], 0x40',
           'image_sample_lz v[12:15], v[0:1], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_2D', XOR)


def sample(directory, consumer, dxc, env):
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, words, rsrc2, group, groups = C.CASES['A']
    assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
    assert words[96:100].hex() == '050d0a3a', 'case A no longer holds v_xor_b32 v5, v5, v6 at byte 96'
    C.SPLICE[XOR] = '05190a3a'
    code = C.splice(*SAMPLED)
    pipeline = C.translate('A', code=code)
    assert 'v5 = v5 ^ v12;' in pipeline.hlsl_cs() and pipeline.manifest().endswith(
        'srv 0 buf table 1 0\nsrv 1 tex2d table 1 64\nuav 0 table 1 16\nsmp 0 0x0 0x0 0x0 0x0\n'), pipeline.manifest()
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    one = struct.unpack('<I', struct.pack('<f', 1.0))[0]
    written = 24 * 3 * 2 * 2  # threads per group times groups; the words past them keep the sentinel
    untouched, plain = [C.SENTINEL] * C.WORDS, C.expected('A')
    assert plain[written:] == untouched[written:]
    expected = [word ^ one for word in plain[:written]] + plain[written:]
    layout = '<%dI' % C.WORDS

    got, result = run(directory, consumer, env, 'dispatch-sampled', layout)
    assert got == untouched, 'a Dispatch without a translation wrote the output'
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)

    got, result = run(directory, consumer, env, 'dispatch-sampled', layout)
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    layer = clean(result)

    got, result = run(directory, consumer, env, 'dispatch-sampled-compute', layout)  # a compute list and queue
    assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    clean(result)

    got, result = run(directory, consumer, env, 'dispatch-sampled-unbound', layout)
    assert got == untouched, 'a refused Dispatch wrote the output'
    refused = lines(result, 'refused: ')
    assert len(refused) == 1 and 'an input texture entry is not the token of a shader resource view' in refused[0], \
        result.stderr[-3000:]
    print(f'texture input: all {C.WORDS} words match after translation; untouched before translation and without a '
          f'view; debug layer: {layer}')


# The texture input case with a constant sampler instead of the zero one: s21 MAX_LOD 12, s22 linear min, mag and mip
# filters, s20 wrap in U, V and W. A mip filter makes the translator write it as `smpl` (its LOD clamp carried over).
WRAP_LINEAR = (0x0, 0xc00000, 0x18500000, 0x0)
LOD_SAMPLED = SAMPLED[:2] + ('s_mov_b32 s21, 0xc00000', 's_mov_b32 s22, 0x18500000') + SAMPLED[2:]


def filtered(directory, consumer, dxc, env):
    """The sampler bound is the manifest's: a 2x1 texture (red 0, 1.0) read at the left edge (the coordinates are the
    bits of the thread ids, 0 or a denormal) gives red 0.5 only through a linear filter that wraps; clamp gives 0."""
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, _, rsrc2, group, _ = C.CASES['A']
    C.SPLICE.update({XOR: '05190a3a', LOD_SAMPLED[2]: 'ff0395be0000c000', LOD_SAMPLED[3]: 'ff0396be00005018'})
    code = C.splice(*LOD_SAMPLED)
    line = 'smpl 0 %s' % ' '.join('0x%x' % w for w in WRAP_LINEAR)
    assert C.translate('A', code=code).manifest().endswith(line + '\n'), C.translate('A', code=code).manifest()
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    written = 24 * 3 * 2 * 2
    plain = C.expected('A')
    layout = '<%dI' % C.WORDS

    def reds(got):
        assert got[written:] == plain[written:], 'a word past the threads changed'
        return {struct.unpack('<f', struct.pack('<I', g ^ p))[0] for g, p in zip(got[:written], plain[:written])}

    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    assert got == [C.SENTINEL] * C.WORDS, 'a Dispatch without a translation wrote the output'
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key, integer_srvs=True)

    got, result = run(directory, consumer, env, 'dispatch-texel-uint', layout)
    assert got == plain and lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
    assert result.stderr.count('srv 1: integer_view_filtered') == 1, result.stderr[-3000:]
    clean(result)

    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    red = reds(got)
    # 8-bit filter weights may round the half: one step of an 8-bit channel is the tolerance.
    assert len(red) == 1 and abs(next(iter(red)) - 0.5) <= 1 / 255, (sorted(red), result.stderr[-3000:])
    layer = clean(result)

    # Control: the same program under the manifest's sampler changed to clamp in U, V and W reads red 0, which the
    # check above refuses; a sampler heap table stays refused with its reason.
    manifest = cache / f'{key:016x}.txt'
    original = manifest.read_text()
    assert original.count(line) == 1, original
    manifest.write_bytes(original.replace(line, line.replace('smpl 0 0x0 ', 'smpl 0 0x92 ')).encode())
    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and reds(got) == {0.0}, result.stderr[-3000:]
    manifest.write_bytes(original.replace(line, line + '\nsmpheap 2048 1 fallback 2047').encode())
    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    refused = lines(result, 'refused: ')
    assert got == [C.SENTINEL] * C.WORDS and len(refused) == 1 and refused[0].endswith(
        'refused: a sampler from the sampler heap'), (refused, result.stderr[-3000:])
    print(f'filtered input: an smpl wrap and linear sampler reads red 0.5 at the left edge of a 2x1 texture; the same '
          f'program with a clamp sampler reads 0; a heap sampler is refused; debug layer: {layer}')


# The texture input case with a comparison sample: s20 clamp in U and V, TRUNC_COORD and LESS (point filters, LOD
# range 0). The reference is v7 = float(thread id z), u is v8 = float(thread id x) and v is v9 = 0. The five new
# instructions are not assembled by llvm-mc here; the decoder is checked to read them as written.
POINT_LESS = 2 | 2 << 3 | 1 << 27 | 1 << 12
POINT_GREATER = POINT_LESS ^ 1 << 12 | 4 << 12
COMPARED = SAMPLED[:2] + ('s_mov_b32 s20, 0x%x' % POINT_LESS, SAMPLED[2], 'v_cvt_f32_u32_e32 v7, v2',
                          'v_cvt_f32_u32_e32 v8, v0', 'v_mov_b32_e32 v9, 0',
                          'image_sample_c_lz v12, v[7:9], s[24:31], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D',
                          'v_xor_b32_e32 v5, v5, v12')


def compared(directory, consumer, dxc, env):
    """The sampler bound is the manifest's `smpc` one: a 2x1 R32_FLOAT texture (1.0, 2.5) is compared at u = 0 (the
    left texel) or u = 1 (clamped to the right one) with the references 0.0 to 3.0. A word is XORed with the bits of
    1.0f where the CPU model of the comparison passes and is left as case A wrote it where the model fails."""
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, _, rsrc2, group, _ = C.CASES['A']
    C.SPLICE.update({COMPARED[2]: 'ff0394be' + struct.pack('<I', POINT_LESS).hex(), COMPARED[4]: '020d0e7e',
                     COMPARED[5]: '000d107e', COMPARED[6]: '8002127e', COMPARED[7]: '0801bcf0070ca600',
                     COMPARED[8]: '05190a3a'})
    code = C.splice(*COMPARED)
    decoded = [i[3] for i in G.decode(code) if i[0] >= C.STORE_A][:len(COMPARED)]
    assert decoded == list(COMPARED), decoded
    pipeline = C.translate('A', code=code)
    line = 'smpc 0 0x%x 0x0 0x0 0x0' % POINT_LESS
    assert 'v12 = U(r1.SampleCmpLevelZero(q0, float2(F(v8), F(v9)), F(v7)));' in pipeline.hlsl_cs() and \
        pipeline.manifest().endswith('srv 1 tex2d table 1 64\nuav 0 table 1 16\n' + line + '\n'), pipeline.manifest()
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    one = struct.unpack('<I', struct.pack('<f', 1.0))[0]
    written = 24 * 3 * 2 * 2
    plain = C.expected('A')
    layout = '<%dI' % C.WORDS
    depths = (1.0, 2.5)  # the consumer's texels, left and right

    def model(passes):
        # Word i is thread (x, y, z) of a 2x3x4 group: i % 24 == x + 2 * y + 6 * z. Reference z, texel x.
        assert group == (2, 3, 4)
        return [word ^ (one if passes(float(i % 24 // 6), depths[i % 2]) else 0)
                for i, word in enumerate(plain[:written])] + plain[written:]

    def match(got, want, result, what):
        refused = lines(result, 'refused: ')
        assert not refused, f'{what}: {refused[0]}'
        wrong = [i for i in range(C.WORDS) if got[i] != want[i]]
        assert not wrong, '%s: word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            what, wrong[0], got[wrong[0]], want[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]

    less, greater = model(lambda ref, texel: ref < texel), model(lambda ref, texel: ref > texel)
    passing = [sum(a != b for a, b in zip(m, plain)) for m in (less, greater)]
    differ = sum(a != b for a, b in zip(less, greater))
    assert passing == [144, 108] and differ == 252, (passing, differ)

    got, result = run(directory, consumer, env, 'dispatch-sampled-depth', layout)
    assert got == [C.SENTINEL] * C.WORDS, 'a Dispatch without a translation wrote the output'
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)

    got, result = run(directory, consumer, env, 'dispatch-sampled-depth', layout)
    match(got, less, result, 'smpc LESS')
    layer = clean(result)

    # Control: the function is the manifest's. Under GREATER the same program passes where the reference is above
    # the texel, a result the check above refuses at 252 words.
    manifest = cache / f'{key:016x}.txt'
    original = manifest.read_text()
    assert original.count(line) == 1, original
    manifest.write_bytes(original.replace(line, 'smpc 0 0x%x 0x0 0x0 0x0' % POINT_GREATER).encode())
    got, result = run(directory, consumer, env, 'dispatch-sampled-depth', layout)
    match(got, greater, result, 'smpc GREATER')
    clean(result)
    print(f'compared input: an smpc point, clamp, LESS sampler passes at 144 of the {written} written words and all '
          f'{C.WORDS} words match the CPU model; the same program under GREATER matches its model (108 pass, 252 '
          f'words differ from LESS); debug layer: {layer}')


def uniform_image_load_mip(directory, consumer, dxc, env):
    """Compute image_load_mip at uniform coordinates fed into v_readfirstlane_b32 (#2644 T5 follow-up).

    Loads texel (1, 0, mip 0) (red 1.0) of the 2x1 texture via image_load_mip, broadcasts channel 0 with
    v_readfirstlane_b32 s20, v12, and XORs s20 into Case A's stored word. Controls: texel (0, 0, mip 0)
    (red 0.0) and out-of-range mip 1 at (1, 0) each differ at all 288 written words.
    """
    cache, base = prepare(directory)
    env = dict(base, **env)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, _, rsrc2, group, _ = C.CASES['A']
    written = 24 * 3 * 2 * 2
    plain = C.expected('A')
    one = struct.unpack('<I', struct.pack('<f', 1.0))[0]
    expected = [word ^ one for word in plain[:written]] + plain[written:]
    layout = '<%dI' % C.WORDS

    C.SPLICE.update({
        'v_mov_b32_e32 v10, 1': '8102147e',
        'v_mov_b32_e32 v10, 0': '8002147e',
        'v_mov_b32_e32 v11, 0': '8002167e',
        'v_mov_b32_e32 v12, 0': '8002187e',
        'v_mov_b32_e32 v12, 1': '8102187e',
        'image_load_mip v[12:15], v[10:12], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f04f00a0c0600',
        'v_readfirstlane_b32 s20, v12': '0c05287e',
        'v_xor_b32_e32 v5, s20, v5': '140a0a3a',
    })
    load_ops = (
        's_load_dwordx8 s[24:31], s[10:11], 0x40',
        'v_mov_b32_e32 v10, 1',
        'v_mov_b32_e32 v11, 0',
        'v_mov_b32_e32 v12, 0',
        'image_load_mip v[12:15], v[10:12], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D',
        'v_readfirstlane_b32 s20, v12',
        'v_xor_b32_e32 v5, s20, v5',
    )
    code = C.splice(*load_ops)
    pipe = C.translate('A', code=code)
    assert 's20 = v12;' in pipe.hlsl_cs() and 'v5 = s20 ^ v5;' in pipe.hlsl_cs(), pipe.hlsl_cs()
    (directory / 'program').write_bytes(code)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))

    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    assert got == [C.SENTINEL] * C.WORDS, 'a Dispatch without a translation wrote the output'
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key)

    got, result = run(directory, consumer, env, 'dispatch-sampled-2x1', layout)
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    layer = clean(result)

    for tag, ctrl_ops in (
            ('ctrl_x0', (load_ops[0], 'v_mov_b32_e32 v10, 0') + load_ops[2:]),
            ('ctrl_mip1', load_ops[:3] + ('v_mov_b32_e32 v12, 1',) + load_ops[4:])):
        ctrl_dir = directory / tag
        ctrl_cache, ctrl_base = prepare(ctrl_dir)
        ctrl_env = dict(env, **ctrl_base)
        (ctrl_dir / 'compute_root').write_bytes(C.root_blob(table_range=True))
        ctrl_env['X2W_TEST_COMPUTE_ROOT'] = str(ctrl_dir / 'compute_root')
        (ctrl_dir / 'program').write_bytes(C.splice(*ctrl_ops))
        (ctrl_dir / 'data').write_bytes(DATA)
        (ctrl_dir / 'state').write_bytes(state_a(group, rsrc2))
        _, ctrl_res = run(ctrl_dir, consumer, ctrl_env, 'dispatch-sampled-2x1', layout)
        ctrl_key = int(lines(ctrl_res, 'x2w-draw compute request')[0].split()[3], 16)
        translate(ctrl_cache, dxc, ctrl_key)
        got_ctrl, ctrl_res = run(ctrl_dir, consumer, ctrl_env, 'dispatch-sampled-2x1', layout)
        assert got_ctrl == plain and sum(a != b for a, b in zip(got_ctrl, expected)) == written, (
            tag, ctrl_res.stderr[-3000:])
        clean(ctrl_res)
    print(f'uniform image_load_mip + v_readfirstlane_b32: all {C.WORDS} words match texel (1, 0, mip 0); '
          f'x=0 and mip=1 controls each differ at {written} words; debug layer: {layer}')


def identifier_code(dim='2d', explicit_mip=False):
    # Keep the fixture's linear dispatch index and buffer offset, then load
    # texel x = index & 7, y = 0. The last three x values are out of bounds.
    prefix = C.WORDS_A[:C.WORDS_A.index(bytes.fromhex('001030e0'))]
    store = bytearray(C.WORDS_A[C.STORE_A:])
    store[5] = 15
    nsa = dim in ('3d', '2da')
    words = [0xf40c0605, 0xfa000040, vop2(0x1b, 12, 135, 3), vop1(1, 13, 128), vop1(1, 14, 128),
             0xbf8cc07f, 0xf0000100 | DIMENSIONS[dim] << 3 | (2 if nsa else 0) | int(explicit_mip) << 18, 0x00060f0c]
    if nsa:
        words += [0x0e0d]
    return prefix + struct.pack(f'<{len(words) + 1}I', *words, 0xbf8c3f70) + store


def identifier_load(directory, consumer, dxc, env, dim, explicit_mip=False):
    cache, base = prepare(directory)
    env = dict(base, **env, X2W_TEST_DEBUG_LAYER='1', X2W_TEST_IDENTIFIER_DIMENSION=dim)
    root = directory / 'compute_root'
    root.write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(root)
    _, _, rsrc2, group, _ = C.CASES['A']
    (directory / 'program').write_bytes(identifier_code(dim, explicit_mip))
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    layout = '<%dI' % C.WORDS
    written = 24 * 3 * 2 * 2
    untouched = [C.SENTINEL] * C.WORDS

    def expected(texels):
        return list((*texels, 0, 0, 0)) * (written // 8) + untouched[written:]

    got, result = run(directory, consumer, env, 'dispatch-texel-uint', layout)
    assert got == untouched
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key, integer_srvs=True)
    stem = cache / f'{key:016x}.cs'
    for suffix, texels in (('uint', IDENTIFIERS), ('sint', IDENTIFIERS), ('float', FLOAT_BITS)):
        got, result = run(directory, consumer, env, f'dispatch-texel-{suffix}', layout)
        assert clean(result) == 'enabled, no error message'
        assert lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
        assert got == expected(texels), (suffix, got[:16], result.stderr[-3000:])

    source = Path(f'{stem}.hlsl').read_text()
    mutant = directory / 'mutant.hlsl'
    for kind in ('inverted', 'swapped registers'):
        mutant.write_text(typed_mutant(source, kind), encoding='ascii')
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', f'{stem}.cso', mutant], check=True)
        got, result = run(directory, consumer, env, 'dispatch-texel-uint', layout)
        assert lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
        assert got != expected(IDENTIFIERS) or debug_errors(result), (kind, got[:16], result.stderr[-3000:])
    print(f'Integer texture dispatch {dim}: UINT, SINT and FLOAT matched the CPU, both mutants rejected')


def default_typed_load(directory, consumer, dxc, env, dim):
    """A float 3D or 2D-array input translated with no switch set (the default) binds through the typed-SRV line."""
    cache, base = prepare(directory)
    env = dict(base, **env, X2W_TEST_DEBUG_LAYER='1', X2W_TEST_IDENTIFIER_DIMENSION=dim)
    root = directory / 'compute_root'
    root.write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(root)
    _, _, rsrc2, group, _ = C.CASES['A']
    (directory / 'program').write_bytes(identifier_code(dim))
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    layout = '<%dI' % C.WORDS
    written = 24 * 3 * 2 * 2
    untouched = [C.SENTINEL] * C.WORDS
    expected = list((*FLOAT_BITS, 0, 0, 0)) * (written // 8) + untouched[written:]
    got, result = run(directory, consumer, env, 'dispatch-texel-float', layout)
    assert got == untouched
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    translate(cache, dxc, key, integer_srvs=False)
    manifest = cache / f'{key:016x}.txt'
    original = manifest.read_text()
    assert original.count('srvtypes 1\n') == 1, original
    got, result = run(directory, consumer, env, 'dispatch-texel-float', layout)
    assert clean(result) == 'enabled, no error message'
    assert lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
    assert got == expected, (dim, got[:16], result.stderr[-3000:])
    # Control: without the typed-SRV line the runtime refuses the kind, by name, and writes nothing.
    manifest.write_bytes(original.replace('srvtypes 1\n', '').encode())
    got, result = run(directory, consumer, env, 'dispatch-texel-float', layout)
    refused = lines(result, 'refused: ')
    assert got == untouched and len(refused) == 1 and 'not a raw table buffer, a 1D/2D/2D-array/3D texture or a cube map' in refused[0], (
        refused, result.stderr[-3000:])
    print(f'Default typed compute input {dim}: the float texels matched the CPU; without srvtypes the kind is refused')


def main(consumer, dxc, adapter, mode=None):
    env = dict(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        if mode == 'two-mips':
            store_two_mips(directory / 'mips', consumer, dxc, env)
        else:
            store(directory / 'store', consumer, dxc, env)
            for form in V.FORMS:
                store_3d1d(directory / form, consumer, dxc, env, form)
            sample(directory / 'sample', consumer, dxc, env)
            filtered(directory / 'filtered', consumer, dxc, env)
            compared(directory / 'compared', consumer, dxc, env)
            uniform_image_load_mip(directory / 'uniform_image_load_mip', consumer, dxc, env)
            for dim in DIMENSIONS:
                identifier_load(directory / f'identifier-{dim}', consumer, dxc, env, dim)
            identifier_load(directory / 'identifier-1d-mip', consumer, dxc, env, '1d', explicit_mip=True)
            for dim in ('3d', '2da'):
                default_typed_load(directory / f'default-{dim}', consumer, dxc, env, dim)
    print(f'startup_compute_dispatch_texture: PASS on {adapter}')


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5) or sys.argv[3] not in ('warp', 'hardware') or sys.argv[4:] not in ([], ['two-mips']):
        sys.exit(__doc__)
    main(*sys.argv[1:])
