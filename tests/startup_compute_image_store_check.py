"""Check 32-bit float compute image stores (2D, unorm and 2D array) from the startup translator."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import compute_mimg_float_store_check as fixture
from startup_gfx2hlsl_compute_check import G, TGID_X, TGID_Y, assemble, refused, root_blob, tidig


# Reuse #2105's source and CPU reference unchanged. Three constants put the descriptor table at s7.
RSRC2 = (10 << 1) | TGID_X | TGID_Y | tidig(2)
PREFIX, STORE, END = fixture.WORDS[:-12], fixture.WORDS[-12:-4], fixture.WORDS[-4:]
STORE_UNORM = bytes.fromhex('081f20f0080e0600')  # image_store v[14:17], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
MASK = bytes.fromhex('8002a47d')  # v_cmpx_eq_u32_e32 0, v1
LINE = 'w0[uint2(v8, v9)] = asfloat(uint4(v14, v15, v16, v17));'

# The 2D-array store (#2742) takes x, y and the layer in v8-v10 and needs no SRV read, so its four data
# words are constants. The bound array has four layers and three mips: layer 1 is in the middle of the
# range and layer 3 its last entry. tests/compute_check.cpp selects the layered RWTexture2DArray view.
# The data words are not LLVM inline float constants: 0x40000000 (2.0) and 0x40800000 (4.0) would be
# assembled as the four-byte inline form instead of the eight-byte literal below, so the values follow
# the literal forms that tests/startup_image_2d_array_check.py and this fixture's own prologue already
# pin against llvm-mc-15.
ARRAY_STORE = bytes.fromhex('280f20f0080e0600')  # image_store v[14:17], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
ARRAY_FLAG = 1 << 31  # tests/compute_check.cpp texture_array_flag
ARRAY_LAYERS = 4
ARRAY_VALUES = (0x3f400000, 0x3e800000, 0x40810000, 0xbf810001)
ARRAY_LINE = 'w0[uint3(v8, v9, v10)] = asfloat(uint4(v14, v15, v16, v17));'
ARRAY_PROLOGUE = bytes.fromhex(
    '07038cbe'                 # s_mov_b32 s12, s7
    '80038dbe'                 # s_mov_b32 s13, 0
    '06060cf4400000fa'         # s_load_dwordx8 s[24:31], s[12:13], 0x40
    '080046d70a040104'         # v_lshl_add_u32 v8, s10, 2, v0
    '090046d70b020504')        # v_lshl_add_u32 v9, s11, 1, v1
ARRAY_SLICE = {1: bytes.fromhex('8102147e'), 3: bytes.fromhex('8302147e')}  # v_mov_b32_e32 v10, 1 or 3
ARRAY_END = bytes.fromhex('000081bf')  # s_endpgm
ARRAY_DATA = bytes.fromhex(
    '7fc08cbf'                 # s_waitcnt lgkmcnt(0)
    'ff021c7e0000403f'         # v_mov_b32_e32 v14, 0x3f400000
    'ff021e7e0000803e'         # v_mov_b32_e32 v15, 0x3e800000
    'ff02207e00008140'         # v_mov_b32_e32 v16, 0x40810000
    'ff02227e010081bf')        # v_mov_b32_e32 v17, 0xbf810001
ARRAY_SOURCE = """\
s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
v_lshl_add_u32 v8, s10, 2, v0
v_lshl_add_u32 v9, s11, 1, v1
v_mov_b32_e32 v10, {layer}
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v14, 0x3f400000
v_mov_b32_e32 v15, 0x3e800000
v_mov_b32_e32 v16, 0x40810000
v_mov_b32_e32 v17, 0xbf810001
image_store v[14:17], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_endpgm
"""


# 1D and 3D stores (c22): same data constants as the array case. The 1D store takes x in v8, the 3D store x, y and a
# constant slice z in v8-v10. Words assembled with llvm-mc-19 (the --llvm-mc check compares them again).
HEAD = ARRAY_PROLOGUE[:24]  # s_mov s12, s7; s_mov s13, 0; s_load_dwordx8 s[24:31]; v_lshl_add_u32 v8, s10, 2, v0
STORE_1D = bytes.fromhex('000f20f0080e0600')  # image_store v[14:17], v8, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_1D
STORE_3D = bytes.fromhex('100f20f0080e0600')  # image_store v[14:17], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_3D
LINE_1D = 'w0[v8] = asfloat(uint4(v14, v15, v16, v17));'
LINE_3D = 'w0[uint3(v8, v9, v10)] = asfloat(uint4(v14, v15, v16, v17));'
TEXTURE_1D_FLAG = 1 << 23  # tests/compute_check.cpp texture_1d_flag
VOLUME_FLAG = (1 << 31) | (1 << 30)  # texture_volume_flag: RWTexture3D, RGBA32 float
SOURCE_1D = HEAD_SOURCE = """\
s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
v_lshl_add_u32 v8, s10, 2, v0
"""
SOURCE_3D_Z = """\
v_lshl_add_u32 v9, s11, 1, v1
v_mov_b32_e32 v10, {z}
"""
SOURCE_DATA = ARRAY_SOURCE.split('v_mov_b32_e32 v10, {layer}\n')[1].split('image_store')[0]


def program_1d():
    return HEAD + ARRAY_DATA + STORE_1D + ARRAY_END


def program_3d(z):
    return ARRAY_PROLOGUE + ARRAY_SLICE.get(z, bytes.fromhex('8102147e' if z == 1 else '8302147e')) + ARRAY_DATA + STORE_3D + ARRAY_END


def pipeline_1d():
    return G.ComputePipeline(program_1d(), root_blob(3), RSRC2, fixture.GROUP)


def pipeline_3d(z):
    return G.ComputePipeline(program_3d(z), root_blob(3), RSRC2, fixture.GROUP)


def sentinel(mip, z, y, x, c):
    return 0xbf000000 | mip << 16 | z << 12 | y << 7 | x << 2 | c


def reference_1d():
    """Initial then expected words of a three-mip 1D image; only mip zero x < 8 is stored to."""
    initial, expected = [], []
    for mip in range(3):
        for x in range(32 >> mip):
            sentinels = [sentinel(mip, 0, 0, x, c) for c in range(4)]
            initial.extend(sentinels)
            expected.extend(ARRAY_VALUES if mip == 0 and x < 4 * fixture.GROUPS else sentinels)
    return initial, expected


def reference_3d(z):
    """Initial then expected words of a three-mip volume (depth 4, 2, 1); only mip zero slice z, x < 8, y < 2 is stored to."""
    initial, expected = [], []
    for mip in range(3):
        for slice_ in range(4 >> mip):
            for y in range(8 >> mip):
                for x in range(32 >> mip):
                    sentinels = [sentinel(mip, slice_, y, x, c) for c in range(4)]
                    initial.extend(sentinels)
                    hit = mip == 0 and slice_ == z and x < 4 * fixture.GROUPS and y < 2
                    expected.extend(ARRAY_VALUES if hit else sentinels)
    return initial, expected


def pipeline(body):
    return G.ComputePipeline(PREFIX + body + END, root_blob(3), RSRC2, fixture.GROUP)


def array_program(layer):
    return ARRAY_PROLOGUE + ARRAY_SLICE[layer] + ARRAY_DATA + ARRAY_STORE + ARRAY_END


def array_pipeline(layer):
    return G.ComputePipeline(array_program(layer), root_blob(3), RSRC2, fixture.GROUP)


def array_reference(layer):
    """Initial then expected words of the four-layer, three-mip array; only mip zero of `layer` is stored to."""
    initial, expected = [], []
    for z in range(ARRAY_LAYERS):
        for mip in range(3):
            for y in range(8 >> mip):
                for x in range(32 >> mip):
                    tag = mip << 16 | z << 12 | y << 7 | x << 2
                    sentinels = [0xbf000000 | tag | c for c in range(4)]
                    initial.extend(sentinels)
                    if mip == 0 and z == layer and x < 4 * fixture.GROUPS and y < 2:
                        expected.extend(ARRAY_VALUES)
                    else:
                        expected.extend(sentinels)
    return initial, expected


def controls(llvm_mc):
    def encoded(line, hex_words):
        words = bytes.fromhex(hex_words)
        if llvm_mc:
            assert assemble(llvm_mc, line + '\n') == words, line
        return words

    if llvm_mc:
        assert assemble(llvm_mc, '\n'.join(fixture.LINES) + '\n') == fixture.WORDS
        assert assemble(llvm_mc, fixture.STORE + ' unorm\n') == STORE_UNORM
        assert assemble(llvm_mc, 'v_cmpx_eq_u32_e32 0, v1\n') == MASK
        for layer in (1, 3):
            assert assemble(llvm_mc, ARRAY_SOURCE.format(layer=layer)) == array_program(layer), layer
        assert assemble(llvm_mc, SOURCE_1D + SOURCE_DATA + 'image_store v[14:17], v8, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_1D\ns_endpgm\n') == program_1d()
        for z in (1, 3):
            assert assemble(llvm_mc, SOURCE_1D + SOURCE_3D_Z.format(z=z) + SOURCE_DATA +
                            'image_store v[14:17], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_3D\ns_endpgm\n') == program_3d(z), z

    pipe = pipeline(STORE)
    assert pipe.manifest().splitlines()[-2:] == [
        'srv 0 tex2d table 1 96',
        'uav 0 tex2d float table 1 64 channels 4',
    ]
    assert LINE in pipe.hlsl_cs() and 'RWTexture2D<float4> w0 : register(u0);' in pipe.hlsl_cs()
    # unorm is a no-op on image_store (integer texel coordinates) and emits identical HLSL and manifest.
    unorm_pipe = pipeline(STORE_UNORM)
    assert unorm_pipe.hlsl_cs() == pipe.hlsl_cs()
    assert unorm_pipe.manifest() == pipe.manifest()
    assert 'if (exec & 1) ' + LINE in pipeline(MASK + STORE_UNORM).hlsl_cs()

    # 2D array (#2742): x, y and the layer are indexed as uint3 into a RWTexture2DArray bound as tex2da.
    for layer in (1, 3):
        array = array_pipeline(layer)
        hlsl, manifest = array.hlsl_cs(), array.manifest()
        first = layer * 1344
        initial, expected = array_reference(layer)
        assert 'RWTexture2DArray<float4> w0 : register(u0);' in hlsl, hlsl
        assert ARRAY_LINE in hlsl and 'RWTexture2D<' not in hlsl, hlsl
        assert f'v10 = 0x{layer:08x}u;' in hlsl, hlsl
        assert 'uav 0 tex2da float table 1 64 channels 4' in manifest.splitlines(), manifest
        # The four VDATA constants land at mip zero of that layer only; the other 5,312 words keep their
        # sentinels. Layer 1 is in the middle of the four-layer range and layer 3 is its last entry.
        assert len(initial) == len(expected) == ARRAY_LAYERS * 1344, (len(initial), len(expected))
        assert expected[first:first + 4] == list(ARRAY_VALUES), expected[first:first + 4]
        changed = [i for i, (before, after) in enumerate(zip(initial, expected)) if before != after]
        assert len(changed) == 16 * 4 and changed[0] == first and changed[-1] == first + 128 + 31, changed[:1]
    # An eight-byte literal is encoded only for a value LLVM does not take as an inline constant: a pattern
    # such as 0x40000000 (2.0) or 0x40800000 (4.0) would assemble to the four-byte inline form instead, and
    # only the --llvm-mc comparison would notice. Keep the hand-written data words clear of that table.
    assert not set(ARRAY_VALUES) & set(G.D.INLINE_FLOAT_LITERALS), G.D.INLINE_FLOAT_LITERALS
    # The array address is exactly x, y and layer: a two- or four-register address is refused.
    array_mods = ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY']
    for address in ('v[8:9]', 'v[8:11]'):
        refused('x, y and layer expected',
                lambda: pipe.cs.op_image_store(['v[14:17]', address, 's[24:31]'], array_mods))

    # 1D (x) and 3D (x, y, z): RWTexture1D and RWTexture3D bound as tex1d and tex3d.
    one = pipeline_1d()
    assert 'uav 0 tex1d float table 1 64 channels 4' in one.manifest().splitlines(), one.manifest()
    assert 'RWTexture1D<float4> w0 : register(u0);' in one.hlsl_cs() and LINE_1D in one.hlsl_cs(), one.hlsl_cs()
    for z in (1, 3):
        volume = pipeline_3d(z)
        assert 'uav 0 tex3d float table 1 64 channels 4' in volume.manifest().splitlines(), volume.manifest()
        assert 'RWTexture3D<float4> w0 : register(u0);' in volume.hlsl_cs() and LINE_3D in volume.hlsl_cs(), volume.hlsl_cs()
        assert f'v10 = 0x{z:08x}u;' in volume.hlsl_cs(), volume.hlsl_cs()
        initial, expected = reference_3d(z)
        assert len(initial) == len(expected) == (32 * 8 * 4 + 16 * 4 * 2 + 8 * 2) * 4
        assert sum(a != b for a, b in zip(initial, expected)) == 16 * 4, z
    initial, expected = reference_1d()
    assert len(initial) == (32 + 16 + 8) * 4 and sum(a != b for a, b in zip(initial, expected)) == 8 * 4
    assert not set(ARRAY_VALUES) & set(G.D.INLINE_FLOAT_LITERALS)
    # The address count is exact: 1D takes x only, 3D takes x, y and z, and a 1D array, cube or 2D MSAA is still refused.
    for address in ('v[8:9]', 'v[8:10]'):
        refused('x expected', lambda: one.cs.op_image_store(['v[14:17]', address, 's[24:31]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_1D']))
    for address in ('v[8:9]', 'v[8:11]'):
        refused('x, y and z expected',
                lambda: one.cs.op_image_store(['v[14:17]', address, 's[24:31]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_3D']))
    for dim in ('SQ_RSRC_IMG_1D_ARRAY', 'SQ_RSRC_IMG_CUBE', 'SQ_RSRC_IMG_2D_MSAA'):
        refused('only 2D images', lambda: one.cs.op_image_store(['v[14:17]', 'v[8:10]', 's[24:31]'], ['dmask:0xf', 'dim:' + dim]))
    # A pixel stage keeps the 2D-only contract.

    mods = ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D', 'unorm']
    for data in ('v14', 'v[14:15]', 'v[14:16]', 'v[253:256]'):
        refused('four VGPRs expected', lambda: pipe.cs.op_image_store([data, 'v[8:9]', 's[24:31]'], mods))
    refused('fixed table image descriptor',
            lambda: pipe.cs.op_image_store(['v[14:17]', 'v[8:9]', 's[28:35]'], mods))

    for asm, code, reason in (
        (fixture.STORE.replace('v[14:17]', 'v[14:15]').replace('dmask:0xf', 'dmask:0x3'),
         '080320f0080e0600', 'typed UAV store writes all four channels'),
        (fixture.STORE.replace('v[8:9]', 'v8') + ' a16', '080f20f0080e0640', 'modifier a16'),
    ):
        instruction = encoded(asm, code)
        refused(reason, lambda: pipeline(instruction))


    wait = encoded('s_waitcnt lgkmcnt(0)', '7fc08cbf')
    changed = encoded('s_mov_b32 s26, 0', '80039abe')
    refused('modified image descriptor', lambda: pipeline(changed + STORE))
    duplicate = encoded('s_load_dwordx8 s[40:47], s[12:13], 0x40', '060a0cf4400000fa')
    changed_alias = encoded('s_or_b32 s46, s46, 0x40000', '2eff2e8800000400')
    refused('modified image descriptor', lambda: pipeline(duplicate + wait + changed_alias + STORE))

    raw = encoded('buffer_store_dword v14, v0, s[32:35], 0 offen', '001070e0000e0880')
    read = encoded('buffer_load_dword v5, v0, s[32:35], 0 offen', '001030e000050880')
    for offset in (64, 128):
        load = encoded(f's_load_dwordx4 s[32:35], s[12:13], {offset}', f'060808f4{offset:02x}0000fa')
        for body in (raw + STORE, STORE + raw):
            if offset == 64:
                refused('used as', lambda: pipeline(load + wait + body))
                continue
            mixed = pipeline(load + wait + body)
            raw_slot, image_slot = (0, 1) if body.startswith(raw) else (1, 0)
            assert f'uav {raw_slot} table 1 128' in mixed.manifest().splitlines()
            assert f'uav {image_slot} tex2d float table 1 64 channels 4' in mixed.manifest().splitlines()
            assert f'RWByteAddressBuffer u{raw_slot} : register(u{raw_slot});' in mixed.hlsl_cs()
            assert f'RWTexture2D<float4> w{image_slot} : register(u{image_slot});' in mixed.hlsl_cs()
        if offset == 64:
            for body in (read + STORE, STORE + read):
                refused('used for a load and a store', lambda: pipeline(load + wait + body))
    print('Compute 32-bit float image_store, unorm, 1D, 2D array, 3D, EXEC, raw/texture binding order, alias and '
          'refusal checks passed')


def draw_cases(directory, compiler, renderer, adapter, payload, cases, selector=None, compared=1344):
    selector = fixture.GROUPS | fixture.FLOAT_TEXTURE_FLAG if selector is None else selector
    hlsl, cso, case = (directory / f'image.{ext}' for ext in ('hlsl', 'cso', 'case'))
    case.write_bytes(struct.pack(f'<2I{len(payload)}I', selector, 0, *payload))
    for name, source, expected_mismatches, needle in cases:
        hlsl.write_text(source, encoding='ascii')
        subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, '--compute-texture-store-case', cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f'{name}:\n{result.stdout}{result.stderr}', end='', flush=True)
        if result.returncode == 77:
            return 77
        if 'debug_layer=enabled' in result.stdout:
            assert 'debug_errors=0 debug_warnings=0' in result.stdout, name
        assert f'compared_words={compared} mismatches={expected_mismatches}' in result.stdout, name
        if expected_mismatches == 0:
            assert result.returncode == 0, name
        else:
            assert result.returncode == 1 and needle in result.stderr, name
    return 0


def dispatch(directory, compiler, renderer, adapter):
    source = pipeline(STORE_UNORM).hlsl_cs()
    assert source.count(LINE) == 1
    cases = (
        ('full EXEC (unorm)', source, 0, None),
        ('omit store', source.replace(LINE, ''), 64, 'mip=0 word=0 expected=3f100000 actual=bf000000'),
        ('swap R/B channels',
         source.replace('uint4(v14, v15, v16, v17)', 'uint4(v15, v14, v16, v17)'),
         32, 'mip=0 word=0 expected=3f100000 actual=3f100004'),
        ('swap RG/BA pairs',
         source.replace('uint4(v14, v15, v16, v17)', 'uint4(v16, v17, v14, v15)'),
         64, 'mip=0 word=0 expected=3f100000 actual=40810000'),
    )
    payload = fixture.reference()
    if draw_cases(directory, compiler, renderer, adapter, payload, cases) == 77:
        return 77
    # v1 is the local Y. Only row zero of the two dispatched rows may change.
    words_per_image = len(payload) // 3
    row = 32 * 4
    payload[words_per_image + row:words_per_image + 2 * row] = payload[row:2 * row]
    masked = pipeline(MASK + STORE_UNORM).hlsl_cs()
    guard = 'if (exec & 1) ' + LINE
    assert masked.count(guard) == 1
    cases = (
        ('partial EXEC', masked, 0, None),
        ('drop EXEC guard', masked.replace(guard, LINE), 32, 'mip=0 word=128 expected=bf001000 actual=3f101000'),
    )
    if draw_cases(directory, compiler, renderer, adapter, payload, cases) == 77:
        return 77
    # 2D-array stores: layer 1 is in the middle of the four-layer range and layer 3 its last entry. The
    # expected words carry the four VDATA constants at mip zero of that layer only. Storing one layer off
    # must change both the wrong subresource and the untouched target, so each control differs by 128 words.
    for layer, shift, needle in (
        (1, '+ 1', 'mip=0 word=1344 expected=3f400000 actual=bf001000'),
        (3, '- 1', 'mip=0 word=2688 expected=bf002000 actual=3f400000'),
    ):
        initial, expected = array_reference(layer)
        source = array_pipeline(layer).hlsl_cs()
        assert source.count(ARRAY_LINE) == 1
        cases = (
            ('layer %d, full EXEC' % layer, source, 0, None),
            ('layer %d, omit store' % layer, source.replace(ARRAY_LINE, ''), 64,
             'mip=0 word=%d expected=3f400000 actual=%08x'
             % (layer * 1344, 0xbf000000 | layer << 12)),
            ('layer %d, one layer off' % layer,
             source.replace('uint3(v8, v9, v10)', 'uint3(v8, v9, v10 %s)' % shift), 128, needle),
        )
        if draw_cases(directory, compiler, renderer, adapter, initial + expected, cases,
                      selector=fixture.GROUPS | ARRAY_FLAG, compared=len(initial)) == 77:
            return 77
    # 1D: x only, so a mutant that stores one texel off or omits the store changes the readback. The renderer's 1D case
    # carries a third block (the SRV image), which this program does not read.
    initial, expected = reference_1d()
    source = pipeline_1d().hlsl_cs()
    assert source.count(LINE_1D) == 1
    cases = (
        ('1D, full EXEC', source, 0, None),
        ('1D, omit store', source.replace(LINE_1D, ''), 32, 'mip=0 word=0 expected=3f400000 actual=bf000000'),
        ('1D, one texel off', source.replace('w0[v8]', 'w0[v8 + 1]'), 8, 'mip=0 word=0 expected=3f400000 actual=bf000000'),
    )
    if draw_cases(directory, compiler, renderer, adapter, initial + expected + [0] * len(initial), cases,
                  selector=fixture.GROUPS | TEXTURE_1D_FLAG, compared=len(initial)) == 77:
        return 77
    # 3D: slice 1 is inside the depth range and slice 3 is its last entry; one slice off must change both slices.
    for z, shift in ((1, '+ 1'), (3, '- 1')):
        initial, expected = reference_3d(z)
        source = pipeline_3d(z).hlsl_cs()
        assert source.count(LINE_3D) == 1
        first = z * 32 * 8 * 4  # the slice's first word in mip zero
        wrong = z + (1 if shift == '+ 1' else -1)
        cases = (
            ('3D slice %d, full EXEC' % z, source, 0, None),
            ('3D slice %d, omit store' % z, source.replace(LINE_3D, ''), 64,
             'mip=0 word=%d expected=3f400000 actual=%08x' % (first, sentinel(0, z, 0, 0, 0))),
            ('3D slice %d, one slice off' % z, source.replace('uint3(v8, v9, v10)', 'uint3(v8, v9, v10 %s)' % shift), 128,
             'mip=0 word=%d expected=%08x actual=%08x' % (min(z, wrong) * 1024, 0x3f400000 if z < wrong else sentinel(0, wrong, 0, 0, 0),
                                                           sentinel(0, z, 0, 0, 0) if z < wrong else 0x3f400000)),
            ('3D slice %d, x and y swapped' % z, source.replace('uint3(v8, v9, v10)', 'uint3(v9, v8, v10)'), 96,
             'mip=0 word=%d expected=3f400000 actual=%08x' % (first + 8, sentinel(0, z, 0, 2, 0))),
        )
        if draw_cases(directory, compiler, renderer, adapter, initial + expected, cases,
                      selector=fixture.GROUPS | VOLUME_FLAG, compared=len(initial)) == 77:
            return 77
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    args = parser.parse_args()
    if args.dispatch:
        out = Path(__file__).resolve().parents[1] / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls(args.llvm_mc)
