"""Read back a gfx1013 compute image_gather4_lz and image_gather4_lz_o through the startup translator."""

import argparse
import math
from pathlib import Path
import struct
import tempfile

from compute_gather4_l_check import HEIGHT, ORDER, SENTINEL, WIDTH, first_mismatch, run_case, texel
from startup_gfx2hlsl_compute_check import G, USER7, WORDS, assemble, refused, root_blob
from startup_compute_gather_l_check import s_mov_word, v_mov_word


SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[16:23], s[10:11], 0
s_load_dwordx4 s[28:31], s[10:11], 32
s_mov_b32 s24, 0x12
s_mov_b32 s25, 0
s_mov_b32 s26, 0x04500000
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0.5
v_mov_b32 v2, 0
s_waitcnt lgkmcnt(0)
image_gather4_lz v[4:7], v[0:1], s[16:23], s[24:27] dmask:{mask:#x} dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], v2, s[28:31], 0 offen
s_endpgm
"""
PREFIX = bytes.fromhex('06038abe80038bbe05040cf4000000fa050708f4200000fa'
                       '920398be800399beff039abe0000500480039bbe'
                       'f002007ef002027e8002047e7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf001078e002040780000081bf')
CHANNELS = ('Red', 'Green', 'Blue', 'Alpha')

# name, X mode, Y mode, border color, channel, view mip, coordinates, unnormalized source, NSA
ADDRESS_CASES = (
    ('wrap', 0, 0, 0, 0, 0, (0.0, 1.0), None, False),
    ('mirror', 1, 1, 0, 1, 0, (1.25, -0.25), None, False),
    ('mirror_once', 3, 3, 0, 2, 0, (-0.25, -0.625), None, False),
    ('white_border', 6, 2, 2, 2, 0, (0.0, 0.625), None, False),
    ('opaque_black_border', 2, 6, 1, 3, 0, (0.625, 1.0), None, False),
    ('transparent_border', 6, 2, 0, 3, 0, (1.0, 0.625), None, False),
    ('unnormalized_sampler', 2, 2, 0, 3, 1, (5.25, 2.25), 'sampler', False),
    ('unnormalized_instruction', 2, 2, 0, 2, 0, (6.25, 3.25), 'instruction', False),
    ('nsa', 2, 2, 0, 1, 0, (0.25, 0.625), None, True),
)


def float_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def address_program(w0, w3, channel, uv, nsa, llvm_mc=None, *, w2=0x04100000, unorm=False):
    # Point minification, linear magnification. v1 remains zero in the NSA case.
    y = 9 if nsa else 1
    coords = [float_word(c) for c in uv]
    prefix = (PREFIX[:24] + s_mov_word(24, w0) + s_mov_word(25, 0) +
              s_mov_word(26, w2) + s_mov_word(27, w3) +
              v_mov_word(0, hex(coords[0])) + v_mov_word(y, hex(coords[1])) + v_mov_word(2, '0') + WAIT)
    first = 0xf11c0008 | (0x100 << channel) | (2 if nsa else 0) | (1 << 12 if unorm else 0)
    instruction = struct.pack('<II', first, 0x00c40400)
    if nsa:
        instruction += bytes((y, 0, 0, 0))
    if llvm_mc:
        addr = '[v0, v9]' if nsa else 'v[0:1]'
        source = (f'image_gather4_lz v[4:7], {addr}, s[16:23], s[24:27] '
                  f'dmask:{1 << channel:#x} dim:SQ_RSRC_IMG_2D')
        if unorm:
            source += ' unorm'
        assert instruction == assemble(llvm_mc, source + '\n'), source
    return prefix + instruction + SUFFIX

# image_gather4_lz_o: v0 holds the packed texel offset (signed X in bits 5:0, signed Y in bits 13:8), v1 and v2 the
# coordinates; the output offset moves to v8 because v2 is now an address register.
SOURCE_O = SOURCE.replace('v_mov_b32 v0, 0.5\nv_mov_b32 v1, 0.5\nv_mov_b32 v2, 0\n',
                          'v_mov_b32 v0, {packed:#x}\nv_mov_b32 v1, 0.5\nv_mov_b32 v2, 0.5\nv_mov_b32 v8, 0\n'
                          ).replace('image_gather4_lz v[4:7], v[0:1]', 'image_gather4_lz_o v[4:7], v[0:2]'
                                    ).replace('v[4:7], v2, s[28:31]', 'v[4:7], v8, s[28:31]')
PREFIX_O = PREFIX[:-16]
SUFFIX_O = SUFFIX.replace(bytes.fromhex('02040780'), bytes.fromhex('08040780'))


def pack_offset(x, y):
    return (x & 0x3f) | ((y & 0x3f) << 8)


def shader_x(mask, movs, nsa=False, prefix=PREFIX_O):
    """image_gather4_lz_o after `movs`; nsa lists the coordinate registers v1 and v9 in the non-sequential form."""
    word = struct.pack('<II', 0xf15c0008 | (mask << 8) | (2 if nsa else 0), 0x00c40400)
    return prefix + movs + word + (bytes.fromhex('01090000') if nsa else b'') + SUFFIX_O


MOV_V1, MOV_V2, MOV_V8, MOV_V9 = (bytes.fromhex(x) for x in ('f002027e', 'f002047e', '8002107e', 'f002127e'))
WAIT = bytes.fromhex('7fc08cbf')


def literal_move(value):
    return bytes.fromhex('ff02007e') + struct.pack('<I', value)


def add_literal(value):  # v_add_nc_u32 v0, <literal>, v0
    return bytes.fromhex('ff00004a') + struct.pack('<I', value)


def shader_o(mask, packed, prefix=PREFIX_O, op=0xf15c0008):
    movs = bytes.fromhex('ff02007e') + struct.pack('<I', packed) + bytes.fromhex('f002027ef002047e8002107e7fc08cbf')
    return prefix + movs + struct.pack('<II', op | (mask << 8), 0x00c40400) + SUFFIX_O


def shader(mask, prefix=PREFIX):
    return prefix + struct.pack('<II', 0xf11c0008 | (mask << 8), 0x00c40400) + SUFFIX


def pipeline(code):
    return G.ComputePipeline(code, root_blob(), USER7, (1, 1, 1))


def controls(llvm_mc):
    for channel, name in enumerate(CHANNELS):
        mask = 1 << channel
        code = shader(mask)
        if llvm_mc:
            assert code == assemble(llvm_mc, SOURCE.format(mask=mask))
        pipe = pipeline(code)
        assert f'r0.Gather{name}(q0,' in pipe.hlsl_cs()
        assert pipe.manifest().splitlines()[-3:] == [
            'srv 0 tex2d table 1 0', 'uav 0 table 1 32', 'smp 0 0x12 0x0 0x4500000 0x0']
    refused('one-hot dmask', lambda: pipeline(shader(3)))
    # Set MIN_LOD to 1/256, outside the constant sampler profile.
    bad_sampler = PREFIX.replace(bytes.fromhex('800399be'), bytes.fromhex('810399be'))
    refused('unsupported filter, LOD or descriptor flags', lambda: pipeline(shader(2, bad_sampler)))
    missing_sampler = PREFIX.replace(bytes.fromhex('80039bbe'), b'')
    refused('constant sampler', lambda: pipeline(shader(2, missing_sampler)))
    # A four-word load at the image slot omits the descriptor's high half.
    short_image = PREFIX.replace(bytes.fromhex('05040cf4'), bytes.fromhex('050408f4'))
    refused('modified image descriptor', lambda: pipeline(shader(2, short_image)))
    no_image = PREFIX.replace(bytes.fromhex('05040cf4000000fa'), b'')
    refused('fixed table image descriptor', lambda: pipeline(shader(2, no_image)))
    # s_mov_b32 s22, 0 overwrites word 6 while the first word still identifies the table slot.
    refused('modified image descriptor', lambda: pipeline(shader(2, PREFIX + bytes.fromhex('800396be'))))
    # Point the raw output and the texture input at the same table slot.
    aliased = PREFIX.replace(bytes.fromhex('050708f4200000fa'), bytes.fromhex('050708f4000000fa'))
    refused('used for a load and a store', lambda: pipeline(shader(2, aliased)))
    check_image_provenance(llvm_mc)
    for _, xmode, ymode, border, channel, _, uv, unnormalized, nsa in ADDRESS_CASES:
        w0 = xmode | ymode << 3 | int(unnormalized == 'sampler') << 15
        w3 = border << 30
        pipe = pipeline(address_program(w0, w3, channel, uv, nsa, llvm_mc, unorm=unnormalized == 'instruction'))
        coords = 'float2(F(v0), F(v9))' if nsa else 'float2(F(v0), F(v1))'
        if unnormalized:
            assert 'r0.GetDimensions(gw, gh);' in pipe.hlsl_cs()
            coords += ' / float2(gw, gh)'
        assert f'r0.Gather{CHANNELS[channel]}(q0, {coords})' in pipe.hlsl_cs()
        assert f'smp 0 {w0:#x} 0x0 0x4100000 {w3:#x}\n' in pipe.manifest()
    for mode in (4, 5, 7):
        refused('unsupported X/Y address mode', lambda: pipeline(address_program(mode, 0, 0, (0.25, 0.625), False)))
    refused('register border color requires a palette',
            lambda: pipeline(address_program(6, 3 << 30, 0, (0.25, 0.625), False)))
    pipe = linear_mip_pipeline(llvm_mc)
    assert 'r0.GatherGreen(q0, float2(F(v0), F(v1)))' in pipe.hlsl_cs()
    assert 'smp 0 0x12 0x0 0x8500000 0x0\n' in pipe.manifest()
    print('Four compute gather channels, manifest bindings and refusal controls passed')


def check_image_provenance(llvm_mc):
    def encoded(source, hex_words):
        words = bytes.fromhex(hex_words)
        if llvm_mc:
            assert assemble(llvm_mc, source + '\n') == words, source
        return words

    # The gather still reads the intact s[16:23]. An update through a second load
    # of the same table slot marks the low or high four-word key in pipe.modified.
    duplicate = encoded('s_load_dwordx8 s[40:47], s[10:11], 0', '050a0cf4000000fa')
    assert 'srv 0 tex2d table 1 0' in pipeline(shader(2, PREFIX + duplicate)).manifest()
    for reg, words in ((41, '29ff298800000400'), (46, '2eff2e8800000400')):
        change = encoded(f's_or_b32 s{reg}, s{reg}, 0x40000', words)
        refused('modified image descriptor', lambda: pipeline(shader(2, PREFIX + duplicate + change)))

    # Fetch the output first since the wide loads can overwrite the table pointer.
    # With s[12:27] the gather starts at loaded word 4 and straddles two descriptors.
    # With s[8:23] it starts at word 8, the complete second descriptor (byte offset 32).
    original = bytes.fromhex('05040cf4000000fa050708f4200000fa')
    output = encoded('s_load_dwordx4 s[28:31], s[10:11], 64', '050708f4400000fa')
    middle = encoded('s_load_dwordx16 s[12:27], s[10:11], 0', '050310f4000000fa')
    aligned = encoded('s_load_dwordx16 s[8:23], s[10:11], 0', '050210f4000000fa')
    refused('fixed table image descriptor', lambda: pipeline(shader(2, PREFIX.replace(original, output + middle))))
    manifest = pipeline(shader(2, PREFIX.replace(original, output + aligned))).manifest().splitlines()
    assert 'srv 0 tex2d table 1 32' in manifest and 'uav 0 table 1 64' in manifest, manifest


def controls_o(llvm_mc):
    for channel, name in enumerate(CHANNELS):
        packed = pack_offset(1, -2)
        code = shader_o(1 << channel, packed)
        if llvm_mc:
            assert code == assemble(llvm_mc, SOURCE_O.format(mask=1 << channel, packed=packed)), name
        pipe = pipeline(code)
        assert f'r0.Gather{name}(q0, float2(F(v1), F(v2)), int2(1, -2))' in pipe.hlsl_cs()
        assert pipe.manifest().splitlines()[-3:] == [
            'srv 0 tex2d table 1 0', 'uav 0 table 1 32', 'smp 0 0x12 0x0 0x4500000 0x0']
    refused('one-hot dmask', lambda: pipeline(shader_o(3, 0)))
    # Offsets the literal Gather cannot take are emulated with Load: out of range, set outside X and Y,
    # not written by one literal move, or written twice.
    for text, code in (
            ('out of range', shader_o(2, pack_offset(8, 0))),
            ('bits outside X and Y', shader_o(2, 0x10000)),
            ('never written', shader_o(2, 0).replace(bytes.fromhex('ff02007e00000000'), b'')),
            ('written twice', shader_x(2, literal_move(0x100) + add_literal(0x3) + MOV_V1 + MOV_V2 + MOV_V8 + WAIT))):
        source = pipeline(code).hlsl_cs()
        assert 'GetDimensions' in source and '.Gather' not in source, text
    # Non-sequential address: the offset and both coordinates are named, in order.
    source = pipeline(shader_x(2, literal_move(pack_offset(1, -2)) + MOV_V1 + MOV_V9 + MOV_V8 + WAIT, True)).hlsl_cs()
    assert 'GatherGreen(q0, float2(F(v1), F(v9)), int2(1, -2))' in source, source
    print('image_gather4_lz_o offset forms and refusals passed')


def reference_o(level, channel, ox, oy):
    # The footprint of (0.5, 0.5) moved by the texel offset; the sampler clamps addresses to the mip.
    width, height = WIDTH >> level, HEIGHT >> level
    x, y = width // 2 - 1 + ox, height // 2 - 1 + oy
    taps = [texel(level, min(max(x + dx, 0), width - 1), min(max(y + dy, 0), height - 1), channel)
            for dx, dy in ORDER]
    return taps + [SENTINEL] * (WORDS - len(taps))


def dispatch_o(directory, compiler, renderer, adapter):
    for level, channel, ox, oy in ((0, 1, 1, -2), (0, 0, -3, 2), (1, 2, 2, 1), (0, 3, 7, 7), (0, 1, -8, -8)):
        packed = pack_offset(ox, oy)
        source = pipeline(shader_o(1 << channel, packed)).hlsl_cs()
        result = run_case(directory, compiler, renderer, adapter, f'o_mip{level}_{channel}_{ox}_{oy}', source,
                          reference_o(level, channel, ox, oy), level)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout

    movs = MOV_V1 + MOV_V2 + MOV_V8 + WAIT
    for level, channel, ox, oy, code in (
            # Out of the literal range, still one literal move.
            (0, 1, 9, -5, shader_x(2, literal_move(pack_offset(9, -5)) + movs)),
            (0, 0, -12, 10, shader_x(1, literal_move(pack_offset(-12, 10)) + movs)),
            # Computed at run time: written twice (0x100 + 0x203 = X 3, Y 3).
            (0, 2, 3, 3, shader_x(4, literal_move(0x100) + add_literal(0x203) + movs)),
            (1, 3, -2, 1, shader_x(8, literal_move(pack_offset(-3, 1)) + add_literal(1) + movs)),
            # Non-sequential address [v0, v1, v9]; v2 is left unset, so a contiguous read finds v = 0.
            (0, 1, 1, -2, shader_x(2, literal_move(pack_offset(1, -2)) + MOV_V1 + MOV_V9 + MOV_V8 + WAIT, True))):
        result = run_case(directory, compiler, renderer, adapter, f'o_dyn_{level}_{channel}_{ox}_{oy}',
                          pipeline(code).hlsl_cs(), reference_o(level, channel, ox, oy), level)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (ox, oy)

    want = reference_o(0, 1, 1, -2)
    source = pipeline(shader_o(2, pack_offset(1, -2))).hlsl_cs()
    emulated = pipeline(shader_x(2, literal_move(0x100) + add_literal(pack_offset(1, -2) - 0x100) + movs)).hlsl_cs()
    assert 'GetDimensions' in emulated
    for name, text, actual in (
        ('o_dropped_offset', source.replace('int2(1, -2)', 'int2(0, 0)'), reference_o(0, 1, 0, 0)),
        ('o_swapped_axes', source.replace('int2(1, -2)', 'int2(-2, 1)'), reference_o(0, 1, -2, 1)),
        ('o_wrong_sign', source.replace('int2(1, -2)', 'int2(-1, 2)'), reference_o(0, 1, -1, 2)),
        ('o_emulated_no_offset', emulated.replace('+ go;', ';'), reference_o(0, 1, 0, 0)),
        ('o_emulated_swapped_rows', emulated.replace('gb + int2(0, 1)', 'gb + int2(0, 0)', 1).replace(
            'clamp(gb, 0, gs - 1)', 'clamp(gb + int2(0, 1), 0, gs - 1)'), None),
    ):
        if actual is None:
            actual = want.copy()
            actual[0], actual[3] = want[3], want[0]
        result = run_case(directory, compiler, renderer, adapter, name, text, want, 0)
        mismatches = sum(a != b for a, b in zip(want, actual))
        assert mismatches and result.returncode == 1 and first_mismatch(want, actual) in result.stderr, name
        assert f'compared_words={WORDS} mismatches={mismatches}' in result.stdout, name
    return 0


def reference(level, channel):
    # At (0.5, 0.5) the footprint straddles the four center texels of the visible mip.
    x, y = (WIDTH >> level) // 2 - 1, (HEIGHT >> level) // 2 - 1
    taps = [texel(level, x + dx, y + dy, channel) for dx, dy in ORDER]
    return taps + [SENTINEL] * (WORDS - len(taps))


def address_texel(index, extent, mode):
    if mode == 0:
        return index % extent
    if mode == 1:
        index %= 2 * extent
        return min(index, 2 * extent - index - 1)
    if mode == 3 and index < 0:
        index = -index - 1
    if mode == 6 and not 0 <= index < extent:
        return None
    return min(max(index, 0), extent - 1)


def linear_mip_pipeline(llvm_mc=None):
    return pipeline(address_program(0x12, 0, 1, (0.5, 0.5), False, llvm_mc, w2=0x08500000))


def address_reference(level, channel, uv, xmode, ymode, border, unnormalized):
    width, height = WIDTH >> level, HEIGHT >> level
    u, v = uv if unnormalized else (uv[0] * width, uv[1] * height)
    x, y = math.floor(u - 0.5), math.floor(v - 0.5)
    border_word = float_word(float(border == 2 or (border == 1 and channel == 3)))
    taps = []
    for dx, dy in ORDER:
        i, j = address_texel(x + dx, width, xmode), address_texel(y + dy, height, ymode)
        taps.append(border_word if i is None or j is None else texel(level, i, j, channel))
    return taps + [SENTINEL] * (WORDS - len(taps))


def dispatch_addresses(directory, compiler, renderer, adapter):
    for name, xmode, ymode, border, channel, level, uv, unnormalized, nsa in ADDRESS_CASES:
        w0 = xmode | ymode << 3 | int(unnormalized == 'sampler') << 15
        w3 = border << 30
        good = pipeline(address_program(w0, w3, channel, uv, nsa, unorm=unnormalized == 'instruction'))
        want = address_reference(level, channel, uv, xmode, ymode, border, unnormalized)
        result = run_case(directory, compiler, renderer, adapter, name, good.hlsl_cs(), want, level,
                          manifest=good.manifest())
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, name

        if nsa:
            # Keep v9 initialized but read v1 through the packed address encoding.
            code = address_program(w0, w3, channel, uv, True)
            end = len(code) - len(SUFFIX)
            instruction = struct.pack('<II', 0xf11c0008 | (0x100 << channel), 0x00c40400)
            wrong = pipeline(code[:end - 12] + instruction + SUFFIX)
            actual = address_reference(level, channel, (uv[0], 0.0), 2, 2, 0, False)
        else:
            # Replace the address mode with clamp or drop unnormalized coordinates.
            wrong = pipeline(address_program(0x12, w3, channel, uv, False))
            actual = address_reference(level, channel, uv, 2, 2, border, False)
        result = run_case(directory, compiler, renderer, adapter, name + '_mutant', wrong.hlsl_cs(), want, level,
                          manifest=wrong.manifest())
        mismatches = sum(a != b for a, b in zip(want, actual))
        assert mismatches and result.returncode == 1 and first_mismatch(want, actual) in result.stderr, name
        assert f'compared_words={WORDS} mismatches={mismatches}' in result.stdout, name
    pipe = linear_mip_pipeline()
    want = reference(0, 1)
    result = run_case(directory, compiler, renderer, adapter, 'linear_mip', pipe.hlsl_cs(), want, 0,
                      manifest=pipe.manifest())
    if result.returncode == 77:
        return 77
    assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout
    result = run_case(directory, compiler, renderer, adapter, 'linear_mip_wrong_view', pipe.hlsl_cs(), want, 1,
                      manifest=pipe.manifest())
    assert result.returncode == 1 and first_mismatch(want, reference(1, 1)) in result.stderr
    assert 'compared_words=320 mismatches=4' in result.stdout
    return 0


def dispatch(directory, compiler, renderer, adapter):
    for level, channel in ((0, 1), (1, 0), (1, 1), (1, 2), (1, 3)):
        source = pipeline(shader(1 << channel)).hlsl_cs()
        result = run_case(directory, compiler, renderer, adapter, f'mip{level}_{CHANNELS[channel]}',
                          source, reference(level, channel), level)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout

    source = pipeline(shader(2)).hlsl_cs()
    want = reference(1, 1)
    swapped = want.copy()
    swapped[0], swapped[3] = want[3], want[0]
    gather = 't = r0.GatherGreen(q0, float2(F(v0), F(v1)));'
    assert source.count(gather) == 1
    for name, text, level, actual in (
        ('dropped_gather', source.replace(gather, ''), 1, [0] * 4 + want[4:]),
        ('wrong_component', source.replace('.GatherGreen(', '.GatherRed('), 1, reference(1, 0)),
        ('wrong_first_mip', source, 0, reference(0, 1)),
        ('wrong_tap_order', source.replace('uint4(v4, v5, v6, v7)', 'uint4(v7, v5, v6, v4)'), 1, swapped),
    ):
        result = run_case(directory, compiler, renderer, adapter, name, text, want, level)
        mismatches = sum(a != b for a, b in zip(want, actual))
        assert result.returncode == 1 and first_mismatch(want, actual) in result.stderr, name
        assert f'compared_words={WORDS} mismatches={mismatches}' in result.stdout, name
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
            code = (dispatch(Path(tmp), *args.dispatch) or dispatch_addresses(Path(tmp), *args.dispatch)
                    or dispatch_o(Path(tmp), *args.dispatch))
            raise SystemExit(code)
    controls(args.llvm_mc)
    controls_o(args.llvm_mc)
