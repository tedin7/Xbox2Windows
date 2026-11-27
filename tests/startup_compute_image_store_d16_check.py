"""Check packed compute image stores from the startup translator on a three-mip RGBA16_FLOAT UAV."""
import argparse
from pathlib import Path
import struct
import tempfile

import compute_mimg_float_store_d16_check as fixture
import startup_compute_image_store_check as A
from startup_gfx2hlsl_compute_check import G, TGID_X, TGID_Y, assemble, refused, root_blob, tidig


# Use #2106's source and CPU reference unchanged. Three constants put the descriptor table at s7.
RSRC2 = (10 << 1) | TGID_X | TGID_Y | tidig(2)
PREFIX, STORE, END = fixture.WORDS[:-12], fixture.WORDS[-12:-4], fixture.WORDS[-4:]
MASK = bytes.fromhex('8002a47d')  # v_cmpx_eq_u32_e32 0, v1
PARTS = ('f16tof32(v14 & 0xffffu)', 'f16tof32(v14 >> 16)',
         'f16tof32(v15 & 0xffffu)', 'f16tof32(v15 >> 16)')
LINE = f'w0[uint2(v8, v9)] = float4({", ".join(PARTS)});'


# Reuse the assembled FP32 array x/y/layer prologue. The packed values
# are four distinct finite nonzero half encodings, low half first.
ARRAY_VALUES = (0x3c00, 0xc000, 0x3800, 0x7bff)
ARRAY_DATA = bytes.fromhex('7fc08cbf ff021c7e003c00c0 ff021e7e0038ff7b')
ARRAY_STORE = bytes.fromhex('280f20f0080e0680')
ARRAY_LINE = LINE.replace('uint2(v8, v9)', 'uint3(v8, v9, v10)')
ARRAY_SOURCE = A.ARRAY_SOURCE.split('s_waitcnt', 1)[0] + """\
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v14, 0xc0003c00
v_mov_b32_e32 v15, 0x7bff3800
{mask}image_store v[14:15], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY d16
s_endpgm
"""


# Three patterns exercise adjacent channels, a gap, and an odd number of halves across two VGPRs.
PARTIAL_CHANNELS = {0x3: (0, 1), 0x9: (0, 3), 0x7: (0, 1, 2)}
PARTIAL_HALVES = (0x4200, 0xbe00, 0x3400, 0x4600)
PARTIAL_DATA = (bytes.fromhex('ff021c7e') + struct.pack('<2H', *PARTIAL_HALVES[:2])
                + bytes.fromhex('ff021e7e') + struct.pack('<2H', *PARTIAL_HALVES[2:]))


def partial_store(mask, store=STORE):
    return store[:1] + bytes([(store[1] & 0xf0) | mask]) + store[2:]


def partial_line(mask, template=LINE):
    components = ['0.0'] * 4
    for channel, value in zip(PARTIAL_CHANNELS[mask], PARTS):
        components[channel] = value
    return template.replace(', '.join(PARTS), ', '.join(components))


def partial_reference(mask):
    full = fixture.reference()
    initial = full[:len(full) // 2]
    expected = initial.copy()
    texel = [0] * 4
    for channel, half in zip(PARTIAL_CHANNELS[mask], PARTIAL_HALVES):
        texel[channel] = half
    for y in range(fixture.GROUP[1]):
        for x in range(fixture.GROUP[0] * fixture.GROUPS):
            start = 4 * (32 * y + x)
            expected[start:start + 4] = texel
    return initial + expected


def array_program(layer, masked=False):
    return A.ARRAY_PROLOGUE + A.ARRAY_SLICE[layer] + ARRAY_DATA + (MASK if masked else b'') + ARRAY_STORE + A.ARRAY_END


def array_controls(llvm_mc):
    for layer in (1, 3):
        for masked in (False, True):
            code = array_program(layer, masked)
            if llvm_mc:
                source = ARRAY_SOURCE.format(layer=layer, mask='v_cmpx_eq_u32_e32 0, v1\n' if masked else '')
                assert assemble(llvm_mc, source) == code, (layer, masked)
            pipe = G.ComputePipeline(code, root_blob(3), RSRC2, fixture.GROUP)
            hlsl = pipe.hlsl_cs()
            assert 'RWTexture2DArray<float4> w0 : register(u0);' in hlsl
            assert ('if (exec & 1) ' if masked else '') + ARRAY_LINE in hlsl
            assert ('if (exec & 1) ' + ARRAY_LINE in hlsl) == masked
            assert f'v10 = 0x{layer:08x}u;' in hlsl
            assert 'uav 0 tex2da float table 1 64 channels 4' in pipe.manifest().splitlines()
    mods = ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY', 'd16']
    for data in ('v14', 'v[14:16]', 'v[14:17]', 'v[255:256]'):
        refused('two VGPRs expected', lambda: pipe.cs.op_image_store([data, 'v[8:10]', 's[24:31]'], mods))
    for addr in ('v[8:9]', 'v[8:11]'):
        refused('x, y and layer expected', lambda: pipe.cs.op_image_store(['v[14:15]', addr, 's[24:31]'], mods))
    refused('fixed table image descriptor', lambda: pipe.cs.op_image_store(['v[14:15]', 'v[8:10]', 's[28:35]'], mods))
    for mask in PARTIAL_CHANNELS:
        code = array_program(1, masked=True).replace(ARRAY_STORE, partial_store(mask, ARRAY_STORE))
        partial = G.ComputePipeline(code, root_blob(3), RSRC2, fixture.GROUP)
        assert 'if (exec & 1) ' + partial_line(mask, ARRAY_LINE) in partial.hlsl_cs()
        assert 'uavload 0' not in partial.manifest().splitlines()
    refused('modifier a16', lambda: pipe.cs.op_image_store(['v[14:15]', 'v[8:10]', 's[24:31]'], mods + ['a16']))
    print('Compute packed D16 array translation and refusal checks passed' + ('; LLVM assembly matched' if llvm_mc else ''))


def pipeline(body):
    return G.ComputePipeline(PREFIX + body + END, root_blob(3), RSRC2, fixture.GROUP)


def controls(llvm_mc):
    def encoded(line, hex_words):
        words = bytes.fromhex(hex_words)
        if llvm_mc:
            assert assemble(llvm_mc, line + '\n') == words, line
        return words

    if llvm_mc:
        assert assemble(llvm_mc, '\n'.join(fixture.LINES) + '\n') == fixture.WORDS
        assert assemble(llvm_mc, 'v_cmpx_eq_u32_e32 0, v1\n') == MASK
    pipe = pipeline(STORE)
    assert pipe.manifest().splitlines()[-1] == 'uav 0 tex2d float table 1 64 channels 4'
    assert LINE in pipe.hlsl_cs() and 'RWTexture2D<float4> w0 : register(u0);' in pipe.hlsl_cs()
    assert 'if (exec & 1) ' + LINE in pipeline(MASK + STORE).hlsl_cs()
    mods = ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D', 'd16']
    for data in ('v14', 'v[14:16]', 'v[255:256]'):
        refused('two VGPRs expected', lambda: pipe.cs.op_image_store([data, 'v[8:9]', 's[24:31]'], mods))
    refused('fixed table image descriptor',
            lambda: pipe.cs.op_image_store(['v[14:15]', 'v[8:9]', 's[28:35]'], mods))

    assert G.lifted_family('image_store with dmask 0x3: a 16-bit store writes all four channels') == 'dmask'
    partial_mods = ['dmask:0x3', *mods[1:]]
    refused('one VGPRs expected',
            lambda: pipe.cs.op_image_store(['v[14:15]', 'v[8:9]', 's[24:31]'], partial_mods))
    for mask, channels in PARTIAL_CHANNELS.items():
        data = 'v14' if len(channels) == 2 else 'v[14:15]'
        asm = fixture.STORE.replace('v[14:15]', data).replace('dmask:0xf', f'dmask:{mask:#x}')
        store = partial_store(mask)
        if llvm_mc:
            assert assemble(llvm_mc, asm + '\n') == store, asm
        partial = pipeline(PARTIAL_DATA + store)
        assert partial.hlsl_cs().count(partial_line(mask)) == 1
        assert partial.hlsl_cs().count('w0[uint2(v8, v9)]') == 1
        assert 'uavload 0' not in partial.manifest().splitlines()
        assert 'if (exec & 1) ' + partial_line(mask) in pipeline(MASK + store).hlsl_cs()

    for asm, code, reason in (
        (fixture.STORE.replace('v[8:9]', 'v8').replace(' d16', ' a16 d16'), '080f20f0080e06c0', 'modifier a16'),
        (fixture.STORE.replace('v[8:9]', 'v[8:10]').replace('IMG_2D', 'IMG_3D'),
         '100f20f0080e0680', 'only 2D images'),
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
    array_controls(llvm_mc)
    print('Compute D16 store, EXEC, raw/texture binding order, alias and refusal checks passed')


def dispatch(directory, compiler, renderer, adapter):
    source = pipeline(STORE).hlsl_cs()
    assert source.count(LINE) == 1
    def swapped(order):
        return source.replace(', '.join(PARTS), ', '.join(PARTS[i] for i in order))
    cases = (
        ('full EXEC', source, 0, None),
        ('omit store', source.replace(LINE, ''), 64, 'mip=0 half=0 expected=3c00 actual=5000'),
        ('swap first two halves', swapped((1, 0, 2, 3)), 32, 'mip=0 half=0 expected=3c00 actual=c000'),
        ('swap packed registers', swapped((2, 3, 0, 1)), 64, 'mip=0 half=0 expected=3c00 actual=0000'),
    )
    payload = fixture.reference()
    if fixture.draw_cases(directory, compiler, renderer, adapter, payload, cases) == 77:
        return 77
    # v1 is the local Y. Only row zero of the two dispatched rows may change.
    halves = len(payload) // 2
    row = 32 * 4
    payload[halves + row:halves + 2 * row] = payload[row:2 * row]
    masked = pipeline(MASK + STORE).hlsl_cs()
    guard = 'if (exec & 1) ' + LINE
    assert masked.count(guard) == 1
    cases = (('partial EXEC', masked, 0, None),
             ('drop EXEC guard', masked.replace(guard, LINE), 32, 'mip=0 half=128 expected=5080 actual=3c08'))
    if fixture.draw_cases(directory, compiler, renderer, adapter, payload, cases) == 77:
        return 77
    for mask, channels in PARTIAL_CHANNELS.items():
        source = pipeline(PARTIAL_DATA + partial_store(mask)).hlsl_cs()
        line = partial_line(mask)
        assert source.count(line) == 1
        # Read-modify-write would retain nonzero sentinels and fail the zero-channel comparison.
        first_zero = next(c for c in range(4) if c not in channels)
        mismatches = (4 - len(channels)) * fixture.GROUP[0] * fixture.GROUP[1] * fixture.GROUPS
        cases = (
            (f'D16 mask {mask:#x}', source, 0, None),
            (f'D16 mask {mask:#x}, fill unselected channels with one',
             source.replace(line, line.replace('0.0', '1.0')), mismatches,
             f'mip=0 half={first_zero} expected=0000 actual=3c00'),
        )
        if fixture.draw_cases(directory, compiler, renderer, adapter, partial_reference(mask), cases) == 77:
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
