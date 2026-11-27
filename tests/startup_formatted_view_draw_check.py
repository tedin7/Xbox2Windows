"""Read typed buffer views the title created through a translated buffer_load_format_xyzw, bit for bit.

usage: startup_formatted_view_draw_check.py <startup_stream_consumer> <warp|hardware> <fxc>

A source-built gfx1013 pixel stage loads one typed-buffer element per output pixel. The consumer
creates a four-element typed view through the stand-in, then reads back a four-pixel float target.
The CPU reference decodes the uploaded bytes independently of the runtime and translator tables.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl
from startup_stream_draw_check import vertex, vop1

PIXEL = container(0, [
    0xf4080304, 0xfa000000,  # s_load_dwordx4 s[12:15], s[8:9], 0x0
    0xbf8cc07f,              # s_waitcnt lgkmcnt(0)
    vop1(1, 4, 129),         # v_mov_b32 v4, 1
    0xe00c2000, 0x80030004,  # buffer_load_format_xyzw v[0:3], v4, s[12:15], 0 idxen
    0xbf8c3f70,              # s_waitcnt vmcnt(0)
    0xf800180f, 0x03020100,  # exp mrt0 v0, v1, v2, v3 done vm
    0xbf810000])             # s_endpgm
TARGET = 2  # DXGI_FORMAT_R32G32B32A32_FLOAT
IDENTITY = 0x1688
F = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]  # noqa: E731
ONE, CLEAR = F(1.0), (0, 0, F(1.0), F(1.0))
UNSUPPORTED = 'view format has no decoded buffer FORMAT'
BGRA = 87
ELEMENTS = 4


def mapping(*sources):
    """D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING: 0-3 channels, 4 forced 0, 5 forced 1."""
    return sum(s << 3 * i for i, s in enumerate(sources)) | 1 << 12


def decode(payload, widths, kind, sources=(0, 1, 2, 3), order=None):
    """Decode four little-endian elements to binary32 pixels, casting UINT values to float."""
    stride = sum(widths) // 8
    result = []
    for offset in range(0, ELEMENTS * stride, stride):
        packed = int.from_bytes(payload[offset:offset + stride], 'little')
        channels = []
        for width in widths:
            mask = (1 << width) - 1
            value, packed = packed & mask, packed >> width
            if kind == 'float':
                value = struct.unpack('<e' if width == 16 else '<f', value.to_bytes(width // 8, 'little'))[0]
            elif kind == 'unorm':
                value /= mask
            elif kind == 'snorm':
                if value & (1 << (width - 1)):
                    value -= 1 << width
                value = max(-1.0, value / (mask >> 1))
            channels.append(F(value))
        if order:
            channels = [channels[i] for i in order]
        channels += [0, 0, 0, ONE][len(channels):] + [0, ONE]
        result.extend(channels[i] for i in sources)
    return tuple(result)


def make_case(label, fmt, widths, kind, payload, sources=(0, 1, 2, 3), order=None):
    return label, fmt, mapping(*sources), 0, payload.ljust(64, b'\0'), decode(payload, widths, kind, sources, order)


def legacy_cases():
    floats = (1.5, -2.25, 3.125, 0.0625)
    for fmt, count, width in ((2, 4, 32), (6, 3, 32), (10, 4, 16), (16, 2, 32), (34, 2, 16), (41, 1, 32)):
        samples = [floats[(i + c) % 4] for i in range(4) for c in range(count)]
        quantized = struct.pack('<' + ('e' if width == 16 else 'f') * len(samples), *samples)
        yield make_case('FLOAT DXGI %d' % fmt, fmt, (width,) * count, 'float', quantized)
    for fmt, count, width in ((28, 4, 8), (35, 2, 16), (49, 2, 8), (61, 1, 8)):
        max_value = (1 << width) - 1
        samples = [max_value if (i + c) % 3 else 0 for i in range(4) for c in range(count)]
        yield make_case('UNORM DXGI %d' % fmt, fmt, (width,) * count, 'unorm',
                        struct.pack('<' + ('B' if width == 8 else 'H') * len(samples), *samples))
    yield make_case('R10G10B10A2_UNORM', 24, (10, 10, 10, 2), 'unorm',
                    struct.pack('<4I', 0, 0x3ff, 0xfffffc00, 0xffffffff))
    yield make_case('R16G16_SNORM', 37, (16, 16), 'snorm',
                    struct.pack('<8h', -32768, 0, 32767, -32767, 0, 32767, -1, 1))
    for fmt, count, width in ((30, 4, 8), (36, 2, 16), (62, 1, 8)):
        samples = [((i + c) * 11) for i in range(4) for c in range(count)]
        yield make_case('UINT DXGI %d' % fmt, fmt, (width,) * count, 'uint',
                        struct.pack('<' + ('B' if width == 8 else 'H') * len(samples), *samples))


LEGACY = tuple(legacy_cases())
LEGACY_SWIZZLES = (
    make_case('float4 with constants', 2, (32,) * 4, 'float', LEGACY[0][4], (3, 4, 0, 5)),
    make_case('RGBA8 with constants', 28, (8,) * 4, 'unorm', bytes((0, 255, 255, 0)) * ELEMENTS, (2, 5, 4, 3)),
)
BGRA_BYTES = bytes((255, 0, 19, 126, 0, 255, 65, 1, 48, 35, 255, 0, 99, 17, 0, 255))
BGRA_CASES = (
    make_case('BGRA identity', BGRA, (8,) * 4, 'unorm', BGRA_BYTES, order=(2, 1, 0, 3)),
    make_case('BGRA with constants', BGRA, (8,) * 4, 'unorm', BGRA_BYTES, (2, 5, 4, 0), (2, 1, 0, 3)),
)
REFUSALS = (
    ('R32_UINT', 42, IDENTITY, 0, bytes(64), UNSUPPORTED),
    ('raw view', 39, IDENTITY, 1, bytes(64), 'not a typed buffer view'),
)


def spec(case):
    _, fmt, mapping_word, raw, payload, _ = case
    return '%d:%x:%d:%s' % (fmt, mapping_word, raw, payload.hex())


def checked_pixels(actual, expected, label, ulps=0):
    assert len(actual) == len(expected), (label, len(actual), len(expected))
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert abs(got - want) <= ulps, (label, i // 4, i % 4, hex(got), hex(want))


def four_pixel_source(source):
    # Replace the synthetic shader's constant index with the output column. Each pixel reads a different element.
    source, edits = re.subn(r'(?m)^( +v4 = )0x00000001u;$', r'\g<1>uint(i.pos.x);', source)
    assert edits == 1, 'synthetic index assignment missing'
    selector = re.search(r'\(\((ud\d+) >> 9\) & 7u\)', source)
    assert selector, 'no W selector in the translation'
    # Render the fixture's UINT values as numeric floats, avoiding subnormal bitcasts at the output.
    source, edits = re.subn(r'(?m)^( +)return o;$',
                            r'\g<1>uint fmt = (%s >> 12) & 127u;\n'
                            r'\g<1>if (fmt == 5u || fmt == 27u || fmt == 60u) o.mrt0 = float4(v0, v1, v2, v3);\n'
                            r'\g<1>return o;' % selector.group(1), source)
    assert edits == 1, 'synthetic export missing'
    return source, selector.group(0), selector.group(1)


def check(work, consumer, backend, fxc):
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(PIXEL)
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc)
    for control in ('X2W_TEST_FORMATTED_REPEAT', 'X2W_TEST_FORMATTED_SWAP_23_24'):
        env.pop(control, None)
    args = [consumer, vs, ps, root, backend, output, str(TARGET)]

    def draw(case, enabled=True, repeat=False, swap=False):
        draw_env = dict(env, X2W_TEST_FORMATTED_VIEW=spec(case), X2W_D3D12_FORMATTED_BGRA='1' if enabled else '0')
        if repeat:
            draw_env['X2W_TEST_FORMATTED_REPEAT'] = '1'
        if swap:
            draw_env['X2W_TEST_FORMATTED_SWAP_23_24'] = '1'
        output.unlink(missing_ok=True)
        result = subprocess.run(args, env=draw_env, capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (result.returncode, result.stderr[-1500:])
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr[-3000:]
        return struct.unpack('<16I', output.read_bytes()), result.stderr

    def drawn(case, enabled=True, swap=False):
        pixels, log = draw(case, enabled, swap=swap)
        assert '-> drawn;' in log and 'srv 0: formatted buffer refused' not in log, (case[0], log[-2000:])
        return pixels

    def refused(case, reason, enabled=True):
        pixels, log = draw(case, enabled, repeat=True)
        label, fmt, component_map, *_ = case
        expected = (f'{reason}; DXGI format={fmt} elements=4 mapping=0x{component_map:x}'
                    if enabled else f'{reason} 0')
        logs = re.findall(r'srv 0: formatted buffer refused \((.+)\)', log)
        assert '-> drawn;' not in log and logs == [expected] * (1 if enabled else 2), (label, logs)
        checked_pixels(pixels, CLEAR * ELEMENTS, label)

    # Publish the base request and its float target variant. Keep the translator's decode and descriptor ABI.
    for _ in range(2):
        draw(LEGACY[0])
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    manifests = [p for p in cache.glob('*.txt') if 'descword3' in p.read_text()]
    assert manifests and all('srv 0 buf table 0 0' in p.read_text() for p in manifests)
    variant = next(p for p in manifests if 'rtv1' in p.read_text())
    hlsl = variant.with_suffix('.ps.hlsl')
    source = hlsl.read_text()
    source, w_select, word3 = four_pixel_source(source)

    def compile_pixel(text):
        hlsl.write_text(text)
        subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', variant.with_suffix('.ps.cso'), hlsl],
                       check=True, capture_output=True)

    compile_pixel(source)
    for case in (*LEGACY, *LEGACY_SWIZZLES):
        off = drawn(case, enabled=False)
        checked_pixels(off, case[5], case[0], ulps=1 if case[1] == 37 else 0)
        checked_pixels(drawn(case), off, case[0])
    for case in BGRA_CASES:
        checked_pixels(drawn(case), case[5], case[0], ulps=1)
    refused(BGRA_CASES[0], UNSUPPORTED, enabled=False)
    for case in REFUSALS:
        refused(case, case[5])

    # Swap adjacent FORMAT codes in the stand-in, then prove the signed view fails its reference.
    snorm = next(case for case in LEGACY if case[1] == 37)
    wrong = drawn(snorm, swap=True)
    checked_pixels(wrong, decode(snorm[4], (16, 16), 'unorm'), 'mutated SNORM as UNORM', ulps=1)
    assert wrong[0] != snorm[5][0], 'swapped UNORM/SNORM code produced the signed reference'

    compile_pixel(source.replace(w_select, '((%s >> 6) & 7u)' % word3))
    words = drawn(LEGACY[0])
    expected = LEGACY[0][5]
    assert words != expected and words[3::4] == expected[2::4], words
    print(f'{backend}: 15 legacy formats identical, two BGRA mappings, three refusals, two readback controls')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    if not Path(sys.argv[3]).is_file():
        print('fxc unavailable')
        sys.exit(77)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
