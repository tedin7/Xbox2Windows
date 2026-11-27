"""Read typed buffer views the title created through a translated buffer_load_format_xyzw, bit for bit.

usage: startup_formatted_view_draw_check.py <startup_stream_consumer> <warp|hardware> <fxc> [--extended]

A source-built gfx1013 pixel stage loads its descriptor from table entry 0, reads element 1 with
buffer_load_format_xyzw (idxen) and exports the four registers to an R32G32B32A32_FLOAT target. The view behind the
entry is made by tests/startup_address_srv.cpp (draw_formatted_view) through the title-side CreateShaderResourceView,
so the draw has no hardware descriptor word 3: it builds FORMAT and DST_SEL from the view's DXGI format and component
mapping (src/startup_stubs/buffer_format_word3.h). One translation serves every case; each draw is compared as four
32-bit words. Views the translation cannot decode refuse the draw with one logged reason, and the
target keeps its clear value. A translation whose W selector reads the Z field must fail.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_buffer_format_word3_table_check import ADDED
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
SENTINELS = (1.5, -2.25, 3.125, 0.0625)  # distinct per channel; exact in binary32 and binary16
HALVES = (0x3e00, 0xc080, 0x4240, 0x2c00)  # the same four values in binary16


def mapping(*sources):
    """D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING: 0-3 channels, 4 forced 0, 5 forced 1."""
    return sum(s << 3 * i for i, s in enumerate(sources)) | 1 << 12


def data(offset, payload):
    raw = bytearray(64)
    raw[offset:offset + len(payload)] = payload
    return bytes(raw)


# (label, DXGI format, mapping, raw view, 64 buffer bytes, expected words or a refusal reason). Element 1 sits at
# byte 16 (16-byte elements), 8 (8-byte) or 4 (4-byte).
CASES = (
    ('R32G32B32A32_FLOAT', 2, IDENTITY, 0, data(16, struct.pack('<4f', *SENTINELS)), tuple(map(F, SENTINELS))),
    ('R16G16B16A16_FLOAT', 10, IDENTITY, 0, data(8, struct.pack('<4H', *HALVES)), tuple(map(F, SENTINELS))),
    ('R16G16_FLOAT, missing z and w', 34, IDENTITY, 0, data(4, struct.pack('<2H', *HALVES[:2])),
     (F(1.5), F(-2.25), 0, ONE)),
    ('R16G16_SNORM, -32768 and 0', 37, IDENTITY, 0, data(4, struct.pack('<2H', 0x8000, 0)), (F(-1.0), 0, 0, ONE)),
    ('R32G32B32A32_FLOAT as (w, 0, x, 1)', 2, mapping(3, 4, 0, 5), 0, data(16, struct.pack('<4f', *SENTINELS)),
     (F(0.0625), 0, F(1.5), ONE)),
    ('R8G8B8A8_UNORM as (z, 1, 0, w)', 28, mapping(2, 5, 4, 3), 0, data(4, bytes(4)), (0, ONE, 0, 0)),
    ('R32_UINT', 42, IDENTITY, 0, data(4, b'\x2a\0\0\0'), 'view format has no decoded buffer FORMAT'),
    ('raw view', 39, IDENTITY, 1, data(16, struct.pack('<4f', *SENTINELS)), 'not a typed buffer view'),
)


# A separate stage for the expanded matrix: table descriptor in s[20:23],
# element index 2 in v8, and result in v[24:27]. The UINT target preserves
# full integer words and floating-point bit patterns through colour export.
EXTENDED_PIXEL = container(0, [
    0xf4080504, 0xfa000000,  # s_load_dwordx4 s[20:23], s[8:9], 0x0
    vop1(1, 8, 130),         # v_mov_b32 v8, 2
    0xbf8cc07f,              # s_waitcnt lgkmcnt(0)
    0xe00c2000, 0x80051808,  # buffer_load_format_xyzw v[24:27], v8, s[20:23], 0 idxen
    0xbf8c3f70,              # s_waitcnt vmcnt(0)
    0xf800180f, 0x1b1a1918,  # exp mrt0 v24, v25, v26, v27 done vm
    0xbf810000,
])


def pack_element(fields, widths):
    bits, shift = 0, 0
    for field, width in zip(fields, widths):
        bits |= field << shift
        shift += width
    return bits.to_bytes(shift // 8, 'little')


def cpu_decode(element, widths, kind):
    packed = int.from_bytes(element, 'little')
    one = 1 if kind in ('UINT', 'SINT') else F(1.0)
    components = [0, 0, 0, one]
    for index, width in enumerate(widths):
        raw = packed & ((1 << width) - 1)
        packed >>= width
        signed = raw - (1 << width) if raw >> (width - 1) else raw
        if kind == 'SINT':
            word = signed & 0xffffffff
        elif kind == 'UNORM':
            word = F(raw / ((1 << width) - 1))
        elif kind == 'SNORM':
            word = F(max(-1.0, signed / ((1 << (width - 1)) - 1)))
        elif kind == 'FLOAT' and width == 16:
            word = F(struct.unpack('<e', raw.to_bytes(2, 'little'))[0])
        elif kind == 'FLOAT':
            mantissa_bits = width - 5
            exponent, mantissa = divmod(raw, 1 << mantissa_bits)
            if exponent == 0:
                word = F(mantissa * 2.0 ** (-14 - mantissa_bits))
            else:
                word = F((1 + mantissa / (1 << mantissa_bits)) * 2.0 ** (exponent - 15))
        else:
            word = raw
        components[index] = word
    return components + [0, one]


def expanded_cases():
    for label, (fmt, _) in ADDED.items():
        channels, kind = label.split('_')
        widths = [int(width) for width in re.findall(r'\d+', channels)]
        if label == 'R11G11B10_FLOAT':
            fields = [0x2e9, 0x031, 0x225]  # distinct channels, including a subnormal G
        elif kind == 'FLOAT':
            fields = [0xc260]
        else:
            fields = [value & ((1 << width) - 1)
                      for value, width in zip((0x9157bcd3, 0x417b9e2d, 0x80008080, 0xf658cffe), widths)]
        element = pack_element(fields, widths)
        words = cpu_decode(element, widths, kind)
        for sources in ((0, 1, 2, 3), (len(widths) - 1, 4, 0, 5)):
            yield (label, fmt, mapping(*sources), 0, data(2 * len(element), element),
                   tuple(words[source] for source in sources))


def spec(case):
    _, fmt, mapping_word, mode, payload, _ = case
    return '%d:%x:%d:%s' % (fmt, mapping_word, mode, payload.hex())


def check(work, consumer, backend, fxc, extended=False):
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(EXTENDED_PIXEL if extended else PIXEL)
    cases = tuple(expanded_cases()) if extended else CASES
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc,
               X2W_EXTENDED_BUFFER_FORMATS='1' if extended else '0',
               X2W_FORMATTED_BUFFER_DIAGNOSTICS='0', X2W_TEST_FORMATTED_VIEW_REPEAT='0')
    args = [consumer, vs, ps, root, backend, output, str(3 if extended else TARGET)]

    def draw(case, source_env=None):
        result = subprocess.run(args, env=dict(source_env or env, X2W_TEST_FORMATTED_VIEW=spec(case)),
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (result.returncode, result.stderr[-1500:])
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr[-3000:]
        return struct.unpack('<4I', output.read_bytes()[:16]), result.stderr

    # Publish the base request, then the typed render-target variant.
    for _ in range(2):
        draw(cases[0])
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    manifests = [p for p in cache.glob('*.txt') if 'descword3' in p.read_text()]
    assert manifests and all('srv 0 buf table 0 0' in p.read_text() for p in manifests), [p.read_text() for p in manifests]
    for case in cases:
        label, expected = case[0], case[5]
        words, log = draw(case)
        refusals = re.findall(r'srv 0: formatted buffer refused \((.+) 0\)', log)
        if isinstance(expected, str):
            assert '-> drawn;' not in log and refusals == [expected], (label, refusals)
            assert words == CLEAR, (label, [hex(w) for w in words])
        else:
            assert '-> drawn;' in log and not refusals, (label, refusals, log[-2000:])
            assert words == expected, (label, [hex(w) for w in words], [hex(w) for w in expected])
        print('%s: %s' % (label, 'refused and logged' if isinstance(expected, str) else 'bit-exact'))

    variant = next(p for p in manifests if 'rtv1' in p.read_text())
    hlsl = variant.with_suffix('.ps.hlsl')
    source = hlsl.read_text()
    if extended:
        oversized = ('oversized R32_UINT', 42, IDENTITY, 2, data(8, struct.pack('<I', 11)), None)
        words, log = draw(oversized)
        assert words == (0, 0, 1, 1) and '-> drawn;' not in log, (words, log)
        assert 'formatted buffer refused (buffer larger than the 16 MiB draw copy)' in log, log
        print('oversized R32_UINT: refused above the 16 MiB draw copy limit')

        # Send the signed byte to the adjacent unsigned decoder. Only the signed
        # channel should change, and the forced integer one must survive.
        case = next(case for case in cases if case[0] == 'R8_SINT' and case[2] != IDENTITY)
        switch = re.search(r'switch \(\((ud\d+) >> 12\) & 127u\)', source)
        assert switch, 'no FORMAT switch in the translation'
        selector = '((%s >> 12) & 127u)' % switch.group(1)
        mutated = source.replace(switch.group(0), 'switch (%s == 6u ? 5u : %s)' % (selector, selector), 1)
        wrong = (0xd3, 0, 0xd3, 1)
    else:
        diagnostic_env = dict(env, X2W_FORMATTED_BUFFER_DIAGNOSTICS='1', X2W_TEST_FORMATTED_VIEW_REPEAT='1')
        refused_case = next(case for case in cases if case[1] == 42)
        words, log = draw(refused_case, diagnostic_env)
        line = 'srv 0: formatted buffer refused (view format has no decoded buffer FORMAT DXGI format 42 mapping 0x1688)'
        assert words == CLEAR and '-> drawn;' not in log and log.count(line) == 1, log
        print('diagnostics: one DXGI format and mapping line for two refused draws')

        # Existing mapping control: read W from the Z selector.
        case = cases[0]
        selector = re.search(r'\(\((ud\d+) >> 9\) & 7u\)', source)
        assert selector, 'no W selector in the translation'
        mutated = source.replace(selector.group(0), '((%s >> 6) & 7u)' % selector.group(1))
        wrong = (*case[5][:3], case[5][2])

    hlsl.write_text(mutated)
    subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', variant.with_suffix('.ps.cso'), hlsl],
                   check=True, capture_output=True)
    words, log = draw(case)
    assert '-> drawn;' in log and words != case[5] and words == wrong, (words, case[5], wrong)
    print('control: %s fails the CPU comparison' % ('unsigned FORMAT for a signed byte' if extended else 'W from Z'))
    print('%s: %d typed-view cases through the translated buffer_load_format_xyzw' % (backend, len(cases)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4 and sys.argv[4:] != ['--extended']:
        raise SystemExit(__doc__)
    if not Path(sys.argv[3]).is_file():
        print('fxc unavailable')
        sys.exit(77)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:4], extended=len(sys.argv) == 5))
