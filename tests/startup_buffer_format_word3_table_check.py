"""Check both directions of the mapping between DXGI views and formatted-load decoders."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402

HEADER = ROOT / 'src' / 'startup_stubs' / 'buffer_format_word3.h'
ALIASES = {'B8G8R8A8_UNORM': 'R8G8B8A8_UNORM'}
# Public DXGI values used by the vertex-fetch table.
FETCH_NAMES = {2: 'R32G32B32A32_FLOAT', 6: 'R32G32B32_FLOAT', 16: 'R32G32_FLOAT', 41: 'R32_FLOAT',
               28: 'R8G8B8A8_UNORM', 49: 'R8G8_UNORM', 61: 'R8_UNORM', 34: 'R16G16_FLOAT',
               35: 'R16G16_UNORM', 37: 'R16G16_SNORM', 30: 'R8G8B8A8_UINT', 24: 'R10G10B10A2_UNORM'}


def dxgi_name(decoder):
    fields = decoder.removeprefix('BUF_FMT_').split('_')
    widths, numeric = fields[:-1], fields[-1]
    if widths == ['2', '10', '10', '10']:
        widths = ['10', '10', '10', '2']
    return ''.join(channel + width for channel, width in zip('RGBA', widths)) + '_' + numeric


def check(header, codes):
    code_fn = header[header.index('buffer_format_code('):header.index('view_word3(')]
    entries = re.findall(r'case DXGI_FORMAT_(\w+): return ([^;]+);', code_fn)
    mapped = {name: int(expr.removeprefix('bgra_enabled ? ').removesuffix(' : 0')) for name, expr in entries}
    assert len(mapped) == len(entries), 'duplicate DXGI format'
    assert len(set(codes.values())) == len(codes), 'duplicate descriptor FORMAT code'
    assert set(codes) == set(gfx2hlsl.FETCH_FORMATS), 'decoder bodies and FORMAT names differ'

    supported = (HEADER.parent / 'formatted_buffer.h').read_text()
    decoded = set(map(int, re.findall(r'case (\d+):', supported[supported.index('formatted_buffer_supported('):])))
    assert decoded == set(codes.values()), sorted(decoded ^ set(codes.values()))
    expected = {dxgi_name(name): code for name, code in codes.items()}
    assert len(expected) == len(codes), 'duplicate canonical DXGI format'
    expected.update({alias: expected[canonical] for alias, canonical in ALIASES.items()})
    assert mapped == expected, {name: (mapped.get(name), expected.get(name))
                                for name in sorted(mapped.keys() | expected.keys()) if mapped.get(name) != expected.get(name)}
    for dxgi, (decoder, _) in gfx2hlsl.FETCH_DXGI.items():
        assert mapped[FETCH_NAMES[dxgi]] == codes[decoder], (dxgi, decoder)
    return len(mapped)


def must_fail(header, codes, label):
    try:
        check(header, codes)
    except AssertionError:
        return
    raise AssertionError(label + ' passed')


if __name__ == '__main__':
    original = HEADER.read_text()
    codes = gfx2hlsl.DESC_FORMAT_CODES
    count = check(original, codes)
    unorm = 'case DXGI_FORMAT_R16G16_UNORM: return 23;'
    snorm = 'case DXGI_FORMAT_R16G16_SNORM: return 24;'
    swapped = original.replace(unorm, unorm.replace('23', '24')).replace(snorm, snorm.replace('24', '23'))
    must_fail(swapped, codes, 'swapped runtime UNORM/SNORM codes')
    must_fail(original.replace('DXGI_FORMAT_R32G32B32A32_FLOAT: return 77;',
                               'DXGI_FORMAT_R32G32B32A32_FLOAT: return 74;'), codes, 'wrong float4 code')
    must_fail(original.replace(snorm, ''), codes, 'missing runtime entry')
    must_fail(original, {n: c for n, c in codes.items() if c != 24}, 'missing translator entry')
    print('buffer format word 3 table: %d DXGI views, %d decoders, 4 controls rejected' % (count, len(codes)))
