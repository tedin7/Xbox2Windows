"""The stand-in's DXGI -> FORMAT table (src/startup_stubs/buffer_format_word3.h) against tools/startup/gfx2hlsl.py.

A typed buffer view the title created has no hardware descriptor, so the draw builds word 3 for the translated
buffer_load_format_* switch from the view's DXGI format. Each mapped format must name a FORMAT code the switch
decodes (DESC_FORMAT_CODES), agree with the vertex-fetch table FETCH_DXGI where that has the format, and the
decoded-code list must equal the switch's cases, so a code is never accepted that would read zeros.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402
from disasm_rdna2 import MTBUF_FORMATS  # noqa: E402

# Public DXGI_FORMAT values (dxgiformat.h) for the names the header uses.
DXGI = {'R32G32B32A32_FLOAT': 2, 'R32G32B32_FLOAT': 6, 'R16G16B16A16_FLOAT': 10, 'R32G32_FLOAT': 16,
        'R8G8B8A8_UNORM': 28, 'R10G10B10A2_UNORM': 24, 'R16G16_FLOAT': 34, 'R16G16_UNORM': 35, 'R16G16_UINT': 36, 'R16G16_SNORM': 37, 'R8G8B8A8_UINT': 30, 'R32_FLOAT': 41,
        'R8G8_UNORM': 49, 'R8_UNORM': 61, 'R8_UINT': 62}
# The buffer format each DXGI format reads as: FETCH_DXGI, plus the three formats it does not list.
EXPECTED = {**{dxgi: name for dxgi, (name, _) in gfx2hlsl.FETCH_DXGI.items()},
            10: 'BUF_FMT_16_16_16_16_FLOAT', 62: 'BUF_FMT_8_UINT', 36: 'BUF_FMT_16_16_UINT'}


# DXGI enumeration and gfx10 unified FORMAT, checked independently of both implementations.
ADDED = {
    'R8_SNORM': (63, 2),
    'R8_SINT': (64, 6),
    'R16_UNORM': (56, 7),
    'R16_SNORM': (58, 8),
    'R16_UINT': (57, 11),
    'R16_SINT': (59, 12),
    'R16_FLOAT': (54, 13),
    'R8G8_SNORM': (51, 15),
    'R8G8_UINT': (50, 18),
    'R8G8_SINT': (52, 19),
    'R32_UINT': (42, 20),
    'R32_SINT': (43, 21),
    'R16G16_SINT': (38, 28),
    'R11G11B10_FLOAT': (26, 36),
    'R10G10B10A2_UINT': (25, 54),
    'R8G8B8A8_SNORM': (31, 57),
    'R8G8B8A8_SINT': (32, 61),
    'R32G32_UINT': (17, 62),
    'R32G32_SINT': (18, 63),
    'R16G16B16A16_UNORM': (11, 65),
    'R16G16B16A16_SNORM': (13, 66),
    'R16G16B16A16_UINT': (12, 69),
    'R16G16B16A16_SINT': (14, 70),
    'R32G32B32A32_UINT': (3, 75),
    'R32G32B32A32_SINT': (4, 76)
}
DXGI.update({name: value for name, (value, _) in ADDED.items()})


def main(header=ROOT / 'src' / 'startup_stubs' / 'buffer_format_word3.h'):
    text = header.read_text()
    code_fn = text[text.index('buffer_format_code('):text.index('view_word3(')]
    legacy, extended = code_fn.split('if (!extended)')
    pattern = r'case DXGI_FORMAT_(\w+): return (\d+);'
    mapped = {DXGI[n]: int(c) for n, c in re.findall(pattern, legacy)}
    added = {n: (DXGI[n], int(c)) for n, c in re.findall(pattern, extended)}
    assert added == ADDED, (added, ADDED)
    supported = (ROOT / 'src' / 'startup_stubs' / 'formatted_buffer.h').read_text()
    legacy_support, extended_support = supported.split('if (!extended)')
    codes = gfx2hlsl.DESC_FORMAT_CODES
    expected_legacy = {codes[name] for name in gfx2hlsl.DESC_LEGACY_FORMATS}
    ext_codes = {code for _, code in ADDED.values()}
    pattern = r'case (\d+):'
    assert {int(c) for c in re.findall(pattern, legacy_support)} == expected_legacy
    assert {int(c) for c in re.findall(pattern, extended_support)} == ext_codes
    assert set(mapped) == set(EXPECTED), sorted(set(mapped) ^ set(EXPECTED))
    for dxgi, code in mapped.items():
        assert code == codes[EXPECTED[dxgi]], (dxgi, code, EXPECTED[dxgi])
    assert set(EXPECTED.values()) == gfx2hlsl.DESC_LEGACY_FORMATS
    assert set(codes.values()) == expected_legacy | ext_codes
    expected_added = {MTBUF_FORMATS[code]: code for code in ext_codes}
    assert {name: code for name, code in codes.items() if name not in gfx2hlsl.DESC_LEGACY_FORMATS} == expected_added
    for name in codes:
        assert (name in gfx2hlsl.DESC_ONE_INTEGER) == name.endswith(('_UINT', '_SINT'))
    return len(mapped) + len(added)


if __name__ == '__main__':
    # Negative control: a header that maps R32G32B32A32_FLOAT to 74 (the three-channel code) must fail.
    original = (ROOT / 'src' / 'startup_stubs' / 'buffer_format_word3.h').read_text()
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        wrong = Path(temp) / 'buffer_format_word3.h'
        wrong.write_text(original.replace('case DXGI_FORMAT_R32G32B32A32_FLOAT: return 77;',
                                          'case DXGI_FORMAT_R32G32B32A32_FLOAT: return 74;'))
        try:
            main(wrong)
        except AssertionError:
            pass
        else:
            raise SystemExit('a wrong FORMAT code for R32G32B32A32_FLOAT passed')
    print('buffer format word 3 table: %d DXGI formats match the translation' % main())
