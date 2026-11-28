"""Direct DXIL fract's strict upper bound, NaNs and output modifier order.

The source-built VOP1 and VOP3 instructions write 32 words per lane in two
wave32 groups. Tiny negative values make a host Frc round to 1.0; gfx1013
instead returns the greatest binary32 below 1.0. A NaN must stay a NaN.
"""
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_dpp_quad_check import request
from native_frexp_f32_check import v_mov_lit, vop1, vop3_unary
from native_mad32_check import PROLOGUE

INPUTS = (0xb0800000, 0xb1800000, 0x3e800000, 0xbe800000,
          0, 0x80000000, 0x7fc12345, 0xc0400000)
CAP = 0x3f7fffff
SENTINEL = 0x69696969
WORDS = 2064


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def fract(word, *, cap=True, nan_to_cap=False):
    x = value(word)
    if math.isnan(x):
        return CAP if nan_to_cap else 0x7fc00000
    result = bits(x - math.floor(x))
    if not result & 0x7fffffff:
        result = 0  # floor retains -0, so subtracting that same zero yields +0.
    return CAP if cap and value(result) >= 1.0 else result


def expected_row(word, **mutation):
    plain = fract(word, **mutation)
    negated = fract(word ^ 0x80000000, **mutation)
    clamped = bits(0.0 if math.isnan(value(plain)) else value(plain))
    scaled = bits(value(plain) * 2.0)
    return plain, negated, clamped, scaled


def reference(**mutation):
    row = [v for word in INPUTS for v in expected_row(word, **mutation)]
    return [SENTINEL] * 8 + row * 64 + [SENTINEL] * 8


def program():
    code = list(PROLOGUE)
    for ci, word in enumerate(INPUTS):
        code += v_mov_lit(1, word)
        forms = ([vop1(0x20, 3, 257)],
                 vop3_unary(0x1a0, 3, 257, neg0=1),
                 vop3_unary(0x1a0, 3, 257, clamp=1),
                 vop3_unary(0x1a0, 3, 257, omod=1))
        for fi, form in enumerate(forms):
            code += form + [0xe0701000 | (4 * (4 * ci + fi)), 0x80080308]
    return code + [0xbf810000]


def is_nan(word):
    return (word & 0x7fffffff) > 0x7f800000


def matches(actual, expected):
    return len(actual) == len(expected) and all(
        a == e or (is_nan(a) and is_nan(e)) for a, e in zip(actual, expected))


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'fract.cso'
    subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output], check=True)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output],
                          capture_output=True, text=True, check=True).stdout
    assert 'Frc(value)' in dump, dump
    # Ordered comparison preserves NaN. The cap precedes the output scale
    # and clamp, which the hardware mode checks through the stored words.
    assert 'fcmp oge float' in dump and '1065353215 to float' in dump, 'missing ordered upper bound'
    expected = reference()
    assert not matches(reference(cap=False), expected), 'uncapped Frc escaped'
    assert not matches(reference(nan_to_cap=True), expected), 'NaN-losing minimum escaped'
    print('fract DXIL validates; uncapped and NaN-losing references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'readback.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = struct.unpack(f'<{WORDS}I', output.read_bytes())
    expected = reference()
    if not matches(actual, expected):
        bad = [(i, f'{a:08x}', f'{e:08x}') for i, (a, e) in enumerate(zip(actual, expected))
               if a != e and not (is_nan(a) and is_nan(e))]
        raise AssertionError(f'{len(bad)}/{WORDS} words differ: {bad[:12]}')
    print('fract e32/e64: 2048 words and 16 guards matched (NaN by class)')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_fract_f32_') as temp:
        work = Path(temp)
        source = work / 'root.hlsl'
        source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG',
                        '-Fo', root, source], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
