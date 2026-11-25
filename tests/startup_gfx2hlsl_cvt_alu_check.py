"""Conversion operations of tools/startup/gfx2hlsl.py (its integer-input tables).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_cvt_f32_u32, v_cvt_f32_i32, v_cvt_u32_f32, v_cvt_f32_ubyte0..3, v_cvt_f32_f16 and v_cvt_f16_f32, which
no check ran before. The fixture and runner are startup_gfx2hlsl_lane_alu.py: one operation per lane, alternating the
e32 and e64 spellings, on flat parameters the compilers cannot fold. The references follow the AMD RDNA 2 ISA
reference: integer to binary32 rounds to nearest even; v_cvt_u32_f32 truncates toward zero and reads 0 for a negative
value or a NaN; the ubyte forms convert byte N of S0; v_cvt_f32_f16 reads S0[15:0] and v_cvt_f16_f32 writes the
binary16 value to bits 15:0 with bits 31:16 clear.

Not covered: v_cvt_u32_f32 at or above 2^32 and of +inf (the HLSL uint() conversion of an out-of-range value is not
pinned by the D3D specification), binary16 denormal inputs (the denormal mode decides them), and v_cvt_f16_f32 of a
binary32 value that is not exactly representable in binary16.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import check_draw, main, signed, words  # noqa: E402


def f32(value):
    """The binary32 word of a Python number, rounded to nearest even."""
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def cvt_u32_f32(word):
    v = value(word)
    return 0 if v != v or v <= 0 else int(v)


def half(word):
    return struct.unpack('<e', struct.pack('<H', word & 0xffff))[0]


def to_half(word):
    return struct.unpack('<H', struct.pack('<e', value(word)))[0]


OPS = {
    'v_cvt_f32_u32': (1, lambda a: f32(float(a)), True),
    'v_cvt_f32_i32': (1, lambda a: f32(float(signed(a))), True),
    'v_cvt_u32_f32': (1, cvt_u32_f32, True),
    'v_cvt_f32_ubyte0': (1, lambda a: f32(float(a & 255)), True),
    'v_cvt_f32_ubyte1': (1, lambda a: f32(float(a >> 8 & 255)), True),
    'v_cvt_f32_ubyte2': (1, lambda a: f32(float(a >> 16 & 255)), True),
    'v_cvt_f32_ubyte3': (1, lambda a: f32(float(a >> 24)), True),
    'v_cvt_f32_f16': (1, lambda a: f32(half(a)), True),
    'v_cvt_f16_f32': (1, to_half, True),
}
# Input sets (src0, unused, unused), eight per operation. Integers with the sign bit set and values that round
# (0x80000001 and 0xffffffff to 2^31 and 2^32, 0x01000001 to 2^24 by ties to even); binary32 values with fractions,
# negatives, -0, a NaN and the largest value below 2^32; byte patterns that differ in every byte; binary16 words
# whose high halves hold other values.
INTS = [(0x80000001, 0, 0), (0xffffffff, 0, 0), (0x01000001, 0, 0), (7, 0, 0),
        (0, 0, 0), (0x7fffffff, 0, 0), (0x01000003, 0, 0), (0xfffffff9, 0, 0)]
FLOATS = [(f32(1.75), 0, 0), (f32(-3.5), 0, 0), (0x7fc00000, 0, 0), (0x4f7fffff, 0, 0),
          (0x80000000, 0, 0), (f32(123456.9), 0, 0), (f32(0.25), 0, 0), (f32(2147483520.0), 0, 0)]
BYTES = [(0x80c0e0ff, 0, 0), (0x01020304, 0, 0), (0xfe7f8001, 0, 0), (0x12345678, 0, 0),
         (0, 0, 0), (0xffffffff, 0, 0), (0x00ff00ff, 0, 0), (0xa5c30f96, 0, 0)]
HALVES = [(0x12343c00, 0, 0), (0xc500c100, 0, 0), (0x00007bff, 0, 0), (0x40008000, 0, 0),
          (0xabcd7c00, 0, 0), (0x0000fc00, 0, 0), (0x3c000400, 0, 0), (0x5555b800, 0, 0)]
SINGLES = [(f32(1.0), 0, 0), (f32(-2.5), 0, 0), (f32(65504.0), 0, 0), (f32(2.0 ** -14), 0, 0),
           (0x80000000, 0, 0), (f32(0.5), 0, 0), (f32(-1024.25), 0, 0), (0, 0, 0)]
DRAW = {'v_cvt_f32_u32': INTS, 'v_cvt_f32_i32': INTS, 'v_cvt_u32_f32': FLOATS, 'v_cvt_f32_f16': HALVES,
        'v_cvt_f16_f32': SINGLES}
DRAW.update({'v_cvt_f32_ubyte%d' % n: BYTES for n in range(4)})
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_cvt_f32_u32': (r'U\(float\((v\d+)\)\)', r'U(float(int(\1)))', lambda a: f32(float(signed(a)))),
    'v_cvt_f32_i32': (r'U\(float\(int\((v\d+)\)\)\)', r'U(float(\1))', lambda a: f32(float(a))),
    'v_cvt_u32_f32': (r'uint\(max\(F\((v\d+)\), 0\.0\)\)', r'uint(round(max(F(\1), 0.0)))',
                      lambda a: 0 if value(a) != value(a) or value(a) <= 0 else round(value(a))),
    'v_cvt_f32_ubyte0': (r'& 255\)\)', r'& 127))', lambda a: f32(float(a & 127))),
    'v_cvt_f32_ubyte1': (r'>> 8\)', r'>> 16)', lambda a: f32(float(a >> 16 & 255))),
    'v_cvt_f32_ubyte2': (r'(v\d+) >> 16\)', r'\1 >> 8)', lambda a: f32(float(a >> 8 & 255))),
    'v_cvt_f32_ubyte3': (r'U\(float\((v\d+) >> 24\)\)', r'U(float((\1 >> 16) & 255))', lambda a: f32(float(a >> 16 & 255))),
    'v_cvt_f32_f16': (r'f16tof32\((v\d+)\)', r'f16tof32(\1 >> 16)', lambda a: f32(half(a >> 16))),
    'v_cvt_f16_f32': (r'f32tof16\(F\((v\d+)\)\)', r'f32tof16(-F(\1))', lambda a: to_half(a ^ 0x80000000)),
}
BODY = {
    'v_cvt_f32_u32': words('000d187e0d0086d501010000020d1c7e0f0086d503010000'),
    'v_cvt_f32_i32': words('000b187e0d0085d501010000020b1c7e0f0085d503010000'),
    'v_cvt_u32_f32': words('000f187e0d0087d501010000020f1c7e0f0087d503010000'),
    'v_cvt_f32_ubyte0': words('0023187e0d0091d50101000002231c7e0f0091d503010000'),
    'v_cvt_f32_ubyte1': words('0025187e0d0092d50101000002251c7e0f0092d503010000'),
    'v_cvt_f32_ubyte2': words('0027187e0d0093d50101000002271c7e0f0093d503010000'),
    'v_cvt_f32_ubyte3': words('0029187e0d0094d50101000002291c7e0f0094d503010000'),
    'v_cvt_f32_f16': words('0017187e0d008bd50101000002171c7e0f008bd503010000'),
    'v_cvt_f16_f32': words('0015187e0d008ad50101000002151c7e0f008ad503010000'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_cvt_f32_u32 v1, v2', 'v_cvt_f32_i32_e64 v1, s2', 'v_cvt_u32_f32 v1, 0.5',
                 'v_cvt_f32_ubyte2_e64 v1, v2', 'v_cvt_f16_f32 v1, v2',
                 'v_cvt_f32_f16_e64 v1, v2 mul:2'],
    'refused': ['v_cvt_f32_u32 v1, v2, v3', 'v_cvt_f32_i32 v1', 'v_cvt_u32_f32_e64 v1, v2 clamp',
                'v_cvt_f32_ubyte0_e64 v1, -v2', 'v_cvt_f32_f16_e64 v1, |v2|', 'v_cvt_f16_f32_e64 v1, v2 mul:2'],
}
# v_cvt_f16_f32 writes a binary16 value to bits 15:0, so every result word is a binary32 denormal when the fixture
# exports it to the float target. FXC ps_5_1 output of such a pixel stage reads 0 in all four channels on WARP and on
# an RX 6600, even for a channel whose value is a constant, while the same words OR 0x3f800000 read back exactly. The
# conversion is therefore drawn through DXC ps_6_0 only; the other operations are drawn through both compilers.
DXC_ONLY = ('v_cvt_f16_f32',)


def subset(names):
    return {key: ({op: v for op, v in value.items() if op in names} if isinstance(value, dict) else value)
            for key, value in FAMILY.items()}


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == 'draw':
        _, _, dxc, renderer, adapter, fxc = sys.argv
        check_draw(subset([op for op in OPS if op not in DXC_ONLY]), dxc, renderer, adapter, fxc)
        check_draw(subset(DXC_ONLY), dxc, renderer, adapter)
    else:
        main(FAMILY, __doc__)
