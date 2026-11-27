"""v_mad_f32, v_mad_legacy_f32, v_mac_f32 and v_mac_legacy_f32 in the direct compute path (#3005).

  native_mad32_check.py controls <compile_dxil> <sdk bin dir>
  native_mad32_check.py hardware <native_store_replay> <sdk bin dir>

The semantics are gfx2hlsl.py's mad32 (and pixel_aot's): flush denormal inputs to signed zero, round the product on
its own (the legacy forms make 0 * anything +0), add the flushed addend and flush the sum. v_mac_* add the
destination. The VOP3 and VOP2 words were assembled with LLVM 15.0.7 (clang -cc1as, gfx1013); the v_mov and store
words follow native_lane_access_check.py, with 32 words per lane.

Each case sets S0, S1 and S2 by v_mov literals and stores the four results; every lane of two groups stores all
cases. The cases need a separately rounded product ((1 + 2^-12)^2 - 1), flush a denormal input and a denormal sum,
take 0 * inf and signed zeros. References that fuse, skip the flush, or treat the legacy forms like the plain ones
must each differ from the expected words.
"""
from pathlib import Path
import math
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]


CASES = [
    (0x3F800800, 0x3F800800, bits(-1.0)),            # product needs 25 bits: fused and unfused differ
    (0x00000001, 0x7E800000, bits(1.0)),             # a denormal input: flushed, so the product is 0
    (bits(0.0), 0x7F800000, bits(2.0)),              # 0 * inf: NaN, or +0 for the legacy forms
    (0x80000000, bits(1.0), 0x80000000),             # -0 * 1 + -0: -0, or +0 + -0 = +0 for the legacy forms
    (bits(1.5 * 2.0 ** -63), bits(2.0 ** -63), 0x80800000),  # 1.5 * 2^-126 - 2^-126: a denormal sum, flushed
    (bits(3.0), bits(-2.5), bits(0.25)),             # an ordinary value
]
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
BODY = [0xD5410004, 0x040E0501, 0xD5400005, 0x040E0501, 0x7E0C0303, 0x3E0C0501, 0x7E0E0303, 0x0C0E0501]
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def program():
    code = list(PROLOGUE)
    for index, (a, b, c) in enumerate(CASES):
        code += [0x7E0202FF, a, 0x7E0402FF, b, 0x7E0602FF, c] + BODY
        for slot, vgpr in enumerate((4, 5, 6, 7)):
            code += [0xE0701000 | (16 * index + 4 * slot), 0x80080008 | vgpr << 8]
    return code + [0xBF810000]


def f32(x):
    if math.isnan(x):
        return 0x7FC00000
    return bits(x)


def ftz(b):
    return b & 0x80000000 if (b & 0x7FFFFFFF) < 0x00800000 else b


def mad(a, b, c, legacy, fused=False, flush=True):
    if flush:
        a, b, c = ftz(a), ftz(b), ftz(c)
    x, y, z = value(a), value(b), value(c)
    if legacy and (x == 0 or y == 0):
        product = 0.0
    else:
        product = x * y if not (math.isinf(x) or math.isinf(y)) or (x != 0 and y != 0) else math.nan
    if fused:
        total = product + z if not math.isnan(product) else math.nan
        out = f32(total) if not math.isnan(total) else 0x7FC00000
    else:
        p = f32(product) if not math.isnan(product) else 0x7FC00000
        if flush:
            p = ftz(p)
        q = value(p)
        total = q + z if not math.isnan(q) else math.nan
        out = f32(total) if not math.isnan(total) else 0x7FC00000
    return ftz(out) if flush else out


def results(a, b, c, same_legacy=False, **mutation):
    plain, legacy = mad(a, b, c, False, **mutation), mad(a, b, c, True, **mutation)
    if same_legacy:
        legacy = plain
    return [plain, legacy, plain, legacy]


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 32
        for index, case in enumerate(CASES):
            expected[base + 4 * index:base + 4 * index + 4] = results(*case, **mutation)
    return expected


def nan_equal(actual, expected):
    """A NaN result compares by class: any quiet or signalling NaN pattern matches an expected NaN."""
    def is_nan(x):
        return (x & 0x7F800000) == 0x7F800000 and x & 0x007FFFFF
    return all(a == e or (is_nan(a) and is_nan(e)) for a, e in zip(actual, expected))


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    # Four unfused multiply-adds per case: a separate fmul and fadd each, and no FMad (DXIL tertiary opcode 46).
    assert dump.count('fmul') >= 4 * len(CASES) and dump.count('fadd') >= 4 * len(CASES), dump
    assert '@dx.op.tertiary.f32(i32 46,' not in dump, dump
    assert ' fast ' not in dump and 'contract' not in dump, dump
    # v_mac_f32_e64 v6, v1, v2 with the neg bit of the accumulator (src2) set: refused, not dropped.
    negated = program()
    at = negated.index(0x3E0C0501)
    negated[at:at + 1] = [0xD51F0006, 0x841A0501]
    marker = request(work, root, negated)
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, marker.with_suffix('.neg.cso')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'accumulator modifier' in result.stdout + result.stderr, (result.stdout,
                                                                                                result.stderr)
    expected = reference()
    for label, mutation in (('fused', {'fused': True}), ('no flush', {'flush': False}),
                            ('legacy like plain', {'same_legacy': True})):
        assert reference(**mutation) != expected, label
    print('compiled with separate fmul and fadd and no FMad or fast-math flags; the fused, unflushed and '
          'legacy-as-plain references differ; an accumulator neg is refused')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    expected = reference()
    assert nan_equal(actual, expected), [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected))
                                         if x != y][:10]
    print('the four unfused multiply-adds: all 2064 words matched (NaN by class)')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
