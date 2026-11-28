"""v_mad_u64_u32 and v_mad_i64_i32 in the direct compute path (#3000).

  native_mad64_check.py controls <compile_dxil> <sdk bin dir>
  native_mad64_check.py hardware <native_store_replay> <sdk bin dir>

D = S0 * S1 + S2 modulo 2^64, with 32-bit factors (zero- or sign-extended) and a 64-bit S2 (gfx2hlsl.py mad64). The
unsigned form writes the carry-out of the 64-bit add to its SDST as a lane mask; the signed form is lowered only with
the null SDST, as in gfx2hlsl.py. All words were assembled with LLVM 15.0.7 (clang -cc1as, gfx1013).

The dispatch has two groups of 32 lanes. With n = lane + 32 * group, S0 = mul_u32_u24(n, 0x0f0f0f) ^ 0xffff0000,
S1 = mul_u32_u24(n, 0x123457) ^ 0xfff00000 and S2 = {((n & 1) << 31) | 0x10, mul_u32_u24(n, 0x777777) ^ 0x89abcdef},
so the products lie just below 2^64 and the unsigned sum carries exactly for the odd lanes. Each lane stores the unsigned and signed results and
the carry mask in s30 (v_mov_b32 v12, s30), which must equal the ballot of its group. References that drop the
high word, zero-extend the signed factors or report no carry must each differ.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

# See the docstring; the program ends with buffer_store_dword v4, v5, v10, v11, v12 at v8 + 0..16 and s_endpgm.
HEX = ('040394be 800395be 0a0808f4200000fa 058a078f 0585088f 85001034 0710104a 0800124a ff1202160f0f0f00 '
       'ff02023a0000ffff ff12041657341200 ff04043a0000f0ff ff120c1677777700 ff0c0c3aefcdab89 81120e36 9f0e0e34 900e0e38 '
       '041e76d501051a04 0a7d77d501051a04 1e02187e 7fc08cbf 001070e008040880 041070e008050880 081070e0080a0880 '
       '0c1070e0080b0880 101070e0080c0880 000081bf')
SIGNED_WITH_SDST = '0a1e77d501051a04'  # v_mad_i64_i32 v[10:11], s30, v1, v2, v[6:7]: refused
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
M32, M64 = 0xFFFFFFFF, (1 << 64) - 1


def words(text):
    out = []
    for h in text.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


def mul24(x, y):
    return ((x & 0xFFFFFF) * (y & 0xFFFFFF)) & M32


def s32(x):
    return x - (1 << 32) if x & 0x80000000 else x


def operands(n):
    a, b = mul24(n, 0x0F0F0F) ^ 0xFFFF0000, mul24(n, 0x123457) ^ 0xFFF00000
    c = ((((n & 1) << 31) | 0x10) << 32) | (mul24(n, 0x777777) ^ 0x89ABCDEF)
    return a, b, c


def reference(drop_high=False, zero_extend_signed=False, no_carry=False):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        mask = 0
        rows = []
        for lane in range(32):
            a, b, c = operands(lane + 32 * group)
            total = a * b + c
            mask |= (0 if no_carry else int(total > M64)) << lane
            signed = ((a * b if zero_extend_signed else s32(a) * s32(b)) + c) & M64
            unsigned = total & M64
            rows.append([unsigned & M32, 0 if drop_high else unsigned >> 32, signed & M32, signed >> 32])
        for lane, row in enumerate(rows):
            base = GUARD_WORDS + (group * 32 + lane) * 8
            expected[base:base + 5] = row + [mask]
    return expected


def compile_program(cli, sdk, work, root, code):
    marker = request(work, root, code)
    return marker, subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker,
                                   marker.with_suffix('.cso')], capture_output=True, text=True)


def controls(cli, sdk, work, root):
    marker, result = compile_program(cli, sdk, work, root, words(HEX))
    assert result.returncode == 0 and marker.with_suffix('.cso').is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', marker.with_suffix('.cso')], capture_output=True,
                          text=True, check=True).stdout
    assert dump.count('mul i64') == 2 and 'sext i32' in dump and 'icmp ult i64' in dump, dump
    signed_sdst = HEX.replace('0a7d77d501051a04', SIGNED_WITH_SDST)
    _, result = compile_program(cli, sdk, work, root, words(signed_sdst))
    assert result.returncode and 'v_mad_i64_i32 carry-out destination' in result.stderr, (result.stdout, result.stderr)
    # The unsigned form with the null SDST is refused by the symbolic transfer, as gfx2hlsl.py refuses it, and an
    # inline float factor is refused like in gfx2hlsl.py.
    for code, message in ((HEX.replace('041e76d501051a04', '047d76d501051a04'), 'unsupported carry mask destination'),
                          (HEX.replace('041e76d501051a04', '041e76d5f2041a04'), 'with an inline float factor')):
        _, result = compile_program(cli, sdk, work, root, words(code))
        assert result.returncode and message in result.stderr, (message, result.stdout, result.stderr)
    expected = reference()
    masks = {expected[GUARD_WORDS + g * 32 * 8 + 4] for g in range(2)}
    assert all(0 < m < M32 for m in masks), [hex(m) for m in masks]  # some lanes carry, some do not
    for label, mutation in (('high word dropped', {'drop_high': True}),
                            ('signed factors zero-extended', {'zero_extend_signed': True}),
                            ('no carry', {'no_carry': True})):
        assert reference(**mutation) != expected, label
    print('compiled with two 64-bit multiplies and an unsigned carry compare; the signed form with an SGPR '
          'carry-out, the unsigned form with null and an inline float factor refused; both groups have mixed carries; the three mutated references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, reference())) if x != y]
    assert not diff, diff[:10]
    print('v_mad_u64_u32 and v_mad_i64_i32: results, carry masks, guards and sentinels matched')
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
