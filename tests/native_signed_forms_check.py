"""v_ashrrev_i64 and v_mad_i32_i24 in the direct compute path (#2991).

  native_signed_forms_check.py controls <compile_dxil> <sdk bin dir>
  native_signed_forms_check.py hardware <native_store_replay> <sdk bin dir>

v_ashrrev_i64 shifts the 64-bit value S1 right arithmetically by S0[5:0]. Its shifted-value rules are those of the
logical 64-bit shifts (tests/native_shift64_check.py), with one difference that gfx2hlsl.py's
i64_words(signed=True) makes: a 32-bit literal is sign-extended, not zero-extended. v_mad_i32_i24 is
sext(S0[23:0]) * sext(S1[23:0]) + S2 modulo 2^32. All words were assembled with LLVM 15.0.7 (clang -cc1as, gfx1013).

The shift dispatches reuse native_shift64_check.compute_program with the VOP3 opcode of v_ashrrev_i64 (0x301), a
negative SGPR pair (0x89abcdef:01234567), inline -1, -16, 64 and 1 and the literal 0x80000001, for shift amounts
0, 1, 31, 32, 33 and 63. Zero-extended-literal and logical-shift references must differ from the expected words.
The multiply-add dispatch uses, with n = lane + 32 * group, S0 = mul_u32_u24(n, 0x05f3d1) ^ 0xa5800000,
S1 = mul_u32_u24(n, 0x0371) ^ 0x7f9abcde and S2 = mul_u32_u24(n, 0x1001), so bit 23 is set in many factors and
bits above it are not zero. Unsigned and unmasked references must differ.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import native_shift64_check as S64
from native_dpp_quad_check import request

S64.SHIFTS = (('arithmetic', 0xD7010002),)  # v_ashrrev_i64 v[2:3], amount, source
S64.SGPR_LOW, S64.SGPR_HIGH = 0x01234567, 0x89ABCDEF
SOURCES = (
    ('inline -1', 193, -1),
    ('inline -16', 208, -16),
    ('inline 64', 192, 64),
    ('inline 1', 129, 1),
    ('literal', 255, 0x80000001),
    ('SGPR pair', 22, (0x89ABCDEF << 32) | 0x01234567),
)
AMOUNTS = (0, 1, 31, 32, 33, 63)
MASK64 = (1 << 64) - 1
# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 9 /
# s_lshl_b32 s8, s5, 5 / v_lshlrev_b32 v8, 4, v0 / v_add_nc_u32 v8, s7, v8 / v_add_nc_u32 v9, s8, v0 /
# v_mul_u32_u24 v1, 0x5f3d1, v9 / v_xor_b32 v1, 0xa5800000, v1 / v_mul_u32_u24 v2, 0x371, v9 /
# v_xor_b32 v2, 0x7f9abcde, v2 / v_mul_u32_u24 v3, 0x1001, v9 / v_mad_i32_i24 v4, v1, v2, v3 /
# s_waitcnt lgkmcnt(0) / buffer_store_dword v4, v8, s[32:35], 0 offen / s_endpgm
MAD = ('040394be 800395be 0a0808f4200000fa 0589078f 0585088f 84001034 0710104a 0800124a ff120216d1f30500 '
       'ff02023a000080a5 ff12041671030000 ff04043adebc9a7f ff12061601100000 040042d501050e04 7fc08cbf '
       '001070e008040880 000081bf')
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def words(text):
    out = []
    for h in text.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


def shifted(source, amount, zero_extend_literal=False, logical=False):
    name, _, value = source
    if name == 'literal' and not zero_extend_literal:
        value = value - (1 << 32) if value & 0x80000000 else value
    bits = value & MASK64
    signed = bits - (1 << 64) if bits >> 63 else bits
    return (bits >> amount if logical else signed >> amount) & MASK64


def mul24(x, y):
    return ((x & 0xFFFFFF) * (y & 0xFFFFFF)) & 0xFFFFFFFF


def sext24(x):
    x &= 0xFFFFFF
    return x - (1 << 24) if x & 0x800000 else x


def mad_reference(unsigned=False, unmasked=False):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for lane in range(32):
            n = lane + 32 * group
            a, b, c = mul24(n, 0x05F3D1) ^ 0xA5800000, mul24(n, 0x0371) ^ 0x7F9ABCDE, mul24(n, 0x1001)
            if unmasked:
                x, y = a, b
            elif unsigned:
                x, y = a & 0xFFFFFF, b & 0xFFFFFF
            else:
                x, y = sext24(a), sext24(b)
            expected[GUARD_WORDS + (group * 32 + lane) * 4] = (x * y + c) & 0xFFFFFFFF
    return expected


def compile_ok(cli, sdk, work, root, program):
    marker = request(work, root, program)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    return subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True,
                          check=True).stdout


def controls(cli, sdk, work, root):
    # The modules, before the GPU run: an arithmetic 64-bit shift (no logical one), and for the multiply-add a
    # 24-bit sign extension of each factor, a shift left by 8 followed by an arithmetic shift right by 8.
    for source in SOURCES:
        dump = compile_ok(cli, sdk, work, root, S64.compute_program(1, (source,)))
        assert 'ashr i64' in dump and 'lshr i64 %' in dump and dump.count('ashr i64') == 1, (source[0], dump)
    dump = compile_ok(cli, sdk, work, root, words(MAD))
    assert dump.count(', 8') >= 4 and 'ashr i32' in dump and 'mul i32' in dump, dump
    changed = {label: any(shifted(s, k, **kw) != shifted(s, k) for s in SOURCES for k in AMOUNTS)
               for label, kw in (('zero-extended literal', {'zero_extend_literal': True}),
                                 ('logical shift', {'logical': True}))}
    assert all(changed.values()), changed
    for label, mutant in (('unsigned factors', mad_reference(unsigned=True)),
                          ('factors not masked to 24 bits', mad_reference(unmasked=True))):
        assert mutant != mad_reference(), label
    print('compiled six v_ashrrev_i64 sources and v_mad_i32_i24; the zero-extended-literal, logical, unsigned '
          'and unmasked references each differ')
    return 0


def readback(replay, sdk, work, root, program):
    marker = request(work, root, program)
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return None
    result.check_returncode()
    return list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))


def hardware(replay, sdk, work, root):
    for amount in AMOUNTS:
        for source in SOURCES:
            actual = readback(replay, sdk, work, root, S64.compute_program(amount, (source,)))
            if actual is None:
                return 77
            expected = shifted(source, amount)
            words_ = (expected & 0xFFFFFFFF, expected >> 32)
            start = GUARD_WORDS
            end = start + S64.LANES * 2
            assert actual[:start] == [SENTINEL] * start
            assert actual[start:end] == list(words_) * S64.LANES, (source[0], amount, words_, actual[start:end][:4])
            assert actual[end:] == [SENTINEL] * (OUTPUT_WORDS - end)
        print(f'v_ashrrev_i64 by {amount}: all 6 sources and guards matched')
    actual = readback(replay, sdk, work, root, words(MAD))
    if actual is None:
        return 77
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, mad_reference())) if x != y]
    assert not diff, diff[:10]
    print('v_mad_i32_i24: 64 words of two groups and the guards matched')
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
