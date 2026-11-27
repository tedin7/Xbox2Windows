"""v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32 and v_msad_u8 in the direct compute path (#2987).

  native_sad_check.py controls <compile_dxil> <sdk bin dir>
  native_sad_check.py hardware <native_store_replay> <sdk bin dir>

Each operation sums unsigned absolute differences of S0 and S1 (four bytes, two halfwords or one dword) and adds
S2, modulo 2^32, as gfx2hlsl.py's sad_u8, sad4_u8, sad_u16 and sad_u32 helpers and tools/pixel_aot.py do.
v_msad_u8 skips each byte whose S1 byte is zero, and v_sad_hi_u8 shifts the byte sum left by 16. All words below
were assembled with LLVM 15.0.7 (clang -cc1as, gfx1013).

The dispatch has two groups of 32 lanes. With n = lane + 32 * group, S0 = mul_u32_u24(n, 0x050311) ^ 0x80f00a33,
S1 = mul_u32_u24(n, 0x010203) & 0xffff00ff (byte 1 is always zero, byte 0 is zero for n = 0) and
S2 = mul_u32_u24(n, 0x1001). Bytes of S0 below and above those of S1 both occur. Each lane stores the five
results. References with signed byte differences, msad without its mask, sad_hi without its shift and S2 dropped
must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 10 /
# s_lshl_b32 s8, s5, 5 / v_lshlrev_b32 v8, 5, v0 / v_add_nc_u32 v8, s7, v8 / v_add_nc_u32 v9, s8, v0 /
# v_mul_u32_u24 v1, 0x50311, v9 / v_xor_b32 v1, 0x80f00a33, v1 / v_mul_u32_u24 v2, 0x10203, v9 /
# v_and_b32 v2, 0xffff00ff, v2 / v_mul_u32_u24 v3, 0x1001, v9 / v_sad_u8 v10, v_sad_hi_u8 v11, v_sad_u16 v12,
# v_sad_u32 v13, v_msad_u8 v14 (each v1, v2, v3) / s_waitcnt lgkmcnt(0) / buffer_store_dword v10..v14 at
# v8 + 0..16 through s[32:35] / s_endpgm
HEX = ('040394be 800395be 0a0808f4200000fa 058a078f 0585088f 85001034 0710104a 0800124a ff12021611030500 '
       'ff02023a330af080 ff12041603020100 ff040436ff00ffff ff12061601100000 0a005ad501050e04 0b005bd501050e04 '
       '0c005cd501050e04 0d005dd501050e04 0e0071d501050e04 7fc08cbf 001070e0080a0880 041070e0080b0880 '
       '081070e0080c0880 0c1070e0080d0880 101070e0080e0880 000081bf')
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
MASK = 0xFFFFFFFF


def words(text):
    out = []
    for h in text.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


def mul24(x, y):
    return ((x & 0xFFFFFF) * (y & 0xFFFFFF)) & MASK


def operands(n):
    return mul24(n, 0x050311) ^ 0x80F00A33, mul24(n, 0x010203) & 0xFFFF00FF, mul24(n, 0x1001)


def parts(x, width):
    return [(x >> at) & ((1 << width) - 1) for at in range(0, 32, width)]


def signed8(x):
    return x - 256 if x & 0x80 else x


def results(a, b, c, signed=False, unmasked=False, no_shift=False, no_add=False):
    def diff(x, y):
        return abs(signed8(x) - signed8(y)) if signed else abs(x - y)
    add = 0 if no_add else c
    bytes_sum = sum(diff(x, y) for x, y in zip(parts(a, 8), parts(b, 8)))
    masked = sum(diff(x, y) for x, y in zip(parts(a, 8), parts(b, 8)) if unmasked or y)
    return [(bytes_sum + add) & MASK,
            ((bytes_sum << (0 if no_shift else 16)) + add) & MASK,
            (sum(abs(x - y) for x, y in zip(parts(a, 16), parts(b, 16))) + add) & MASK,
            (abs(a - b) + add) & MASK,
            (masked + add) & MASK]


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for lane in range(32):
            base = GUARD_WORDS + (group * 32 + lane) * 8
            expected[base:base + 5] = results(*operands(lane + 32 * group), **mutation)
    return expected


def controls(cli, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    # The same v_sad_u8 with a clamp bit is refused, as source and output modifiers are on integer VOP3 here.
    clamped = words(HEX)
    clamped[clamped.index(0xD55A000A)] |= 0x8000
    marker = request(work, root, clamped)
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, marker.with_suffix('.cso')],
                            capture_output=True, text=True)
    assert result.returncode and 'direct ALU operand modifiers' in result.stderr, (result.stdout, result.stderr)
    expected = reference()
    a, b, _ = zip(*(operands(n) for n in range(64)))
    assert any(x > y for v, w in zip(a, b) for x, y in zip(parts(v, 8), parts(w, 8)))
    assert any(x < y for v, w in zip(a, b) for x, y in zip(parts(v, 8), parts(w, 8)))
    for label, mutation in (('signed byte differences', {'signed': True}), ('msad without its mask', {'unmasked': True}),
                            ('sad_hi without its shift', {'no_shift': True}), ('S2 dropped', {'no_add': True})):
        assert reference(**mutation) != expected, label
    print('compiled; clamp refused; the signed, unmasked, unshifted and S2-dropped references each differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    expected = reference()
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if x != y]
    assert not diff, diff[:10]
    print('five SAD operations: 320 words of two groups and the guards matched')
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
