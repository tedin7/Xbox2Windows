"""v_perm_b32 and v_lerp_u8 in the direct compute path.

  native_perm_lerp_check.py controls <compile_dxil> <sdk bin dir>
  native_perm_lerp_check.py hardware <native_store_replay> <sdk bin dir>

The semantics are those of the RDNA 2 ISA, as the gfx2hlsl.py lowerings (#2875, #2873) state them. v_perm_b32: each
result byte i takes selector s = S2[8i+7:8i]; s 0..7 picks byte s of the 64-bit value {S0, S1} (S1 is bytes 0..3),
s 8..11 gives 0xff or 0x00 from bit 15 of S1, bit 31 of S1, bit 15 of S0 or bit 31 of S0, s 12 gives 0x00 and
s 13..255 give 0xff. v_lerp_u8: each byte is (S0 + S1 + S2[8i]) >> 1. The words were assembled with LLVM 15
(llvm-mc -mcpu=gfx1013).

Two groups of 32 lanes; G = L + 32g is the lane's global index. v_perm_b32 uses S0 = 0x1b6c9e37, S1 = 0x80f07ad5
and S2 bytes 4G .. 4G+3, so the 64 lanes cover all 256 selectors once; a second v_perm_b32 exchanges S0 and S1. The
two words have different bits 15 and 31, and bits 7 and 23 differ from bits 15 and 31. v_lerp_u8 uses
S1 = G * 0x9e3779b9, S0 = S1 ^ 0xff80017f and S2 = G * 0x01030507. References with the sources exchanged, the sign
read from bit 7 instead of bit 15, selector 12 as 0xff, and the lerp rounding bit dropped or taken from every bit of
the S2 byte must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_align_bfm_check import words
from native_dpp_quad_check import request

# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 9 /
# s_lshl_b32 s8, s5, 5 / v_lshlrev_b32 v8, 4, v0 / v_add_nc_u32 v8, s7, v8 / v_add_nc_u32 v3, s8, v0 /
# v_mov_b32 v1, 0x1b6c9e37 / v_mov_b32 v2, 0x80f07ad5 / v_mul_lo_u32 v9, v3, 0x4040404 /
# v_add_nc_u32 v9, 0x3020100, v9 / v_perm_b32 v4, v1, v2, v9 / v_mul_lo_u32 v10, v3, 0x9e3779b9 /
# v_mul_lo_u32 v11, v3, 0x1030507 / v_xor_b32 v12, 0xff80017f, v10 / v_lerp_u8 v5, v12, v10, v11 /
# v_perm_b32 v7, v2, v1, v9 / s_waitcnt lgkmcnt(0) / buffer_store_dword v4, v5, v7 at v8 + 0, 4, 8 / s_endpgm
HEX = ('040394be 800395be 0a0808f4200000fa 0589078f 0585088f 84001034 0710104a 0800064a ff02027e379e6c1b '
       'ff02047ed57af080 090069d503ff010004040404 ff12124a00010203 040044d701052604 0a0069d503ff0100b979379e '
       '0b0069d503ff010007050301 ff14183a7f0180ff 05004dd50c152e04 070044d702032604 7fc08cbf '
       '001070e008040880 041070e008050880 081070e008070880 000081bf')
S0, S1 = 0x1B6C9E37, 0x80F07AD5
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
MASK = 0xFFFFFFFF


def perm(s0, s1, s2, sign_bits=(15, 31), twelve=0x00):
    pair = (s0 << 32) | s1
    out = 0
    for i in range(4):
        sel = (s2 >> (8 * i)) & 255
        if sel < 8:
            value = (pair >> (8 * sel)) & 255
        elif sel < 12:
            bit = (32 if sel >= 10 else 0) + sign_bits[sel & 1]
            value = 255 if (pair >> bit) & 1 else 0
        else:
            value = twelve if sel == 12 else 255
        out |= value << (8 * i)
    return out


def lerp(s0, s1, s2, rounding='bit0'):
    out = 0
    for i in range(4):
        a, b, c = ((x >> (8 * i)) & 255 for x in (s0, s1, s2))
        r = {'bit0': c & 1, 'none': 0, 'byte': c}[rounding]
        out |= (((a + b + r) >> 1) & 255) << (8 * i)
    return out


def reference(swapped=False, sign_bits=(15, 31), twelve=0x00, rounding='bit0'):
    expected = [SENTINEL] * OUTPUT_WORDS
    hi, lo = (S1, S0) if swapped else (S0, S1)
    for g in range(64):
        selectors = sum((4 * g + i) << (8 * i) for i in range(4))
        b = (g * 0x9E3779B9) & MASK
        a, c = b ^ 0xFF80017F, (g * 0x01030507) & MASK
        base = GUARD_WORDS + g * 4
        expected[base:base + 3] = [perm(hi, lo, selectors, sign_bits, twelve), lerp(a, b, c, rounding),
                                   perm(lo, hi, selectors, sign_bits, twelve)]
    return expected


def controls(cli, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    # Spot values of the definition: selectors 0..7 read {S0, S1} bytes, 8..11 the four sign bits, 12 is zero.
    assert perm(S0, S1, 0x03020100) == S1 and perm(S0, S1, 0x07060504) == S0
    assert perm(S0, S1, 0x0B0A0908) == 0x00FFFF00 and perm(S0, S1, 0xFF0D0C0C) == 0xFFFF0000
    assert lerp(0xFF00FF01, 0xFF01FF00, 0x01010101) == 0xFF01FF01 and lerp(0xFF00FF01, 0xFF01FF00, 0) == 0xFF00FF00
    expected = reference()
    for label, mutant in (('S0 and S1 exchanged', reference(swapped=True)),
                          ('sign read from bit 7 and 23', reference(sign_bits=(7, 23))),
                          ('selector 12 as 0xff', reference(twelve=0xFF)),
                          ('lerp rounding bit dropped', reference(rounding='none')),
                          ('lerp rounding from the whole S2 byte', reference(rounding='byte'))):
        assert mutant != expected, label
    print('compiled; the exchanged, sign-bit, selector-12 and both lerp-rounding references each differ')
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
    diff = [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    assert not diff, diff[:10]
    print('v_perm_b32 and v_lerp_u8: 192 words of two groups, all 256 selectors and the guards matched')
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
