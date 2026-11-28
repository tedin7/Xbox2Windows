"""v_alignbit_b32, v_alignbyte_b32 and v_bfm_b32 in the direct compute path (#2985).

  native_align_bfm_check.py controls <compile_dxil> <sdk bin dir>
  native_align_bfm_check.py hardware <native_store_replay> <sdk bin dir>

The semantics are those of the gfx2hlsl.py VOP3 table: v_alignbit_b32 returns bits [k + 31 : k] of {S0, S1} with
k = S2[4:0], v_alignbyte_b32 the same with k = 8 * S2[1:0], and both return S1 for k = 0; v_bfm_b32 returns
((1 << S0[4:0]) - 1) << S1[4:0]. All words below were assembled with LLVM 15.0.7 (clang -cc1as, gfx1013).

The dispatch has two groups of 32 lanes. Lane L of group g uses S0 = 0x89abcdef, S1 = 0x01234567 and
S2 = L + 32g, so every shift 0..31 and every byte count appears, and group 1 repeats them with bit 5 set, which
the five-bit field must ignore. v_bfm_b32 takes S0 = S2 and S1 = S2 ^ 0x15. Each lane stores the three results.
References with the zero shift not special-cased, with S0 and S1 exchanged, and with the field not masked must
each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 9 /
# s_lshl_b32 s8, s5, 5 / v_lshlrev_b32 v8, 4, v0 / v_add_nc_u32 v8, s7, v8 / v_add_nc_u32 v3, s8, v0 /
# v_mov_b32 v1, 0x89abcdef / v_mov_b32 v2, 0x1234567 / v_alignbit_b32 v4, v1, v2, v3 /
# v_alignbyte_b32 v5, v1, v2, v3 / v_xor_b32 v6, 0x15, v3 / v_bfm_b32 v7, v3, v6 / s_waitcnt lgkmcnt(0) /
# buffer_store_dword v4, v5, v7 at v8 + 0, 4, 8 through s[32:35] / s_endpgm
HEX = ('040394be 800395be 0a0808f4200000fa 0589078f 0585088f 84001034 0710104a 0800064a ff02027eefcdab89 '
       'ff02047e67452301 04004ed501050e04 05004fd501050e04 95060c3a 070063d7030d0200 7fc08cbf '
       '001070e008040880 041070e008050880 081070e008070880 000081bf')
S0, S1 = 0x89ABCDEF, 0x01234567
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def words(text):
    out = []
    for h in text.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


def align(high, low, k, zero_special=True):
    """Bits [k + 31 : k] of {high, low}. Without the zero special case, a five-bit-masked 32 - k turns k = 0 into
    low | high, the result of an unguarded lowering."""
    if k == 0 and not zero_special:
        return low | high
    return (((high << 32) | low) >> k) & 0xFFFFFFFF


def reference(zero_special=True, swapped=False, unmasked=False):
    expected = [SENTINEL] * OUTPUT_WORDS
    high, low = (S1, S0) if swapped else (S0, S1)
    field = 63 if unmasked else 31
    for group in range(2):
        for lane in range(32):
            s2 = lane + 32 * group
            bit = align(high, low, s2 & field, zero_special)
            byte = align(high, low, 8 * (s2 & (7 if unmasked else 3)) % 64, zero_special)
            bfm = (((1 << (s2 & field)) - 1) << ((s2 ^ 0x15) & field)) & 0xFFFFFFFF
            base = GUARD_WORDS + (group * 32 + lane) * 4
            expected[base:base + 3] = [bit, byte, bfm]
    return expected


def controls(cli, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    expected = reference()
    for label, mutant in (('zero shift not special-cased', reference(zero_special=False)),
                          ('S0 and S1 exchanged', reference(swapped=True)),
                          ('field not masked to five bits', reference(unmasked=True))):
        assert mutant != expected, label
    print('compiled; the zero-shift, exchanged-source and unmasked references each differ from the expected words')
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
    print('v_alignbit_b32, v_alignbyte_b32 and v_bfm_b32: 192 words of two groups and the guards matched')
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
