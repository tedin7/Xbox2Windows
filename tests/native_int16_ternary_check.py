"""Three-source 16-bit integer VOP3 operations in the direct compute path (#669).

  native_int16_ternary_check.py controls <compile_dxil> <sdk bin dir>
  native_int16_ternary_check.py hardware <native_store_replay> <sdk bin dir>

Lowers the eight three-source 16-bit integer VOP3 operations:
- v_mad_u16, v_mad_i16
- v_min3_u16, v_min3_i16
- v_max3_u16, v_max3_i16
- v_med3_u16, v_med3_i16

Each operation computes D[15:0] from S0[15:0], S1[15:0] and S2[15:0], and D[31:16] preserves its previous value
(measured on RDNA 2 silicon; LLVM GCNSubtarget::zeroesHigh16BitsOfDest: "On gfx10, all 16-bit instructions preserve
the high bits"). Modifiers (clamp, scale, neg, abs) and op_sel are refused.

The dispatch has two groups of 32 lanes (64 lanes total). S0, S1 and S2 have nonzero high halves, cross the
signed 16-bit sign boundary, wrap mad modulo 2^16, and cover permutations of signed and unsigned orders.
Prior to each operation, destination registers are initialized to 0xcafebabe, so preserving D[31:16] leaves
0xcafe in the upper half.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

HEX = ('040394be 800395be 0a0808f4 200000fa 058a078f 0585088f 85001034 0710104a 0800124a ff120216 d1f30500 '
       'ff02023a 000080a5 ff120416 71030000 ff04043a debc9a7f ff120616 01100000 ff06063a ff7f0080 ff02147e '
       'bebafeca ff02167e bebafeca ff02187e bebafeca ff021a7e bebafeca ff021c7e bebafeca ff021e7e bebafeca '
       'ff02207e bebafeca ff02227e bebafeca 0a0040d7 01050e04 0b005ed7 01050e04 0c0053d7 01050e04 0d0052d7 '
       '01050e04 0e0056d7 01050e04 0f0055d7 01050e04 100059d7 01050e04 110058d7 01050e04 7fc08cbf 001070e0 '
       '080a0880 041070e0 080b0880 081070e0 080c0880 0c1070e0 080d0880 101070e0 080e0880 141070e0 080f0880 '
       '181070e0 08100880 1c1070e0 08110880 000081bf')

PRESET = 0xCAFEBABE
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
MASK32 = 0xFFFFFFFF


def words(text):
    out = []
    for h in text.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


def mul24(x, y):
    return ((x & 0xFFFFFF) * (y & 0xFFFFFF)) & MASK32


def operands(g):
    s0 = mul24(g, 0x05F3D1) ^ 0xA5800000
    s1 = mul24(g, 0x000371) ^ 0x7F9ABCDE
    s2 = mul24(g, 0x001001) ^ 0x80007FFF
    return s0, s1, s2


def sext16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def results(s0, s1, s2, zero_high=False, signed_as_unsigned=False, med_as_min=False,
            mad_no_add=False, mad_high_half=False):
    high = 0 if zero_high else (PRESET & 0xFFFF0000)
    a, b, c = s0 & 0xFFFF, s1 & 0xFFFF, s2 & 0xFFFF
    sa, sb, sc = sext16(s0), sext16(s1), sext16(s2)

    # mad
    if mad_no_add:
        r_mad_u = (a * b) & 0xFFFF
        r_mad_i = (sa * sb) & 0xFFFF
    elif mad_high_half:
        r_mad_u = ((s0 >> 16) * b + c) & 0xFFFF
        r_mad_i = ((s0 >> 16) * b + c) & 0xFFFF
    else:
        r_mad_u = (a * b + c) & 0xFFFF
        r_mad_i = (sa * sb + sc) & 0xFFFF

    # min3, max3, med3
    u_ordered = sorted([a, b, c])
    s_ordered = sorted([a, b, c]) if signed_as_unsigned else sorted([a, b, c], key=sext16)

    r_min_u = u_ordered[0]
    r_min_i = s_ordered[0]
    r_max_u = u_ordered[2]
    r_max_i = s_ordered[2]
    r_med_u = u_ordered[0 if med_as_min else 1]
    r_med_i = s_ordered[0 if med_as_min else 1]

    return [high | r_mad_u,
            high | r_mad_i,
            high | r_min_u,
            high | r_min_i,
            high | r_max_u,
            high | r_max_i,
            high | r_med_u,
            high | r_med_i]


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for lane in range(32):
            g = group * 32 + lane
            base = GUARD_WORDS + g * 8
            expected[base:base + 8] = results(*operands(g), **mutation)
    return expected


def controls(cli, sdk, work, root):
    prog = words(HEX)
    marker = request(work, root, prog)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)

    # op_sel on v_mad_u16 (bit 14 of word 0) must be refused
    op_sel_prog = list(prog)
    at = op_sel_prog.index(0xD740000A)
    op_sel_prog[at] |= (1 << 14)
    marker_op_sel = request(work, root, op_sel_prog)
    refused = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker_op_sel,
                              marker_op_sel.with_suffix('.cso')], capture_output=True, text=True)
    assert refused.returncode and "v_mad_u16 op_sel" in refused.stderr, (refused.stdout, refused.stderr)

    # clamp on v_mad_u16 (bit 15 of word 0) must be refused
    clamped_prog = list(prog)
    clamped_prog[at] |= (1 << 15)
    marker_clamp = request(work, root, clamped_prog)
    refused_clamp = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker_clamp,
                                    marker_clamp.with_suffix('.cso')], capture_output=True, text=True)
    assert refused_clamp.returncode and "direct ALU operand modifiers" in refused_clamp.stderr, (refused_clamp.stdout, refused_clamp.stderr)

    # Spot checks of definitions
    assert results(0x10002, 0x20003, 0x30004)[:2] == [PRESET & 0xFFFF0000 | 10, PRESET & 0xFFFF0000 | 10]
    # 0x8001 is -32767 signed, 32769 unsigned; 0x0001 is 1; 0x7FFF is 32767
    s0, s1, s2 = 0x8001, 0x0001, 0x7FFF
    res = results(s0, s1, s2)
    assert res[2] & 0xFFFF == 0x0001  # unsigned min3 is 1
    assert res[3] & 0xFFFF == 0x8001  # signed min3 is -32767
    assert res[4] & 0xFFFF == 0x8001  # unsigned max3 is 32769
    assert res[5] & 0xFFFF == 0x7FFF  # signed max3 is 32767
    assert res[6] & 0xFFFF == 0x7FFF  # unsigned med3 is 32767
    assert res[7] & 0xFFFF == 0x0001  # signed med3 is 1

    expected = reference()
    for label, mutant in (('high half zeroed', reference(zero_high=True)),
                          ('signed as unsigned', reference(signed_as_unsigned=True)),
                          ('median as minimum', reference(med_as_min=True)),
                          ('mad without addend', reference(mad_no_add=True)),
                          ('mad source from high half', reference(mad_high_half=True))):
        assert mutant != expected, f"Mutant failed to differ: {label}"

    print('compiled; op_sel and clamp refused; definitions spot-checked; all mutants differ')
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
    assert not diff, f"First diffs: {diff[:10]}"
    print('eight 16-bit ternary operations: 512 words of two groups and the guards matched, high halves kept')
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
