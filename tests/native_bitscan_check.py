"""32-bit bit counts and bit scans in the direct compute path.

  native_bitscan_check.py controls <compile_dxil> <sdk bin dir>
  native_bitscan_check.py hardware <native_store_replay> <sdk bin dir>

The operations follow the RDNA 2 ISA:
- s_bcnt0_i32_b32 and s_bcnt1_i32_b32 count the zero or one bits of S0 and set SCC to D != 0.
- s_ff0_i32_b32, s_ff1_i32_b32 and v_ffbl_b32 return the index of the first zero or one bit from bit 0.
- s_flbit_i32_b32 and v_ffbh_u32 return the number of bits above the first one bit, counted from bit 31.
- s_flbit_i32 and v_ffbh_i32 do the same for the first bit that differs from the sign bit.
- Every scan returns 0xffffffff when no bit qualifies (so for 0, and for 0xffffffff in the signed and ff0 forms).
- No scan writes SCC.

The words are encoded with the helpers of the other native fixtures and decode with tools/disasm_rdna2.py to
the listing below.

The dispatch has two groups of 32 lanes. Lane L of group g holds x = (0x80000000 >> g) >> L (arithmetic) and
z = x ^ mul_u32_u24(L + 32g, 0x9e3779). So group 0 holds negative values down to 0xffffffff, and group 1 holds
single bits down to 0. The scalar inputs are s9 = g * 0xff0f0010 (0 in group 0) and s10 = ~s9.
Each lane stores 16 words:
- the three vector scans of x;
- the three vector scans of z;
- the six scalar results for s9;
- SCC after s_bcnt1 s9 with an s_ff1 in between, so a scan that wrote SCC would show;
- SCC after s_bcnt0 s9;
- s_flbit_i32 s10 and s_ff0 s10.

Each of these mutated references must differ from the expected words:
- scans that count from the wrong end;
- an unsigned scan in place of the signed one;
- a scan of 0 that returns 32;
- a bit count that leaves SCC alone;
- a scan that writes SCC.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove

SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
STRIDE = 16
MASK = 0xFFFFFFFF
LITERAL, ZERO, ONE = 255, 128, 129
# SOP1 opcodes
S_MOV, S_NOT, S_BCNT0, S_BCNT1, S_FF0, S_FF1, S_FLBIT, S_FLBIT_I32 = 3, 7, 13, 15, 17, 19, 21, 23
# SOP2 opcodes
S_CSELECT, S_LSHL, S_LSHR, S_MUL = 10, 30, 32, 38
# VOP1 opcodes
V_FFBH_U32, V_FFBL, V_FFBH_I32 = 0x39, 0x3A, 0x3B
# VOP2 opcodes
V_MUL_U24, V_LSHLREV, V_ASHRREV, V_XOR, V_ADD = 0x0B, 0x1A, 0x18, 0x1D, 0x25
VECTOR_SCANS = (V_FFBH_U32, V_FFBL, V_FFBH_I32)


def vop1(op, dst, src):
    return 0x7E000000 | dst << 17 | op << 9 | src


def mubuf_store(data, offset):
    # buffer_store_dword v<data>, v8, s[32:35], 0 offen offset:<offset>
    return [0xE0701000 | offset, 0x80080008 | data << 8]


def program(mutant=None):
    words = [sop1(S_MOV, 20, 4), sop1(S_MOV, 21, ZERO),          # s[20:21] = the table pointer
             0xF4080000 | 32 << 6 | 10, 0xFA000020,              # s_load_dwordx4 s[32:35], s[20:21], 0x20
             scalar(S_LSHL, 7, 5, 139),                          # s7 = group * 2048 bytes
             scalar(S_LSHL, 8, 5, 133),                          # s8 = group * 32
             vop2(V_LSHLREV, 8, 134, 0),                         # v8 = lane * 64 bytes
             vop2(V_ADD, 8, 7, 8),                               # v8 += s7
             vop2(V_ADD, 9, 8, 0),                               # v9 = n = lane + 32 * group
             scalar(S_LSHR, 10, LITERAL, 5), 0x80000000,         # s10 = 0x80000000 >> group
             vmove(2, 10),                                       # v2 = s10
             vop2(V_ASHRREV, 1, 256, 2),                         # v1 = x = v2 >> lane (arithmetic)
             vop2(V_MUL_U24, 3, LITERAL, 9), 0x009E3779,         # v3 = mul_u32_u24(n, 0x9e3779)
             vop2(V_XOR, 3, 257, 3)]                             # v3 = z = x ^ v3
    words += [vop1(op, 10 + k, 257) for k, op in enumerate(VECTOR_SCANS)]
    words += [vop1(op, 13 + k, 259) for k, op in enumerate(VECTOR_SCANS)]
    words += [scalar(S_MUL, 9, 5, LITERAL), 0xFF0F0010,         # s9 = group * 0xff0f0010
              sop1(S_NOT, 10, 9),                                # s10 = ~s9
              sop1(S_BCNT0, 11, 9),
              scalar(S_CSELECT, 16, ONE, ZERO),                  # s16 = SCC after s_bcnt0
              sop1(S_BCNT1, 12, 9),
              sop1(S_FF1, 13, 9),                                # must leave SCC from s_bcnt1
              scalar(S_CSELECT, 17, ONE, ZERO),
              sop1(S_FF0, 14, 9), sop1(S_FLBIT, 15, 9), sop1(S_FLBIT_I32, 18, 9),
              sop1(S_FLBIT_I32, 19, 10), sop1(S_FF0, 22, 10)]
    scalars = (11, 12, 14, 13, 15, 18, 17, 16, 19, 22)
    words += [vmove(16 + k, s) for k, s in enumerate(scalars)]
    if mutant == 'clamp':
        # v_ffbl_b32 v11, v1 clamp: the VOP3 form (opcode 0x180 + the VOP1 opcode) with bit 15 set.
        at = words.index(vop1(V_FFBL, 11, 257))
        words[at:at + 1] = [0xD4000000 | (0x180 + V_FFBL) << 16 | 0x8000 | 11, 257]
    words.append(0xBF8CC07F)                                     # s_waitcnt lgkmcnt(0)
    for k in range(STRIDE):
        words += mubuf_store(10 + k, 4 * k)
    return words + [0xBF810000]                                  # s_endpgm


def first_low(v):
    return (v & -v).bit_length() - 1 if v else MASK


def first_high(v):
    return 32 - v.bit_length() if v else MASK


def first_high_signed(v):
    return first_high(v ^ MASK if v & 0x80000000 else v)


def lane_words(n, mutation=None):
    group, lane = divmod(n, 32)
    base = 0x80000000 >> group
    x = ((base - (1 << 32) if base & 0x80000000 else base) >> lane) & MASK
    z = x ^ ((n * 0x9E3779) & MASK)
    low, high, signed = first_low, first_high, first_high_signed
    if mutation == 'wrong end':
        low, high = (lambda v: 31 - first_low(v) if v else MASK), (lambda v: 31 - first_high(v) if v else MASK)
        signed = lambda v: 31 - first_high_signed(v) if first_high_signed(v) != MASK else MASK
    elif mutation == 'unsigned':
        signed = first_high
    elif mutation == 'zero is 32':
        low, high = (lambda v: first_low(v) if v else 32), (lambda v: first_high(v) if v else 32)
    s9 = (group * 0xFF0F0010) & MASK
    s10 = s9 ^ MASK
    bcnt0, bcnt1 = 32 - bin(s9).count('1'), bin(s9).count('1')
    ff1 = low(s9)
    scc_bcnt1 = int(bcnt1 != 0)
    if mutation == 'scan writes SCC':
        scc_bcnt1 = int(ff1 != 0)
    scc_bcnt0 = int(bcnt0 != 0)
    if mutation == 'count leaves SCC':
        scc_bcnt0 = scc_bcnt1 = 0
    return [high(x), low(x), signed(x), high(z), low(z), signed(z),
            bcnt0, bcnt1, low(s9 ^ MASK), ff1, high(s9), signed(s9), scc_bcnt1, scc_bcnt0,
            signed(s10), low(s10 ^ MASK)]


def reference(mutation=None):
    expected = [SENTINEL] * OUTPUT_WORDS
    for n in range(64):
        base = GUARD_WORDS + n * STRIDE
        expected[base:base + STRIDE] = lane_words(n, mutation)
    return expected


def compile_words(cli, sdk, work, root, words):
    marker = request(work, root, words)
    output = marker.with_suffix('.cso')
    return subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                          capture_output=True, text=True), output


def controls(cli, sdk, work, root):
    result, output = compile_words(cli, sdk, work, root, program())
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    # The VOP3 form of v_ffbl_b32 with the clamp bit is refused, like the other integer VOP3 modifiers.
    result, _ = compile_words(cli, sdk, work, root, program('clamp'))
    assert result.returncode and 'direct ALU operand modifiers' in result.stderr, (result.stdout, result.stderr)
    xs = [lane_words(n)[:3] for n in range(64)]
    assert any(w[0] == MASK for w in xs) and any(w[2] == MASK for w in xs), 'a lane must scan 0 and 0xffffffff'
    expected = reference()
    for label in ('wrong end', 'unsigned', 'zero is 32', 'count leaves SCC', 'scan writes SCC'):
        assert reference(label) != expected, label
    print('compiled; clamp refused; the wrong-end, unsigned, zero-is-32, SCC-dropped and SCC-written references '
          'each differ')
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
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if x != y]
    assert not diff, diff[:10]
    print(f'bit counts and scans: {64 * STRIDE} words of two groups and the guards matched')
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
