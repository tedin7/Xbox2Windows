"""s_absdiff_i32 in the direct compute path.

  native_absdiff_check.py controls <compile_dxil> <sdk bin dir>
  native_absdiff_check.py hardware <native_store_replay> <sdk bin dir>

RDNA 2 ISA: D = S0 - S1 as a 32-bit signed difference; if D < 0 then D = -D; SCC = D != 0. The difference wraps,
and negating 0x80000000 gives 0x80000000. gfx2hlsl.py and tools/pixel_aot.py lower it the same way.

The words are encoded with the helpers of the other native fixtures and decode with tools/disasm_rdna2.py.

The dispatch has two groups of 32 lanes. s24 = group * 0x01000193, so group 0 sees the pairs below as written
and group 1 sees both operands of the register cases XORed with 0x01000193 (equal pairs stay equal). For each
case the program stores the result and SCC (through s_cselect_b32). The operands come from registers set at run
time, except the last two cases, whose S1 is a literal and the inline constant -1.
Before the equal pair, SCC is 1 (the s_xor_b32 before it leaves a non-zero value), so a lowering that kept the
old SCC would show there.

Each of these mutated references must differ from the expected words:
- a difference that does not wrap (exact integers, then the low 32 bits);
- the unsigned distance max(S0, S1) - min(S0, S1);
- 0x80000000 saturated to 0x7fffffff;
- the difference without the absolute value;
- SCC left unchanged.
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
LITERAL, ZERO, ONE, MINUS_ONE = 255, 128, 129, 193
GROUP_XOR = 0x01000193
# SOP1 and SOP2 opcodes
S_MOV = 3
S_CSELECT, S_XOR, S_LSHL, S_MUL, S_ABSDIFF = 10, 18, 30, 38, 44
# VOP2 opcodes
V_LSHLREV, V_ADD = 0x1A, 0x25
# Register pairs (S0, S1): ordinary, swapped, the wrap at 0x80000000 and the equal pair.
PAIRS = [(5, 3), (3, 5), (0x80000000, 0), (0x7FFFFFFF, 0xFFFFFFFF), (0x80000000, 1), (7, 7)]
# Constant S1: a literal and the inline constant -1.
LITERAL_CASE = (0x00000010, 0xFFFFFFF0)
INLINE_CASE = 0x7FFFFFFE


def mubuf_store(data, offset):
    # buffer_store_dword v<data>, v8, s[32:35], 0 offen offset:<offset>
    return [0xE0701000 | offset, 0x80080008 | data << 8]


def program():
    words = [sop1(S_MOV, 20, 4), sop1(S_MOV, 21, ZERO),          # s[20:21] = the table pointer
             0xF4080000 | 32 << 6 | 10, 0xFA000020,              # s_load_dwordx4 s[32:35], s[20:21], 0x20
             scalar(S_LSHL, 7, 5, 139),                          # s7 = group * 2048 bytes
             vop2(V_LSHLREV, 8, 134, 0),                         # v8 = lane * 64 bytes
             vop2(V_ADD, 8, 7, 8),                               # v8 += s7
             scalar(S_MUL, 24, 5, LITERAL), GROUP_XOR]           # s24 = group * 0x01000193
    out = 10
    for a, b in PAIRS:
        words += [scalar(S_XOR, 9, LITERAL, 24), a,              # s9 = a ^ s24
                  scalar(S_XOR, 10, LITERAL, 24), b,             # s10 = b ^ s24
                  scalar(S_ABSDIFF, 11, 9, 10),
                  scalar(S_CSELECT, 12, ONE, ZERO),
                  vmove(out, 11), vmove(out + 1, 12)]
        out += 2
    words += [scalar(S_XOR, 9, LITERAL, 24), LITERAL_CASE[0],
              scalar(S_ABSDIFF, 11, 9, LITERAL), LITERAL_CASE[1],
              scalar(S_CSELECT, 12, ONE, ZERO),
              vmove(out, 11), vmove(out + 1, 12),
              scalar(S_XOR, 9, LITERAL, 24), INLINE_CASE,
              scalar(S_ABSDIFF, 11, 9, MINUS_ONE),
              scalar(S_CSELECT, 12, ONE, ZERO),
              vmove(out + 2, 11), vmove(out + 3, 12)]
    words.append(0xBF8CC07F)                                     # s_waitcnt lgkmcnt(0)
    for k in range(STRIDE):
        words += mubuf_store(10 + k, 4 * k)
    return words + [0xBF810000]                                  # s_endpgm


def signed(v):
    return v - (1 << 32) if v & 0x80000000 else v


def absdiff(a, b, mutation=None):
    if mutation == 'no wrap':
        return abs(signed(a) - signed(b)) & MASK
    if mutation == 'unsigned':
        return max(a, b) - min(a, b)
    d = (a - b) & MASK
    if mutation == 'no abs':
        return d
    d = (-d) & MASK if d & 0x80000000 else d
    if mutation == 'saturated' and d == 0x80000000:
        return 0x7FFFFFFF
    return d


def operands(group):
    key = group * GROUP_XOR
    cases = [(a ^ key, b ^ key) for a, b in PAIRS]
    return cases + [(LITERAL_CASE[0] ^ key, LITERAL_CASE[1]), (INLINE_CASE ^ key, MASK)]


def group_words(group, mutation=None):
    words = []
    for a, b in operands(group):
        d = absdiff(a, b, mutation)
        words += [d, 1 if mutation == 'SCC unchanged' else int(d != 0)]
    return words


def reference(mutation=None):
    expected = [SENTINEL] * OUTPUT_WORDS
    for n in range(64):
        base = GUARD_WORDS + n * STRIDE
        expected[base:base + STRIDE] = group_words(n // 32, mutation)
    return expected


def compile_words(cli, sdk, work, root, words):
    marker = request(work, root, words)
    output = marker.with_suffix('.cso')
    return subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                          capture_output=True, text=True), output


def controls(cli, sdk, work, root):
    result, output = compile_words(cli, sdk, work, root, program())
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    for group in (0, 1):
        words = group_words(group)
        assert 0x80000000 in words[::2] and 0x7FFFFFFF in words[::2], 'both wrap results must be present'
        assert words[11] == 0 and words[10] == 0, 'the equal pair must give 0 and SCC 0'
    expected = reference()
    for label in ('no wrap', 'unsigned', 'saturated', 'no abs', 'SCC unchanged'):
        assert reference(label) != expected, label
    print('compiled; the unwrapped, unsigned, saturated, signed-difference and SCC-unchanged references each differ')
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
    print(f's_absdiff_i32: {64 * STRIDE} words of two groups and the guards matched')
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
