"""Non-packed 16-bit integer VOP3 operations in the direct compute path (#669).

  native_int16_check.py controls <compile_dxil> <sdk bin dir>
  native_int16_check.py hardware <native_store_replay> <sdk bin dir>

v_add_nc_u16, v_sub_nc_u16, v_add_nc_i16, v_sub_nc_i16, v_mul_lo_u16, v_lshlrev_b16, v_lshrrev_b16, v_ashrrev_i16,
v_min_u16, v_max_u16, v_min_i16 and v_max_i16 write D[15:0] from S0[15:0] and S1[15:0] and keep D[31:16], as #669
records for RDNA 2: an inline-assembly measurement on RDNA 2 silicon, and LLVM's GCNSubtarget::zeroesHigh16BitsOfDest
("On gfx10, all 16-bit instructions preserve the high bits"). The shifts take S1 shifted by S0[3:0]. The words were
assembled with LLVM 15 (llvm-mc -mcpu=gfx1013).

Two groups of 32 lanes; G = L + 32g. S0 = G * 0x9e3779b9 and S1 = (G * 0x7f4a7c15) ^ 0x5bd1e995, so both high halves
are nonzero, the shift counts cover 0..15 and bit 4 of S0 varies. Each destination is set to 0xcafebabe first, and
the twelve results are stored per lane. References with a zeroed high half, a five-bit shift count, signed and
unsigned order exchanged, or a logical shift for v_ashrrev_i16 must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_align_bfm_check import words
from native_dpp_quad_check import request

# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 11 /
# s_lshl_b32 s8, s5, 5 / v_lshlrev_b32 v8, 6, v0 / v_add_nc_u32 v8, s7, v8 / v_add_nc_u32 v3, s8, v0 /
# v_mul_lo_u32 v1, v3, 0x9e3779b9 / v_mul_lo_u32 v2, v3, 0x7f4a7c15 / v_xor_b32 v2, 0x5bd1e995, v2 /
# for each op k in OPS: v_mov_b32 v[10+k], 0xcafebabe / <op> v[10+k], v1, v2 /
# s_waitcnt lgkmcnt(0) / buffer_store_dword v[10+k] at v8 + 4k / s_endpgm
HEX = ('040394be 800395be 0a0808f4200000fa 058b078f 0585088f 86001034 0710104a 0800064a 010069d503ff0100b979379e '
       '020069d503ff0100157c4a7f ff04043a95e9d15b ff02147ebebafeca 0a0003d701050200 ff02167ebebafeca 0b0004d701050200 '
       'ff02187ebebafeca 0c000dd701050200 ff021a7ebebafeca 0d000ed701050200 ff021c7ebebafeca 0e0005d701050200 '
       'ff021e7ebebafeca 0f0014d701050200 ff02207ebebafeca 100007d701050200 ff02227ebebafeca 110008d701050200 '
       'ff02247ebebafeca 12000bd701050200 ff02267ebebafeca 130009d701050200 ff02287ebebafeca 14000cd701050200 '
       'ff022a7ebebafeca 15000ad701050200 7fc08cbf 001070e0080a0880 041070e0080b0880 081070e0080c0880 '
       '0c1070e0080d0880 101070e0080e0880 141070e0080f0880 181070e008100880 1c1070e008110880 201070e008120880 '
       '241070e008130880 281070e008140880 2c1070e008150880 000081bf')
OPS = ("v_add_nc_u16", "v_sub_nc_u16", "v_add_nc_i16", "v_sub_nc_i16", "v_mul_lo_u16", "v_lshlrev_b16",
       "v_lshrrev_b16", "v_ashrrev_i16", "v_min_u16", "v_max_u16", "v_min_i16", "v_max_i16")
PRESET = 0xCAFEBABE
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
MASK = 0xFFFFFFFF


def sext16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def op16(name, s0, s1, shift_bits=4, swap_order=False, logical_ashr=False):
    a, b = s0 & 0xFFFF, s1 & 0xFFFF
    count = s0 & ((1 << shift_bits) - 1)
    signed = name.endswith("i16") != swap_order
    if name in ("v_add_nc_u16", "v_add_nc_i16"):
        r = a + b
    elif name in ("v_sub_nc_u16", "v_sub_nc_i16"):
        r = a - b
    elif name == "v_mul_lo_u16":
        r = a * b
    elif name == "v_lshlrev_b16":
        r = b << count
    elif name == "v_lshrrev_b16" or (name == "v_ashrrev_i16" and logical_ashr):
        r = b >> count
    elif name == "v_ashrrev_i16":
        r = sext16(b) >> count
    else:
        x, y = (sext16(a), sext16(b)) if signed else (a, b)
        r = min(x, y) if "_min_" in name else max(x, y)
    return r & 0xFFFF


def reference(zero_high=False, **wrong):
    expected = [SENTINEL] * OUTPUT_WORDS
    for g in range(64):
        s0 = (g * 0x9E3779B9) & MASK
        s1 = ((g * 0x7F4A7C15) & MASK) ^ 0x5BD1E995
        high = 0 if zero_high else PRESET & 0xFFFF0000
        base = GUARD_WORDS + g * 16
        expected[base:base + len(OPS)] = [high | op16(name, s0, s1, **wrong) for name in OPS]
    return expected


def controls(cli, sdk, work, root):
    marker = request(work, root, words(HEX))
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    # Spot values of the definitions.
    assert op16("v_sub_nc_u16", 0x10001, 0x20002) == 0xFFFF and op16("v_ashrrev_i16", 0x1F, 0x8000) == 0xFFFF
    assert op16("v_lshrrev_b16", 0x1F, 0x8000) == 1 and op16("v_min_i16", 0x8000, 0x7FFF) == 0x8000
    assert op16("v_min_u16", 0x8000, 0x7FFF) == 0x7FFF and op16("v_lshlrev_b16", 0x11, 0xC001) == 0x8002
    # op_sel (VOP3 word 0 bits 11..14) selects other halves and stays refused.
    program = words(HEX)
    at = program.index(0xD703000A)
    program[at] |= 1 << 14
    marker = request(work, root, program)
    refused = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, marker.with_suffix('.cso')],
                             capture_output=True, text=True)
    assert refused.returncode and "v_add_nc_u16 op_sel" in refused.stderr, (refused.stdout, refused.stderr)
    expected = reference()
    for label, mutant in (('high half zeroed', reference(zero_high=True)),
                          ('five-bit shift count', reference(shift_bits=5)),
                          ('signed and unsigned order exchanged', reference(swap_order=True)),
                          ('logical shift for v_ashrrev_i16', reference(logical_ashr=True))):
        assert mutant != expected, label
    print('compiled; op_sel refused; the zeroed-high, five-bit-shift, exchanged-order and logical-shift references '
          'each differ')
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
    print('16-bit integer VOP3: 768 words of two groups and the guards matched, high halves kept')
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
