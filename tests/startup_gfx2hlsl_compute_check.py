"""Compute stage of tools/startup/gfx2hlsl.py: entry registers, numthreads and the offen buffer store (#1412).

  (no arguments)                                  translation controls, pure Python
  binary <llvm-mc>                                reassemble the fixture sources and compare them with the words below
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                                  translate, compile with DXC cs_6_0, dispatch and compare every word
  control <swapped-axes|dropped-store|b-offset|c-unpredicated|j-scalar-dropped|h-inner-unpredicated> <dxc> <offscreen_check> <warp|hardware>
                                                  the same dispatch with a mutated translation, which must mismatch

The fixtures are source-built gfx1013 compute programs. The root signature puts two root constants (key, stride)
in s4-s5 and a descriptor table in s6; the table holds a read buffer at byte 0 and a writable buffer at byte 16.
COMPUTE_PGM_RSRC2 and the thread group size are pipeline register state, given here as the runtime would give them.
Each thread names its group and local IDs only by the registers that state gives them, computes its own index,
reads the read buffer at that index and stores to its own destination:

- case A enables all three group-ID axes and v0-v2, with group size 2x3x4 and 3x2x2 groups, and stores one word;
- case B enables X and Z only (Y omitted, so Z is s8), v0-v1, group size 5x2x1 and 4x1x3 groups, and stores two
  words at an immediate offset of 8 bytes;
- case C enables Y only (s7), v0 only, group size 8x1x1 and 1x5x1 groups, stores four words with every lane, then
  three words at offset 16 under an exec mask set from a compare (lanes with tid.x < 6), and restores exec;
- case D counts sparse mask bits below each lane with v_mbcnt_lo/hi, then obtains the full-mask count in lanes
  16-31 under partial EXEC. Two 4x2x4 groups write separate records to the raw buffer.
- case E runs five SAVEEXEC forms with overlapping and distinct source registers. Four single-thread groups
  cover both input bits for EXEC and S0; each stores the saved EXEC, resulting SCC and new EXEC.
- case F exchanges LDS dwords across the two halves of each 64-thread group and broadcasts lane 37.
- case J enables X only, v0, group size 8x1x1 and 4x1x1 groups, leaves lanes tid.x < 3 active and branches over
  a block with s_cbranch_execz; the block loads the store descriptor and computes a scalar from the group ID and
  the root constants that the active lanes store, and its scalars are dead where it joins;
- case H is case J's shape with a nested if inside the skipped block (s_and_saveexec, restored with s_or of the
  saved mask): lanes tid.x < 3 store two words, lanes 3-5 one word, the rest none.
- case G reads two independently offset LDS dwords with the address in the first destination VGPR.

The dispatch runs on a plain D3D12 device (the stand-in has no slot-67 path yet). The root signature is the runner's:
root SRV t0 (read buffer), root UAV u0 (writable buffer) and 16 root constants at b0, which match the translation's
r0, u0 and entry cbuffer. The manifest's table lines and the descriptor fetch are therefore not exercised on the GPU.
All 320 words are compared, so a word outside the written records must keep the sentinel. The controls swap the X
and Y group IDs at entry, drop the store, drop case B's immediate offset, drop case C's exec predicate, or drop a
scalar statement inside case J's block; each must change the readback.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

SENTINEL = 0xc5c5c5c5
WORDS = 320
MASK = 0xffffffff
KEY, STRIDE = 0x01234567, 0x9e3779b9
TGID_X, TGID_Y, TGID_Z, TG_SIZE_EN = 1 << 7, 1 << 8, 1 << 9, 1 << 10
USER7 = 7 << 1  # s0-s3, two root constants, one table


def tidig(components):
    return (components - 1) << 11


SOURCE_A = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_mad_u32_u24 v3, v1, 2, v0
v_mad_u32_u24 v3, v2, 6, v3
s_mul_i32 s20, s8, 3
s_add_u32 s20, s20, s7
s_mul_i32 s21, s9, 6
s_add_u32 s20, s20, s21
s_mul_i32 s20, s20, 24
v_add_nc_u32 v3, s20, v3
v_lshlrev_b32 v4, 2, v3
buffer_load_dword v5, v4, s[12:15], 0 offen
v_mul_lo_u32 v6, v3, s5
v_add_nc_u32 v6, s4, v6
s_waitcnt vmcnt(0)
v_mul_lo_u32 v5, v5, 33
v_xor_b32 v5, v5, v6
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_A = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '030043d501050104' '030043d5020d0d04' '08831493' '14071480'
    '09861593' '14151480' '14981493' '1406064a' '82060834' '001030e004050380' '060069d5030b0000' '040c0c4a'
    '703f8cbf' '050069d505430100' '050d0a3a' '001070e004050480' '000081bf')

SOURCE_B = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_mad_u32_u24 v3, v1, 5, v0
s_lshl_b32 s20, s8, 2
s_add_u32 s20, s20, s7
s_mul_i32 s20, s20, 10
v_add_nc_u32 v3, s20, v3
v_lshlrev_b32 v4, 3, v3
v_lshlrev_b32 v7, 2, v3
buffer_load_dword v5, v7, s[12:15], 0 offen
v_mul_lo_u32 v6, v3, s5
v_add_nc_u32 v6, s4, v6
s_waitcnt vmcnt(0)
v_xor_b32 v5, v5, v6
v_xor_b32 v6, 0x5a000000, v3
buffer_store_dwordx2 v[5:6], v4, s[16:19], 0 offen offset:8
s_endpgm
"""
WORDS_B = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '030043d5010b0104' '0882148f' '14071480' '148a1493'
    '1406064a' '83060834' '82060e34' '001030e007050380' '060069d5030b0000' '040c0c4a' '703f8cbf' '050d0a3a'
    'ff060c3a0000005a' '081074e004050480' '000081bf')

SOURCE_C = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v7, 2, v3
buffer_load_dword v8, v7, s[12:15], 0 offen
v_mul_u32_u24 v4, 28, v3
v_mov_b32 v9, v3
v_mul_lo_u32 v10, v3, s5
v_add_nc_u32 v10, s4, v10
s_waitcnt vmcnt(0)
v_xor_b32 v11, v8, v10
buffer_store_dwordx4 v[8:11], v4, s[16:19], 0 offen
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 6, v0
s_mov_b32 exec_lo, vcc_lo
buffer_store_dwordx3 v[8:10], v4, s[16:19], 0 offen offset:16
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_C = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '82060e34' '001030e007080380'
    '9c060816' '0303127e' '0a0069d5030b0000' '0414144a' '703f8cbf' '0815163a' '001078e004080480' '7e0395be'
    '8600887d' '6a03febe' '10107ce004080480' '1503febe' '000081bf')

SOURCE_D = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 0
s_mov_b32 s20, 0x85241249
v_mad_u32_u24 v3, v2, 2, v1
v_mad_u32_u24 v3, v3, 4, v0
v_lshlrev_b32 v4, 5, s7
v_add_nc_u32 v4, v4, v3
v_lshlrev_b32 v4, 4, v4
v_mov_b32 v6, 7
v_mbcnt_lo_u32_b32 v8, s20, v6
v_mbcnt_hi_u32_b32 v9, s20, v8
v_mov_b32 v10, -1
s_mov_b32 s21, exec_lo
v_cmp_lt_u32 vcc_lo, 15, v3
s_mov_b32 exec_lo, vcc_lo
v_mbcnt_lo_u32_b32 v10, -1, 0
s_mov_b32 exec_lo, s21
v_mbcnt_hi_u32_b32 v11, -1, v3
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[8:11], v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_D = bytes.fromhex(
    '06038abe80038bbe050408f4000000faff0394be49122485030043d502050504030043d50309010404001ad5850e0000'
    '0407084a8408083487020c7e080065d7140c0200090066d714100200c102147e7e0395be8f06827d6a03febe'
    '0a0065d7c10001001503febe0b0066d7c10602007fc08cbf001078e004080480000081bf')
# A known SGPR base of UINT_MAX wraps to zero when the first mask bit is counted.
SOURCE_D_WRAP = SOURCE_D.replace('v_mov_b32 v6, 7', 's_mov_b32 s22, -1').replace('v8, s20, v6', 'v8, s20, s22')
WORDS_D_WRAP = WORDS_D.replace(bytes.fromhex('87020c7e080065d7140c0200'), bytes.fromhex('c10396be080065d7142c0000'))
SOURCE_J =  """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[12:15], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 3, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz 10
s_load_dwordx4 s[16:19], s[10:11], 0x10
s_mul_i32 s22, s7, s5
s_add_u32 s22, s22, s4
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0) lgkmcnt(0)
v_xor_b32 v5, s22, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_J = bytes.fromhex(
    '06038abe' '80038bbe' '050308f4000000fa' '7fc08cbf' '0783148f' '1400064a' '82060834' '7e0395be' '8300887d'
    '6a03febe' '0a0088bf' '050408f4100000fa' '07051693' '16041680' '001030e004050380' '70008cbf' '160a0a3a'
    '001070e004050480' '1503febe' '000081bf')

SOURCE_H = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 3, v3
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 6, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz 11
s_mul_i32 s22, s7, s5
s_add_u32 s22, s22, s4
v_xor_b32 v5, s22, v3
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_saveexec_b32 s0, vcc_lo
buffer_store_dword v5, v4, s[16:19], 0 offen
s_or_b32 exec_lo, exec_lo, s0
v_not_b32 v6, v5
buffer_store_dword v6, v4, s[16:19], 0 offen offset:4
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_H = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '83060834' '7e0395be' '8600887d'
    '6a03febe' '0b0088bf' '07051693' '16041680' '16060a3a' '8300887d' '6a3c80be' '001070e004050480' '7e007e88'
    '056f0c7e' '041070e004060480' '1503febe' '000081bf')

# Case K is case J with the condition on the group ID: only group 1 enters the block, and in groups 0, 2 and 3 the
# whole wave skips it.
SOURCE_K = SOURCE_J.replace('v_cmp_gt_u32 vcc_lo, 3, v0\n', 'v_mov_b32 v6, s7\nv_cmp_eq_u32_e32 vcc_lo, 1, v6\n')
WORDS_K = WORDS_J.replace(bytes.fromhex('8300887d'), bytes.fromhex('07020c7e' '810c847d'))

def saveexec_fixture():
    # RDNA 2 ISA, SOP1: D takes old EXEC, then EXEC takes the bitwise result and SCC reports nonzero.
    # A single-thread group makes the lane bit the whole participating wave mask.
    program = [
        ('s_mov_b32 s10, s6', '06038abe'),
        ('s_mov_b32 s11, 0', '80038bbe'),
        ('s_load_dwordx8 s[12:19], s[10:11], 0x0', '05030cf4000000fa'),
        ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
        ('v_lshlrev_b32 v4, 2, s7', '04001ad5820e0000'),
        ('s_and_b32 s22, s7, 1', '07811687'),
        ('s_mov_b32 s23, 0', '800397be'),
        ('s_mov_b32 s21, 0', '800395be'),
        ('s_lshr_b32 s26, s7, 1', '07811a90'),
    ]
    snapshot = [
        ('s_cselect_b32 s28, 1, 0', '81801c85'),
        ('s_mov_b32 s29, exec_lo', '7e039dbe'),
        ('s_mov_b32 exec_lo, 1', '8103febe'),  # every group writes its result, including EXEC == 0
        ('v_lshlrev_b32 v5, 2, s20', '05001ad582280000'),
        ('v_lshlrev_b32 v6, 1, s28', '06001ad581380000'),
        ('v_or_b32 v5, v5, v6', '050d0a38'),
        ('v_or_b32 v5, s29, v5', '1d0a0a38'),
        ('buffer_store_dword v5, v4, s[16:19], 0 offen', '001070e004050480'),
        ('v_add_nc_u32 v4, 16, v4', '9008084a'),
    ]
    for op, alias_word, distinct_word in (
        ('s_and_saveexec_b64', '142494be', '162494be'),
        ('s_and_saveexec_b32', '143c94be', '163c94be'),
        ('s_or_saveexec_b64', '142594be', '162594be'),
        ('s_or_saveexec_b32', '143d94be', '163d94be'),
        ('s_andn2_saveexec_b64', '142794be', '162794be'),
    ):
        dst, src = ('s[20:21]', 's[22:23]') if op.endswith('b64') else ('s20', 's22')
        for init, operand, word in (
            (('s_mov_b32 s20, s22', '160394be'), dst, alias_word),
            (('s_xor_b32 s20, s22, 1', '16811489'), src, distinct_word),
        ):
            program.extend((init, ('s_mov_b32 exec_lo, s26', '1a03febe'), (f'{op} {dst}, {operand}', word)))
            program.extend(snapshot)
    program.append(('s_endpgm', '000081bf'))
    return '\n'.join(line for line, _ in program) + '\n', bytes.fromhex(''.join(word for _, word in program))


SOURCE_F = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 8
v_add_nc_u32 v5, s20, v0
v_lshlrev_b32 v4, 2, v0
ds_write_b32 v4, v5 offset:16
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v8, 0x80, v4
v_and_b32 v8, 0xfc, v8
ds_read_b32 v6, v8 offset:16
s_waitcnt lgkmcnt(0)
v_mov_b32 v8, 0
ds_read_b32 v7, v8 offset:164
s_waitcnt lgkmcnt(0)
s_lshl_b32 s21, s7, 6
v_add_nc_u32 v9, s21, v0
v_lshlrev_b32 v9, 3, v9
buffer_store_dwordx2 v[6:7], v9, s[16:19], 0 offen
s_endpgm
"""
WORDS_F = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf0788148f14000a4a82000834100034d804050000'
    '7fc08cbf00008abfff08103a80000000ff101036fc0000001000d8d8080000067fc08cbf8002107e'
    'a400d8d8080000077fc08cbf0786158f1500124a83121234001074e009060480000081bf')

SOURCE_G = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_xor_b32 v5, s20, v0
v_and_b32 v5, 31, v5
v_lshlrev_b32 v5, 2, v5
v_lshlrev_b32 v4, 2, v0
ds_write_b32 v4, v5 offset:16
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32 v6, 0x7c, v4
ds_read2_b32 v[6:7], v6 offset0:5 offset1:36
s_waitcnt lgkmcnt(0)
s_lshl_b32 s21, s7, 6
v_add_nc_u32 v9, s21, v0
v_lshlrev_b32 v9, 3, v9
buffer_store_dwordx2 v[6:7], v9, s[16:19], 0 offen
s_endpgm
"""
WORDS_G = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf0783148f14000a3a9f0a0a36820a0a3482000834100034d804050000'
    '7fc08cbf00008abfff080c367c0000000524dcd8060000067fc08cbf0786158f1500124a83121234001074e009060480000081bf')

CASES = {
    # name: (source, words, rsrc2, group size, group counts)
    'A': (SOURCE_A, WORDS_A, USER7 | TGID_X | TGID_Y | TGID_Z | tidig(3), (2, 3, 4), (3, 2, 2)),
    'B': (SOURCE_B, WORDS_B, USER7 | TGID_X | TGID_Z | tidig(2), (5, 2, 1), (4, 1, 3)),
    'C': (SOURCE_C, WORDS_C, USER7 | TGID_Y | tidig(1), (8, 1, 1), (1, 5, 1)),
    'D': (SOURCE_D, WORDS_D, USER7 | TGID_X | tidig(3), (4, 2, 4), (2, 1, 1)),
    'D-wrap': (SOURCE_D_WRAP, WORDS_D_WRAP, USER7 | TGID_X | tidig(3), (4, 2, 4), (2, 1, 1)),
    'E': (*saveexec_fixture(), USER7 | TGID_X | tidig(1), (1, 1, 1), (4, 1, 1)),
    'F': (SOURCE_F, WORDS_F, USER7 | TGID_X | (1 << 15), (64, 1, 1), (2, 1, 1)),
    'J': (SOURCE_J, WORDS_J, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'H': (SOURCE_H, WORDS_H, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'K': (SOURCE_K, WORDS_K, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'G': (SOURCE_G, WORDS_G, USER7 | TGID_X | (1 << 15), (64, 1, 1), (2, 1, 1)),
}

# Single instructions the controls splice into case A before its store, recorded from llvm-mc 15.
STORE_A = WORDS_A.index(bytes.fromhex('001070e004050480'))
SPLICE = {
    's_cmp_lt_u32 s7, 4': '07840abf',
    's_and_b32 s13, s13, 0xc000ffff': '0dff0d87ffff00c0',
    's_add_u32 s13, s13, 1': '0d810d80',
    's_mov_b32 s13, s13': '0d038dbe',
    's_cselect_b32 s20, 1, 0': '81801485',
    's_cselect_b64 s[20:21], 1, 0': '81809485',
    's_cmov_b32 s20, 1': '810594be',
    's_cmov_b64 s[20:21], s[4:5]': '040694be',
    's_cmovk_i32 s20, 3': '030014b1',
    's_cbranch_scc0 1': '010084bf',
    's_cbranch_scc1 1': '010085bf',
    's_cbranch_vccz 1': '010086bf',
    'v_mov_b32 v3, 0': '8002067e',
    'v_add_nc_u32 v3, 1, v3': '8106064a',
    'v_cmp_gt_u32 vcc_lo, 2, v3': '8206887d',
    's_cbranch_vccnz 65530': 'faff87bf',
    's_branch 2': '020082bf',
    's_and_b32 s22, s7, 1': '07811687',
    's_mov_b32 s18, 0': '800392be',
    's_or_b32 s17, s17, 0x40000': '11ff118800000400',
    's_and_b32 s18, s18, 0xffff': '12ff1287ffff0000',
    's_mov_b32 s18, 4': '840392be',
    'v_cmp_gt_u32 vcc_lo, 1, v0': '8100887d',
    's_and_saveexec_b32 s0, vcc_lo': '6a3c80be',
    's_cbranch_execz 1': '010088bf',
    's_add_u32 s20, s7, 1': '07811480',
    's_or_b32 exec_lo, exec_lo, s0': '7e007e88',
    'v_mov_b32 v5, s20': '14020a7e',
    's_mov_b32 s21, exec_lo': '7e0395be',
    's_mov_b32 exec_lo, vcc_lo': '6a03febe',
    's_mov_b32 exec_lo, s21': '1503febe',
    's_mov_b64 s[20:21], 0': '800494be',
    's_mov_b64 s[22:23], 0': '800496be',
    's_load_dwordx8 s[24:31], s[10:11], 0x40': '05060cf4400000fa',
    'image_sample v[12:15], v[0:1], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f80f0000ca600',
    'image_sample_lz v[12:15], v[0:1], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_2D': '080f9cf0000ca600',
    'buffer_store_dword v5, v4, s[16:19], 0 offen': '001070e004050480',
    'buffer_store_dword v5, v4, s[16:19], 0 offen offset:6': '061070e004050480',
    'buffer_store_dword v5, v4, s[16:19], 0 offen offset:4092': 'fc1f70e004050480',
    's_add_u32 s23, s7, 1': '07811780',
    's_cselect_b32 s23, 1, 0': '81801785',
    'v_cmp_gt_u32 vcc_lo, 3, v0': '8300887d',
    's_cbranch_execz 0': '000088bf',
    's_cbranch_execz 3': '030088bf',
    's_cbranch_execz 65534': 'feff88bf',
    's_or_b32 exec_lo, exec_lo, s21': '7e157e88',
    's_and_b32 s17, s17, 0xc000ffff': '11ff1187ffff00c0',
    's_and_b32 s17, s17, 0xc000fff0': '11ff1187f0ff00c0',
    's_and_b32 s18, s18, 0xc000ffff': '12ff1287ffff00c0',
    's_and_b32 s17, 0xc000ffff, s17': 'ff111187ffff00c0',
    's_and_b32 s21, s21, 0xc000ffff': '15ff1587ffff00c0',
}


def splice(*lines, at=STORE_A):
    return WORDS_A[:at] + b''.join(bytes.fromhex(SPLICE[line]) for line in lines) + WORDS_A[at:]


# Case I is case A with the store descriptor's stride field cleared before the store (expected words as case A).
STRIDE_CLEAR = 's_and_b32 s17, s17, 0xc000ffff'
CASES['I'] = (SOURCE_A.replace('buffer_store_dword', STRIDE_CLEAR + '\nbuffer_store_dword'), splice(STRIDE_CLEAR),
              *CASES['A'][2:])

# Replace the nonzero stride constant in s5 with the first active lane of a
# fresh 24-thread group. Ignoring FF1 leaves a different word at every i > 0.
CASES['A-ff1'] = ('s_ff1_i32_b32 s5, exec_lo\n' + SOURCE_A,
                  bytes.fromhex('7e1385be') + WORDS_A, *CASES['A'][2:])


def root_blob(constants=2, table_range=False):
    """RTS0: param 0 = root constants (b0), param 1 = a descriptor table with no ranges.

    table_range gives the table one SRV range (t0, one descriptor) for a blob that a device creates: the Windows 10
    in-box D3D12 runtime (10.0.19041) corrupts its heap in CreateRootSignature for a table with no ranges. The
    default bytes are unchanged because they are part of pinned request keys.
    """
    params_at, constants_at = 24, 24 + 2 * 12
    table_at = constants_at + 12
    ranges = struct.pack('<6I', 0, 1, 0, 0, 0, 0xffffffff) if table_range else b''  # SRV, 1, t0, space 0, append
    rts0 = (struct.pack('<6I', 2, 2, params_at, 0, table_at + 8 + len(ranges), 0)
            + struct.pack('<3I', 1, 0, constants_at) + struct.pack('<3I', 0, 0, table_at)
            + struct.pack('<3I', 0, 0, constants) + struct.pack('<2I', len(ranges) // 24, table_at + 8) + ranges)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part


def input_word(i):
    return (0x8b02a7f1 + i * 0x413b1) & MASK  # tests/compute_check.cpp fills the read buffer this way


def expected(name):
    if name == 'I':
        return expected('A')
    out = [SENTINEL] * WORDS
    _, _, _, local, groups = CASES[name]
    producer_offsets = {'G': (1, 32), 'G-offset0-only': (31, 0), 'G-offset1-only': (0, 31)}
    if name in producer_offsets:
        # The original fill begins at dword 4, the single-field variants at dword 0.
        first, second = producer_offsets[name]
        for group in range(2):
            stored = [4 * ((lane ^ (8 * group)) & 31) for lane in range(64)]
            for lane in range(64):
                base = lane & 31
                index = 2 * (64 * group + lane)
                out[index:index + 2] = [stored[base + first], stored[base + second]]
        return out
    if name == 'F':
        for group in range(2):
            for lane in range(64):
                index = (64 * group + lane) * 2
                out[index:index + 2] = [256 * group + (lane ^ 32), 256 * group + 37]
        return out
    if name == 'E':
        # Group IDs 0..3 encode (old EXEC, S0) = (0,0), (0,1), (1,0), (1,1).
        # Store old EXEC in bit 2, SCC in bit 1 and new EXEC in bit 0. Both operand layouts agree.
        and_bits, or_bits, andn2_bits = [0, 0, 4, 7], [0, 3, 7, 7], [0, 3, 4, 4]
        out[:40] = and_bits * 4 + or_bits * 4 + andn2_bits * 2
        return out
    if name in ('D', 'D-wrap'):
        for group in range(groups[0]):
            count = 7 if name == 'D' else MASK
            for lane in range(32):
                i = 4 * (32 * group + lane)
                out[i:i + 4] = [count, count, lane if lane >= 16 else MASK, lane]
                count = (count + ((0x85241249 >> lane) & 1)) & MASK
        return out
    for gz in range(groups[2]):
        for gy in range(groups[1]):
            for gx in range(groups[0]):
                for tz in range(local[2]):
                    for ty in range(local[1]):
                        for tx in range(local[0]):
                            if name in ('A', 'A-ff1'):
                                i = (gx + 3 * gy + 6 * gz) * 24 + tx + 2 * ty + 6 * tz
                                stride = 0 if name == 'A-ff1' else STRIDE
                                out[i] = ((input_word(i) * 33) & MASK) ^ ((KEY + i * stride) & MASK)
                            elif name == 'K':
                                i = gx * 8 + tx
                                if gx == 1:  # only group 1's wave enters the block
                                    out[i] = input_word(i) ^ ((gx * STRIDE + KEY) & MASK)
                            elif name == 'H':
                                i = gx * 8 + tx
                                value = i ^ ((KEY + gx * STRIDE) & MASK)
                                if tx < 3:  # the nested if inside the skipped block
                                    out[2 * i] = value
                                if tx < 6:  # the skipped block
                                    out[2 * i + 1] = ~value & MASK
                            elif name == 'J':
                                i = gx * 8 + tx
                                if tx < 3:  # the lanes the compare leaves active; the others skip the block
                                    out[i] = input_word(i) ^ ((gx * STRIDE + KEY) & MASK)
                            elif name == 'C':
                                i = gy * 8 + tx
                                record = [input_word(i), i, (KEY + i * STRIDE) & MASK]
                                out[7 * i:7 * i + 4] = record + [record[0] ^ record[2]]
                                if tx < 6:
                                    out[7 * i + 4:7 * i + 7] = record
                            else:
                                i = (gx + 4 * gz) * 10 + tx + 5 * ty
                                out[2 + 2 * i] = input_word(i) ^ ((KEY + i * STRIDE) & MASK)
                                out[3 + 2 * i] = 0x5a000000 ^ i
    return out


def translate(name, rsrc2=None, code=None, group=None):
    source, words, default_rsrc2, local, _ = CASES[name]
    return G.ComputePipeline(code or words, root_blob(), default_rsrc2 if rsrc2 is None else rsrc2, group or local)


def refused(fragment, build):
    try:
        build()
    except G.Unsupported as reason:
        assert fragment in str(reason), (fragment, str(reason))
        return
    raise AssertionError('accepted, expected a refusal naming %r' % fragment)


def patch(words, old, new):
    old, new = bytes.fromhex(old), bytes.fromhex(new)
    assert words.count(old) == 1, old.hex()
    return words.replace(old, new)


# Move the fill to dword zero so either read field can be omitted. Both variants retain the
# overlapping destination/address and participate in the regular LLVM reassembly and GPU dispatch.
for field, offset_bytes in (('offset0', '1f00'), ('offset1', '001f')):
    source = SOURCE_G.replace('ds_write_b32 v4, v5 offset:16', 'ds_write_b32 v4, v5')
    source = source.replace('offset0:5 offset1:36', field + ':31')
    words = patch(WORDS_G, '100034d804050000', '000034d804050000')
    words = patch(words, '0524dcd806000006', offset_bytes + 'dcd806000006')
    CASES[f'G-{field}-only'] = (source, words, *CASES['G'][2:])


MBCNT_REFUSALS = (
    ('v_mbcnt_lo_u32_b32 v8, v0, 0', '080065d700010100', 'known mask'),
    ('v_mbcnt_hi_u32_b32 v8, v0, 0', '080066d700010100', 'known mask'),
    ('v_mbcnt_lo_u32_b32 v8, s4, 0', '080065d704000100', 'known mask'),
    ('v_mbcnt_lo_u32_b32 v8, exec_lo, 0', '080065d77e000100', 'known mask'),
    ('v_mbcnt_hi_u32_b32 v8, vcc_lo, 0', '080066d76a000100', 'known mask'),
    ('v_mbcnt_lo_u32_b32 v8, -1, exec_lo', '080065d7c1fc0000', 'VGPR or known base'),
    ('v_mbcnt_lo_u32_b32 v8, -1, s4', '080065d7c1080000', 'VGPR or known base'),
)


def check_mbcnt():
    for _, words, reason in MBCNT_REFUSALS:
        refused(reason, lambda: translate('D', code=bytes.fromhex(words + '000081bf')))
    for op in ('lo', 'hi'):
        # Keep the two refusals independent: a program starting with _lo would hide
        # an incorrectly accepted _hi when testing a group whose wave size is unknown.
        word = '0a0065d7c1000100' if op == 'lo' else '0b0066d7c1060200'
        single = bytes.fromhex(word + '000081bf')
        refused('at most 32 threads', lambda: translate('D', code=single, group=(3, 11, 1)))
        stage = translate('D', code=single, group=(3, 5, 2)).cs
        emit = getattr(stage, f'op_v_mbcnt_{op}_u32_b32')
        refused('three unmodified operands', lambda: emit(['v0', '-1', '0'], ['clamp']))
        refused('destination', lambda: emit(['v[0:1]', '-1', '0'], []))
        stage.kind = 'ps'
        refused(f'v_mbcnt_{op}_u32_b32', lambda: emit(['v0', '-1', '0'], []))
    assert 'uav 0 table 1 0' in translate('D').manifest()


def check_descriptor_scc():
    # Case A loads the input descriptor into s[12:15] and stores through s[16:19].
    # Put the condition into the stored value, or let the branch skip the move.
    compare = 's_cmp_lt_u32 s7, 4'
    mask = 's_and_b32 s13, s13, 0xc000ffff'
    select = 's_cselect_b32 s20, 1, 0'
    move = 'v_mov_b32 v5, s20'

    def pipeline(*lines):
        return translate('A', code=splice(*lines))

    def rejected(*lines):
        refused('scc from descriptor data', lambda: pipeline(*lines))

    for reader in (select, 's_cselect_b64 s[20:21], 1, 0', 's_cmov_b32 s20, 1', 's_cmov_b64 s[20:21], s[4:5]',
                   's_cmovk_i32 s20, 3', 's_cbranch_scc0 1', 's_cbranch_scc1 1'):
        rejected(compare, mask, reader, move)
        assert 'uav 0 table 1 16' in pipeline(mask, compare, reader, move).manifest()
    rejected(compare, 's_add_u32 s13, s13, 1', select, move)
    assert 's20 = scc ?' in pipeline(compare, 's_mov_b32 s13, s13', select, move).hlsl_cs()
    assert 's20 = scc ?' in pipeline(mask, 's_and_b32 s22, s7, 1', select, move).hlsl_cs()
    assert 's20 = scc ?' in pipeline(mask, 's_and_saveexec_b32 s0, vcc_lo', select, move).hlsl_cs()

    # A conditional edge can bypass the fresh comparison. A back edge can carry
    # the descriptor-derived flag to a read that comes first in program order.
    rejected(mask, 's_cbranch_vccz 1', compare, select, move)
    rejected(compare, 'v_mov_b32 v3, 0', select, mask, 'v_add_nc_u32 v3, 1, v3',
             'v_cmp_gt_u32 vcc_lo, 2, v3', 's_cbranch_vccnz 65530', move)
    assert 's20 = scc ?' in pipeline(compare, 's_branch 2', mask, select, move).hlsl_cs()


def check_lds():
    # Actual instruction substitutions in the LLVM-assembled exchange. Keep the prefix, except where the
    # local-ID proof itself is under test; a different first refusal would hide the regression.
    store = '100034d804050000'
    peer = '1000d8d808000006'
    broadcast = 'a400d8d808000007'
    wait = '7fc08cbf'
    barrier = '00008abf'
    nop = '000080bf'
    hlsl = translate('F').hlsl_cs()
    sync = 'GroupMemoryBarrierWithGroupSync();'
    assert hlsl.count(sync) == 1 and hlsl.index('] = v5;') < hlsl.index(sync) < hlsl.index('v6 = lds[')
    for old, new, reason in (
        ('82000834', '8002007e82000834', 'unchanged entry local X'),  # write v0 before computing the address
        ('82000834', '00cb087e82000834', 'unchanged entry local X'),  # v_swap also writes its second operand
        ('82000834', '82020834', 'one bounded dword'),              # use Y instead of X
        (store, '040134d804050000', 'one bounded dword'),             # last lane exceeds the 512-byte allocation
        (store, store + store, 'one bounded dword'),                  # overlapping fill
        (wait + barrier, nop + barrier, 'completed fill'),
        (barrier, '01008abf', 'one barrier'),                         # nonzero reserved immediate
        (barrier, nop, 'after the fill barrier'),
        (barrier, barrier + barrier, 'one barrier'),
        (peer, '0c00d8d808000006', 'initialized span'),               # read before the filled span
        (broadcast, '1001d8d808000007', 'initialized span'),          # first dword after the span
        (broadcast, '1200d8d808000007', 'aligned'),
        (broadcast, 'a400dad808000007', 'optional byte offset'),      # GDS
        ('ff101036fc000000', 'ff101036ff000000', 'aligned'),          # mask leaves unaligned byte addresses
        ('8002107e', '0403107e', 'literal or literal-masked'),        # unbounded VGPR address
        (peer + wait, peer + nop, 'completed LGKM wait'),
        (store, '8003febe' + store, 'entry EXEC'),
        (store, '020084bf' + store, 'straight-line'),
    ):
        refused(reason, lambda: translate('F', code=patch(WORDS_F, old, new)))
    base = USER7 | TGID_X
    for units in (0, 65):
        refused('allocation', lambda: translate('F', rsrc2=base | units << 15))
    refused('1D group', lambda: translate('F', group=(32, 2, 1)))
    # Slide the whole span to the end of the 512-byte allocation. Thread 63 now writes its last dword.
    at_end = patch(WORDS_F, store, '000134d804050000')
    at_end = patch(at_end, peer, '0001d8d808000006')
    at_end = patch(at_end, broadcast, '9401d8d808000007')
    assert 'lds[(v4 + 256u) >> 2] = v5;' in translate('F', code=at_end).hlsl_cs()
    assert 'groupshared uint lds[8192];' in translate('F', rsrc2=base | 64 << 15).hlsl_cs()

    pair = '0524dcd806000006'
    # The first result overwrites v6. Both loads must use the address saved before that write.
    body = translate('G').hlsl_cs()
    assert 'a = v6;\n    v6 = lds[(a + 20u) >> 2];\n    v7 = lds[(a + 144u) >> 2];' in body, body
    for other, reason in (
        ('0324dcd806000006', 'initialized span'),  # offset0 alone precedes the fill
        ('0525dcd806000006', 'initialized span'),  # offset1 alone crosses the upper end
        ('0524ded806000006', 'ordinary dword offsets'),  # GDS
        ('0524e0d806000006', 'does not cover'),  # stride-64 addressing
        ('0524dcd8060000ff', 'VGPR pair'),  # v[255:256]
        ('0505dcd806000006', 'distinct offsets'),
        ('0000dcd806000006', 'distinct offsets'),
    ):
        refused(reason, lambda: translate('G', code=patch(WORDS_G, pair, other)))
    # The binary decoder only emits unsigned 8-bit offsets and consecutive pairs. Exercise
    # the proof's operand checks directly for inputs the instruction cannot encode.
    decoded = list(G.decode(WORDS_G))
    index, = (i for i, ins in enumerate(decoded) if ins[2] == 'ds_read2_b32')
    prefix, assembly = decoded[index][:3], decoded[index][3]
    for old, new, reason in (('offset0:5', 'offset0:256', 'ordinary dword offsets'),
                             ('v[6:7]', 'v[6:8]', 'VGPR pair')):
        assert old in assembly
        decoded[index] = (*prefix, assembly.replace(old, new))
        refused(reason, lambda: G.check_lds_exchange(decoded, (64, 1, 1), 128))
    # Offset fields omitted by the disassembler default independently to zero. With the fill at zero,
    # both zero and +31 dwords are valid from any masked base in this fixture.
    for name, expected_offsets in (('G-offset0-only', (124, 0)), ('G-offset1-only', (0, 124))):
        shader = translate(name).hlsl_cs()
        for dest, offset in zip((6, 7), expected_offsets):
            assert f'v{dest} = lds[(a + {offset}u) >> 2];' in shader
    assert 'groupshared' not in translate('A', rsrc2=CASES['A'][2] | 1 << 15).hlsl_cs()


def check_controls():
    ff1 = bytes.fromhex('7e1385be')
    end = bytes.fromhex('000081bf')
    # A write to EXEC can disable lane zero; a backward branch can re-enter
    # the instruction with a different mask even at code offset zero.
    for code in (bytes.fromhex('8003febe') + ff1 + end,
                 ff1 + bytes.fromhex('feff82bf') + end):
        refused('compute entry EXEC mask', lambda: translate('A', code=code))
    refused('requires compute exec_lo', lambda: translate('A', code=bytes.fromhex('6a1385be') + end))
    assert 's5 = 0u;' in translate('A-ff1').hlsl_cs()
    assert expected('A-ff1') != expected('A')
    check_lds()
    check_descriptor_scc()
    check_mbcnt()
    a = translate('A')
    # This buffer-only compute program has no typed UAV declaration or image manifest entry.
    assert a.uavs and a.stored_images() == {}, (a.uavs, a.stored_images())
    hlsl, manifest = a.hlsl_cs(), a.manifest()
    assert not any(line.startswith('RWTexture2D') for line in hlsl.splitlines()), hlsl
    assert manifest.splitlines()[:2] == ['x2w-manifest 2', 'stage cs'], manifest
    assert [line for line in manifest.splitlines() if line.startswith('uav ')] == ['uav 0 table 1 16'], manifest
    for line in ('[numthreads(2, 3, 4)]', 'void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {',
                 '    s7 = gid.x;', '    s8 = gid.y;', '    s9 = gid.z;', '    v0 = tid.x;', '    v1 = tid.y;',
                 '    v2 = tid.z;', 'ByteAddressBuffer r0 : register(t0);', 'RWByteAddressBuffer u0 : register(u0);',
                 '    a = (v4 + 0) & 0xfffffffcu;', '    u0.Store(a, v5);'):
        assert line in hlsl.splitlines(), (line, hlsl)
    assert manifest == ('x2w-manifest 2\nstage cs\ngroup 2 3 4\nud 2\nudsrc 0 rootconst 0 0\nudsrc 1 rootconst 0 1\n'
                        'srv 0 buf table 1 0\nuav 0 table 1 16\n'), manifest
    b = translate('B')
    hlsl = b.hlsl_cs()
    for line in ('[numthreads(5, 2, 1)]', '    s7 = gid.x;', '    s8 = gid.z;', '    v0 = tid.x;', '    v1 = tid.y;',
                 '    a = (v4 + 8) & 0xfffffffcu;', '    u0.Store2(a, uint2(v5, v6));'):
        assert line in hlsl.splitlines(), (line, hlsl)
    assert 'gid.y' not in hlsl and 'tid.z' not in hlsl, hlsl
    # Noncontiguous and single axes take the next SGPR after the user SGPRs, with no gap for an omitted axis.
    for rsrc2, entry in ((USER7 | TGID_Y, ['s7 = gid.y;']), (USER7 | TGID_Z, ['s7 = gid.z;']),
                         (USER7 | TGID_Y | TGID_Z, ['s7 = gid.y;', 's8 = gid.z;']),
                         ((9 << 1) | TGID_X, ['s9 = gid.x;'])):
        stage = G.ComputePipeline(bytes.fromhex('000081bf'), root_blob(), rsrc2, (1, 1, 1)).cs
        assert stage.entry == entry + ['v0 = tid.x;'], (hex(rsrc2), stage.entry)
    # Entry refusals.
    end = bytes.fromhex('000081bf')
    refused('TG_SIZE_EN', lambda: G.ComputePipeline(end, root_blob(), USER7 | TG_SIZE_EN, (1, 1, 1)))
    refused('thread-ID component count', lambda: G.ComputePipeline(end, root_blob(), USER7 | (3 << 11), (1, 1, 1)))
    refused('gives 6 user SGPRs', lambda: G.ComputePipeline(end, root_blob(), 6 << 1, (1, 1, 1)))
    refused('thread group', lambda: G.ComputePipeline(end, root_blob(), USER7, (1024, 2, 1)))
    # A user SGPR outside the root's span has no known value: s0 (s_mov_b32 s10, s0) and s7 when user is 8.
    refused('numeric use of s0', lambda: translate('A', code=patch(WORDS_A, '06038abe', '00038abe')).hlsl_cs())
    refused('numeric use of s7', lambda: translate('A', rsrc2=(8 << 1) | tidig(3)).hlsl_cs())
    # Store forms outside this slice (the store is the dword at +0x60 of WORDS_A: offen, soffset 0x80, s[16:19]).
    store = '001070e004050480'
    for word, fragment in (('000070e004050480', 'offen address form'),     # off: no offen
                           ('002070e004050480', 'modifier idxen'),         # idxen, no offen
                           ('005070e004050480', 'modifier glc'),           # glc offen
                           ('001070e004050380', 'used for a load and a store')):  # store through the read descriptor
        refused(fragment, lambda w=word: translate('A', code=patch(WORDS_A, store, w)).hlsl_cs())
    # Graphics stages keep the unknown-op refusal text, so refusal census buckets do not move.
    for count, text in ((1, 'buffer_store_dword'), (4, 'buffer_store_dwordx4')):
        try:
            G.Stage.op_buffer_store_dword(type('P', (), {'kind': 'ps'})(), ['v5', 'v4', 's[16:19]', '0'], ['offen'], count)
        except G.Unsupported as reason:
            assert str(reason) == text, str(reason)
        else:
            raise AssertionError('graphics store accepted')
    # A register soffset is added to the address.
    hlsl = translate('A', code=patch(WORDS_A, store, '001070e004050414')).hlsl_cs()
    assert '    a = (v4 + s20 + 0) & 0xfffffffcu;' in hlsl.splitlines(), hlsl
    # An immediate offset that is not a multiple of 4 is refused, not rounded down.
    refused('unaligned immediate offset 6', lambda: translate(
        'A', code=patch(WORDS_A, store, SPLICE['buffer_store_dword v5, v4, s[16:19], 0 offen offset:6'])).hlsl_cs())
    # Past the range: today the translation carries no range check (num_records and stride are not read) and the
    # manifest no size, so an out-of-range store reaches the UAV unguarded and only a runtime that sizes the view
    # to the descriptor's range drops it. Pinned so a change of that policy has to update this control.
    far = translate('A', code=patch(WORDS_A, store, SPLICE['buffer_store_dword v5, v4, s[16:19], 0 offen offset:4092']))
    lines = far.hlsl_cs().splitlines()
    at = lines.index('    a = (v4 + 4092) & 0xfffffffcu;')
    assert lines[at + 1] == '    u0.Store(a, v5);', lines[at:at + 2]
    body = far.hlsl_cs().split('void main(', 1)[1]
    assert not any(word in body for word in ('GetDimensions', 'min(', 'if (a')), body
    assert far.manifest().splitlines()[-1] == 'uav 0 table 1 16', far.manifest()
    # Descriptor words: a word that is overwritten, moved or modified before the store refuses it, including a word
    # that is not the first of its s_load and a constant move. Each input below was accepted before.
    for lines in (['s_mov_b32 s18, 0'], ['s_or_b32 s17, s17, 0x40000'], ['s_and_b32 s18, s18, 0xffff'],
                  ['buffer_store_dword v5, v4, s[16:19], 0 offen', 's_mov_b32 s18, 4']):
        refused('modified descriptor', lambda lines=lines: translate('A', code=splice(*lines)).hlsl_cs())
    # Clearing the stride field of dword 1 (s_and with 0xc000ffff, either operand order) gives the raw descriptor a
    # byte-address store assumes: accepted, and case I runs it on the GPU. Any other mask, another dword, or the same
    # mask on an image store stays refused.
    for line in ('s_and_b32 s17, s17, 0xc000ffff', 's_and_b32 s17, 0xc000ffff, s17'):
        cleared = translate('A', code=splice(line)).hlsl_cs()
        assert '    u0.Store(a, v5);' in cleared.splitlines(), cleared
    for line in ('s_and_b32 s17, s17, 0xc000fff0', 's_and_b32 s18, s18, 0xc000ffff'):
        refused('buffer store through a modified descriptor', lambda line=line: translate('A', code=splice(line)).hlsl_cs())
    # The clear changes SCC (descriptor data in a scalar ALU op), so a read before a fresh compare is refused by the
    # stale-SCC rule; with a compare in between the read is emitted. A repeated modification stays refused.
    clear = 's_and_b32 s17, s17, 0xc000ffff'
    for reader in ('s_cselect_b32 s20, 1, 0', 's_cbranch_scc1 1'):
        refused('scc from descriptor data', lambda reader=reader: translate(
            'A', code=splice(clear, reader, 'v_mov_b32 v5, s20')).hlsl_cs())
    assert 's20 = scc ?' in translate('A', code=splice(clear, 's_cmp_lt_u32 s7, 4', 's_cselect_b32 s20, 1, 0',
                                                       'v_mov_b32 v5, s20')).hlsl_cs()
    for lines in ((clear, clear), (clear, 's_or_b32 s17, s17, 0x40000'), ('s_or_b32 s17, s17, 0x40000', clear)):
        refused('buffer store through a modified descriptor', lambda lines=lines: translate('A', code=splice(*lines)).hlsl_cs())
    # The modification is recorded for the descriptor the word belongs to (byte 16 of the x8 load), not for the
    # load's first descriptor: loads through the read descriptor at byte 0 are unaffected.
    refusals = []
    pipe = G.ComputePipeline(splice('s_and_b32 s18, s18, 0xffff'), root_blob(), CASES['A'][2], (2, 3, 4), refusals=refusals)
    assert pipe.modified == {(('table', 1), 16)}, pipe.modified
    assert [G.normalize_refusal(r) for r, _ in refusals] == ['buffer store through a modified descriptor'], refusals
    # Scalar instructions in a block s_cbranch_execz skips. The wave runs the block when any lane is active, so an
    # inactive lane that skips it can differ only by reading, after the join, scalar state the block defines.
    # Live after the join: the block's s20 is read by v_mov_b32 v5, s20, so the hardware stores gid.x + 1 for every
    # lane while a per-lane skip would store 0. Still refused, with exec from a plain move and in the saveexec form.
    enter = ['s_mov_b32 s21, exec_lo', 'v_cmp_gt_u32 vcc_lo, 1, v0', 's_mov_b32 exec_lo, vcc_lo']
    leave = ['s_mov_b32 exec_lo, s21']
    for lines in (enter + ['s_cbranch_execz 1', 's_add_u32 s20, s7, 1'] + leave + ['v_mov_b32 v5, s20'],
                  ['v_cmp_gt_u32 vcc_lo, 1, v0', 's_and_saveexec_b32 s0, vcc_lo', 's_cbranch_execz 1',
                   's_add_u32 s20, s7, 1', 's_or_b32 exec_lo, exec_lo, s0', 'v_mov_b32 v5, s20']):
        refused('s_add_u32 inside a block s_cbranch_execz skips per lane: definition of s20',
                lambda lines=lines: translate('A', code=splice(*lines)).hlsl_cs())
    # Dead after the join (s23 and SCC are not read again): accepted, and case J runs such a block on the GPU.
    translate('A', code=splice(*enter, 's_cbranch_execz 1', 's_add_u32 s23, s7, 1', *leave)).hlsl_cs()
    translate('J').hlsl_cs()
    # SCC set in the block and consumed after the join by s_cselect: refused.
    refused('definition of scc', lambda: translate('A', code=splice(
        *enter, 's_cbranch_execz 1', 's_add_u32 s23, s7, 1', *leave, 's_cselect_b32 s23, 1, 0')).hlsl_cs())
    # Exec writes inside the block. For a lane whose exec bit is clear on entry, the wave's writes must keep it clear,
    # so its exec at the join is what the model has. Exec set from a compare alone can set it: refused.
    refused('exec write', lambda: translate('A', code=splice(
        *enter, 's_cbranch_execz 3', 'v_cmp_gt_u32 vcc_lo, 3, v0', 's_mov_b32 exec_lo, vcc_lo', 's_cbranch_execz 0',
        *leave)).hlsl_cs())
    # A nested if (s_and_saveexec, restored with s_or of the mask saved in the block) keeps it clear: accepted, and
    # case H runs one on the GPU. Restoring with a mask saved before the block (s21) could set it: refused.
    translate('A', code=splice(*enter, 's_cbranch_execz 3', 'v_cmp_gt_u32 vcc_lo, 3, v0',
                               's_and_saveexec_b32 s0, vcc_lo', 's_or_b32 exec_lo, exec_lo, s0', *leave)).hlsl_cs()
    translate('H').hlsl_cs()
    refused('exec write', lambda: translate('A', code=splice(
        *enter, 's_cbranch_execz 3', 'v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_saveexec_b32 s0, vcc_lo',
        's_or_b32 exec_lo, exec_lo, s21', *leave)).hlsl_cs())
    # A backward s_cbranch_execz: refused.
    refused('s_cbranch_execz to a block before it', lambda: translate('A', code=splice(
        's_add_u32 s23, s7, 1', 's_cbranch_execz 65534')).hlsl_cs())
    # The same live-out block without the branch translates: the scalar write then runs for every lane.
    translate('A', code=splice(*enter, 's_add_u32 s20, s7, 1', *leave, 'v_mov_b32 v5, s20')).hlsl_cs()
    # Review inputs that were accepted. v_cmpx prints no destination: in `v_cmpx_gt_u32_e32 s23, v0` after the join,
    # s23 is read (the wave compares against gid.x + 1 in every lane), so the block's s23 is live: refused.
    prologue = '06038abe' '80038bbe' '050308f4000000fa' '7fc08cbf'  # s[12:15] = the table's first descriptor
    for cmpx in ('1700a87d', '0000d4d417000200'):  # v_cmpx_gt_u32 s23, v0 in the e32 and e64 forms
        program = bytes.fromhex(prologue + '800397be' '7e0395be' '8300887d' '6a03febe' '010088bf' '07811780' '1503febe'
                                + cmpx + '82000834' '001070e004000380' '000081bf')
        refused('s_add_u32 inside a block s_cbranch_execz skips per lane: definition of s23',
                lambda program=program: translate('A', code=program).hlsl_cs())
    # A branch inside the block: on the path that skips `s_mov_b32 s21, exec_lo` the restore reads the mask saved
    # before the block, which can set inactive lanes. Exec writes are accepted in straight-line blocks only: refused.
    program = bytes.fromhex(prologue + '82000834' '7e0395be' '8300887d' '6a03febe' '040088bf' '048007bf' '010085bf'
                            '7e0395be' '1503febe' '001070e004000380' '000081bf')
    refused('s_mov_b32 inside a block s_cbranch_execz skips per lane: exec write',
            lambda: translate('A', code=program).hlsl_cs())
    # saved tracks register names, not which exec half was copied: two 32-bit copies restored through a 64-bit move
    # would set exec_hi from exec_lo. Refused; the 32-bit restore above stays accepted.
    SPLICE.update({'s_mov_b32 s22, exec_lo': '7e0396be', 's_mov_b32 s23, exec_lo': '7e0397be',
                   's_mov_b64 exec, s[22:23]': '1604febe'})
    refused('s_mov_b64 inside a block s_cbranch_execz skips per lane: exec write', lambda: translate('A', code=splice(
        *enter, 's_cbranch_execz 3', 's_mov_b32 s22, exec_lo', 's_mov_b32 s23, exec_lo', 's_mov_b64 exec, s[22:23]',
        *leave)).hlsl_cs())
    # Refusals of the block scan with short programs: a branch leaving the block, a vcc definition that a vcc branch
    # reads after the join, and a proven s_setpc_b64 (GETPC, ADD, ADDC, SETPC) inside the block.
    SPLICE['s_branch 2'] = '020082bf'
    refused('s_branch inside a block s_cbranch_execz skips per lane: branch out of the block', lambda: translate(
        'A', code=splice(*enter, 's_cbranch_execz 1', 's_branch 2', *leave)).hlsl_cs())
    refused('v_cmp_gt_u32_e32 inside a block s_cbranch_execz skips per lane: definition of vcc', lambda: translate(
        'A', code=splice(*enter, 's_cbranch_execz 1', 'v_cmp_gt_u32 vcc_lo, 1, v0', *leave,
                         's_cbranch_vccz 1', 'v_mov_b32 v5, s20')).hlsl_cs())
    SPLICE.update({'s_cbranch_execz 4': '040088bf', 's_getpc_b64 s[22:23]': '001f96be', 's_add_u32 s22, s22, 12': '168c1680',
                   's_addc_u32 s23, s23, 0': '17801782', 's_setpc_b64 s[22:23]': '162080be'})
    refused('s_setpc_b64 inside a block s_cbranch_execz skips per lane: effect outside scalar registers', lambda: translate(
        'A', code=splice(*enter, 's_cbranch_execz 4', 's_getpc_b64 s[22:23]', 's_add_u32 s22, s22, 12',
                         's_addc_u32 s23, s23, 0', 's_setpc_b64 s[22:23]', *leave)).hlsl_cs())
    # A proven calculated jump into the block bypasses the copy like a direct branch does: the restore at the target
    # reads a mask the block did not save. GETPC / ADD 20 / ADDC / SETPC lands on the restore; both are refused.
    SPLICE.update({'s_add_u32 s22, s22, 20': '16941680', 's_cbranch_execz 2': '020088bf'})
    block = ['s_cbranch_execz 2', 's_mov_b32 s21, exec_lo', *leave]
    refused('s_mov_b32 inside a block s_cbranch_execz skips per lane: exec write', lambda: translate(
        'A', code=splice('s_branch 2', *block)).hlsl_cs())
    refused('s_mov_b32 inside a block s_cbranch_execz skips per lane: exec write', lambda: translate(
        'A', code=splice('s_getpc_b64 s[22:23]', 's_add_u32 s22, s22, 20', 's_addc_u32 s23, s23, 0',
                         's_setpc_b64 s[22:23]', *block)).hlsl_cs())
    # An implicit-LOD sample has no derivatives in a compute lane; the level-zero form translates.
    load = ('s_mov_b64 s[20:21], 0', 's_mov_b64 s[22:23], 0', 's_load_dwordx8 s[24:31], s[10:11], 0x40')
    sample = 'image_sample v[12:15], v[0:1], s[24:31], s[20:23] dmask:0xf dim:SQ_RSRC_IMG_2D'
    refused('image_sample with implicit LOD', lambda: translate('A', code=splice(*load, sample)).hlsl_cs())
    pipeline = translate('A', code=splice(*load, sample.replace('image_sample', 'image_sample_lz')))
    lz = pipeline.hlsl_cs()
    assert '.SampleLevel(' in lz, lz
    refused('compute d16 image load requires validated SRV formats',
            lambda: pipeline.cs.image(['v12', 'v[0:1]', 's[24:31]'],
                                      ['dim:SQ_RSRC_IMG_2D', 'dmask:0x3', 'd16'], 'load'))
    print('compute entry, numthreads, masked bit counts, manifest, offen store forms and the store, descriptor, execz and sample '
          'refusals pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    for name, (source, words, *_rest) in CASES.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for line, words in SPLICE.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == words, '%s: assembled %s, recorded %s' % (line, built, words)
    for line, words, _ in MBCNT_REFUSALS:
        assert assemble(llvm_mc, line + '\n').hex() == words, line
    print('fixture words match llvm-mc')


# control -> (case, text the translation must contain once, replacement)
MBCNT_LANE = '(tid.x + 4u * (tid.y + 2u * tid.z))'
MBCNT_COUNT = f'countbits(0x85241249u & ((1u << {MBCNT_LANE}) - 1u))'
CONTROLS = {
    'swapped-axes': ('A', '    s7 = gid.x;\n    s8 = gid.y;\n', '    s7 = gid.y;\n    s8 = gid.x;\n'),
    'dropped-store': ('A', '    u0.Store(a, v5);\n', ''),
    'b-offset': ('B', '    a = (v4 + 8) & 0xfffffffcu;\n', '    a = (v4 + 0) & 0xfffffffcu;\n'),
    'c-unpredicated': ('C', 'if (exec & 1) u0.Store3(', 'u0.Store3('),
    'd-mask': ('D', MBCNT_COUNT, MBCNT_LANE),
    'd-whole-mask': ('D', MBCNT_COUNT, 'countbits(0x85241249u)'),
    'd-x-only': ('D', MBCNT_COUNT, MBCNT_COUNT.replace(MBCNT_LANE, 'tid.x')),
    'd-base': ('D', MBCNT_COUNT + ' + v6', MBCNT_COUNT),
    'd-hi': ('D', 'v9 = v8;', f'v9 = v8 + {MBCNT_COUNT};'),
    'd-exec': ('D', 'if (exec & 1) v10 =', 'v10 ='),
    'f-peer': ('F', '0x00000080u ^ v4', 'v4'),
    'f-broadcast': ('F', 'lds[(v8 + 164u) >> 2]', 'lds[(v8 + 160u) >> 2]'),
    'j-scalar-dropped': ('J', '        s22 = s7 * ud1;\n', ''),
    'h-inner-unpredicated': ('H', 'if (exec & 1) u0.Store(a, v5);', 'u0.Store(a, v5);'),
    'k-block-entered': ('K', 'pc = ((exec & 1) == 0) ? 2 : 1;', 'pc = 1; exec = 1;'),  # skipped waves run it
    'g-swapped-offsets': ('G', 'v6 = lds[(a + 20u) >> 2];\n    v7 = lds[(a + 144u) >> 2];',
                          'v6 = lds[(a + 144u) >> 2];\n    v7 = lds[(a + 20u) >> 2];'),
    'g-overlap': ('G', 'v7 = lds[(a + 144u)', 'v7 = lds[(v6 + 144u)'),
}

# First bad word for the masked bit-count and paired LDS read controls.
CONTROL_MISMATCH = {
    'd-mask': 'word=8 expected=00000008 actual=00000009',
    'd-whole-mask': 'word=0 expected=00000007 actual=00000011',
    'd-x-only': 'word=16 expected=00000009 actual=00000007',
    'd-base': 'word=0 expected=00000007 actual=00000000',
    'd-hi': 'word=5 expected=00000008 actual=00000009',
    'd-exec': 'word=2 expected=ffffffff actual=00000000',
    'g-swapped-offsets': 'word=0 expected=00000004 actual=00000000',
    'g-overlap': 'word=1 expected=00000000 actual=00000004',
}


def mutate(hlsl, control):
    _, old, new = CONTROLS[control]
    assert hlsl.count(old) == 1, (control, hlsl)
    return hlsl.replace(old, new)


def run_case(name, dxc, runner, adapter, directory, control=None):
    hlsl = translate(name).hlsl_cs()
    if control:
        hlsl = mutate(hlsl, control)
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    groups = CASES[name][4]
    constants = [KEY, STRIDE] + [0] * 14
    case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                     + struct.pack('<%dI' % WORDS, *expected(name)) + struct.pack('<16I', *constants))
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        if control == 'mbcnt':
            runs = [('D', mutation) for mutation, (case, _, _) in CONTROLS.items() if case == 'D']
        elif control:
            runs = [(CONTROLS[control][0], control)]
        else:
            runs = [(name, None) for name in CASES]
        for name, mutation in runs:
            run = run_case(name, dxc, runner, adapter, Path(tmp), mutation)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = []
            for line in run.stderr.splitlines():
                if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
                    print(line, file=sys.stderr)
                else:
                    diagnostics.append(line)
            if mutation:
                first = diagnostics[:1]
                if (run.returncode != 1 or not any('expected=' in line for line in diagnostics) or
                        mutation in CONTROL_MISMATCH and first != [CONTROL_MISMATCH[mutation]]):
                    raise AssertionError('control %s: unexpected result (exit %d) %s' % (mutation, run.returncode, run.stderr))
                print('control %s: rejected at %s' % (mutation, first[0]))
            elif run.returncode or diagnostics:
                raise AssertionError('case %s: %s' % (name, run.stderr or 'exit %d' % run.returncode))
            else:
                print('case %s: all %d words match' % (name, WORDS))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit(__doc__)
