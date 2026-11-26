"""Compute stage of tools/startup/gfx2hlsl.py: entry registers, numthreads and the offen buffer store (#1412).

  (no arguments)                                  translation controls, pure Python
  binary <llvm-mc>                                reassemble the fixture sources and compare them with the words below
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                                  translate, compile with DXC cs_6_0, dispatch and compare every word
  control <swapped-axes|dropped-store|b-offset|c-unpredicated|j-scalar-dropped|h-inner-unpredicated|m-else-unswitched|n-unmasked|o-mask-kept|s-min-strict|s-borrow-flipped> <dxc> <offscreen_check> <warp|hardware>
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
- case M is case H with an if/else in the skipped block: each arm has its own s_cbranch_execz, s_andn2_b32 of the
  saved mask switches to the else arm and s_mov_b32 restores it. Lanes tid.x < 3 store the first word, lanes 3-5
  the second, the rest none.
- case N skips a block that only sets vcc from a compare; the join ANDs vcc with exec before it restores exec, and
  the result selects the storing lanes. Lanes tid.x < 3 store one word; lanes 6 and 7 skip the block with vcc set.
- case O removes the lanes tid.x < 3 from the saved mask with s_andn2_b32 s0, s0, exec_lo inside the skipped block,
  and the join moves s0 into exec. Lanes 3-7 store one word; lanes 6 and 7 skip the block and keep their bit.
- case G reads two independently offset LDS dwords with the address in the first destination VGPR.
- cases L-uint and L-unorm use the same formatted load with native descriptor stride and word 3 supplied by the
  harness. The descriptor selects unaligned UINT16 pairs or an UNORM8 field and permutes the outputs.
- cases O-dword and O-shift store in the `off` form (no VGPR address), so every lane of a group writes the same
  address. O-dword: 8 lanes per group store key ^ group ID at byte 4 * group + 8 (SGPR soffset plus immediate).
  O-shift: 32 lanes per group store (group ID, the (key, stride) pair shifted right by 4, 0) as four words at
  byte 16 * group, the shape of a 64-bit shift of a scalar pair.

The dispatch runs on a plain D3D12 device (the stand-in has no slot-67 path yet). The root signature is the runner's:
root SRV t0 (read buffer), root UAV u0 (writable buffer) and 16 root constants at b0, which match the translation's
r0, u0 and entry cbuffer. The manifest's table lines and the descriptor fetch are therefore not exercised on the GPU.
All 320 words are compared, so a word outside the written records must keep the sentinel. The controls swap the X
and Y group IDs at entry, drop the store, drop case B's immediate offset, drop case C's exec predicate, or drop a
scalar statement inside case J's block; each must change the readback.
"""
import itertools
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

# Case P is the compiler's i1 phi with a per-lane old value (a blend). M (s23) holds lanes tid.x < 4 before the branch; the
# block, entered by lanes tid.x < 3 * group, clears M for the lanes it runs (s_andn2 with exec) and ORs in the odd lanes
# (a compare masked with exec); the lane select after the join reads M per lane. Group 0 skips the block with the whole
# wave, groups 1 and 2 enter with a partial wave (3 and 6 lanes) and group 3 with all of it. Case Q is the same with M set
# to all ones before the branch and `s_orn2 M, C, exec_lo` in the block.
SOURCE_P = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
s_mul_i32 s24, s7, 3
v_cmp_gt_u32 s23, 4, v0
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, s24, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz 5
s_andn2_b32 s23, s23, exec_lo
v_and_b32 v6, 1, v0
v_cmp_ne_u32 vcc_lo, 0, v6
s_and_b32 s22, vcc_lo, exec_lo
s_or_b32 s23, s23, s22
s_mov_b32 exec_lo, s21
v_not_b32 v6, v5
v_cndmask_b32 v7, v5, v6, s23
buffer_store_dword v7, v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_P = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '82060834' '001030e004050380' '703f8cbf'
    '07831893' '1700c4d484000200' '7e0395be' '1800887d' '6a03febe' '050088bf' '177e178a' '81000c36' '800c8a7d'
    '6a7e1687' '17161788' '1503febe' '056f0c7e' '070001d5050d5e00' '001070e004070480' '000081bf')
SOURCE_Q = (SOURCE_P.replace('v_cmp_gt_u32 s23, 4, v0\n', 's_mov_b32 s23, -1\n').replace('s_cbranch_execz 5', 's_cbranch_execz 3')
            .replace('s_andn2_b32 s23, s23, exec_lo\n', '')
            .replace('s_and_b32 s22, vcc_lo, exec_lo\ns_or_b32 s23, s23, s22\n', 's_orn2_b32 s23, vcc_lo, exec_lo\n'))
WORDS_Q = (WORDS_P.replace(bytes.fromhex('1700c4d484000200'), bytes.fromhex('c10397be'))
           .replace(bytes.fromhex('050088bf'), bytes.fromhex('030088bf')).replace(bytes.fromhex('177e178a'), b'')
           .replace(bytes.fromhex('6a7e168717161788'), bytes.fromhex('6a7e178b')))

# Case M is case H with the nested if as an if/else. Each arm has its own s_cbranch_execz; the then arm stores the
# first word, s_andn2_b32 of the saved mask switches to the else arm, which stores the second, and s_mov_b32
# restores the saved mask.
SOURCE_M = """\
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
s_cbranch_execz 14
s_mul_i32 s22, s7, s5
s_add_u32 s22, s22, s4
v_xor_b32 v5, s22, v3
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_saveexec_b32 s0, vcc_lo
s_cbranch_execz 2
buffer_store_dword v5, v4, s[16:19], 0 offen
s_andn2_b32 exec_lo, s0, exec_lo
s_cbranch_execz 3
v_not_b32 v6, v5
buffer_store_dword v6, v4, s[16:19], 0 offen offset:4
s_mov_b32 exec_lo, s0
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_M = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '83060834' '7e0395be' '8600887d'
    '6a03febe' '0e0088bf' '07051693' '16041680' '16060a3a' '8300887d' '6a3c80be' '020088bf' '001070e004050480'
    '007e7e8a' '030088bf' '056f0c7e' '041070e004060480' '0003febe' '1503febe' '000081bf')

# Case NM skips a block that only sets vcc from a compare (lanes tid.x < 3 of the lanes tid.x < 6 that enter it);
# the join masks vcc with exec (s_and_b32 s23, vcc_lo, exec_lo) before it restores exec, and s23 later selects
# the lanes that store. vcc is all ones before the branch, so lanes 6 and 7, which skip the block, hold a stale
# vcc that only the exec mask removes.
SOURCE_NM = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 6, v0
s_and_saveexec_b32 s0, vcc_lo
s_mov_b32 vcc_lo, -1
s_cbranch_execz 1
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_b32 s23, vcc_lo, exec_lo
s_or_b32 exec_lo, exec_lo, s0
v_xor_b32 v5, s4, v3
s_mov_b32 exec_lo, s23
buffer_store_dword v5, v4, s[16:19], 0 offen
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_NM = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '82060834' '7e0395be' '8600887d'
    '6a3c80be' 'c103eabe' '010088bf' '8300887d' '6a7e1787' '7e007e88' '04060a3a' '1703febe' '001070e004050480'
    '1503febe' '000081bf')

# Case O enters the block with the lanes tid.x < 6 (s_and_saveexec_b32 s0 saves all eight), narrows exec to the
# lanes tid.x < 3 and removes them from the saved mask with s_andn2_b32 s0, s0, exec_lo inside the block; the join
# moves s0 into exec. Lanes 3-5 lose their bit nowhere and lanes 6 and 7 skip the block with theirs, so lanes 3-7
# store one word.
SOURCE_O = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s7, 3
v_add_nc_u32 v3, s20, v0
v_lshlrev_b32 v4, 2, v3
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 6, v0
s_and_saveexec_b32 s0, vcc_lo
s_cbranch_execz 3
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_b32 exec_lo, exec_lo, vcc_lo
s_andn2_b32 s0, s0, exec_lo
s_mov_b32 exec_lo, s0
v_xor_b32 v5, s4, v3
buffer_store_dword v5, v4, s[16:19], 0 offen
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_O = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '0783148f' '1400064a' '82060834' '7e0395be' '8600887d'
    '6a3c80be' '030088bf' '8300887d' '7e6a7e87' '007e008a' '0003febe' '04060a3a' '001070e004050480' '1503febe'
    '000081bf')

# The two-mask loop from #2645. The displacements name the join, exit and head, respectively.
LOOP_BREAK_SOURCE = """\
v_lshlrev_b32 v4, 2, v0
v_mov_b32 v5, 9
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 3, v0
s_and_b32 s22, exec_lo, vcc_lo
s_mov_b32 s23, s22
s_mov_b32 s24, 1
s_mov_b32 exec_lo, s22
s_cbranch_execz 6
v_cmpx_gt_u32 s24, v0
s_andn2_b32 s22, s22, exec_lo
s_andn2_b32 s23, s23, exec_lo
s_cbranch_scc0 6
s_mov_b32 exec_lo, s22
v_mov_b32 v5, s24
s_mov_b32 exec_lo, s23
s_add_u32 s24, s24, 1
s_cmp_ge_u32 s24, 4
s_cbranch_scc0 65524
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, s21
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
LOOP_BREAK_WORDS = bytes.fromhex(
    '82000834 89020a7e 7e0395be 8300887d 7e6a1687 160397be 810398be 1603febe '
    '060088bf 1800a87d 167e168a 177e178a 060084bf 1603febe 18020a7e 1703febe '
    '18811880 188409bf f4ff84bf 06038abe 80038bbe 05030cf4000000fa 7fc08cbf '
    '1503febe 001070e004050480 000081bf')


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

# Case F-two-fills keeps the first bank and writes each lane's byte address to a second bank.
# Broadcast lane 37 from the second bank while the peer read still addresses the first.
SOURCE_F2 = SOURCE_F.replace('ds_write_b32 v4, v5 offset:16',
                             'ds_write_b32 v4, v5 offset:16\nds_write_b32 v4, v4 offset:320').replace(
                                 'v8 offset:164', 'v8 offset:468')
WORDS_F2 = WORDS_F.replace(bytes.fromhex('100034d804050000'),
                           bytes.fromhex('100034d804050000400134d804040000')).replace(
                               bytes.fromhex('a400d8d808000007'), bytes.fromhex('d401d8d808000007'))

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

SOURCE_L = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, 1
buffer_load_format_xyzw v[5:8], v[0:1], s[12:15], 0 idxen offen offset:2
s_waitcnt vmcnt(0)
v_lshlrev_b32 v4, 4, v0
buffer_store_dwordx4 v[5:8], v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_L = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '8102027e' '02300ce000050380'
    '703f8cbf' '84000834' '001078e004050480' '000081bf')
LOAD_L = bytes.fromhex('02300ce000050380')
L_STRIDE = 6
L_WORD3 = {
    'L-uint': (27 << 12) | 5 | (4 << 3) | (1 << 6),  # Y, X, one, zero
    'L-unorm': (1 << 12) | 1 | (4 << 3) | (7 << 6) | (5 << 9),  # one, X, W, Y
}

# Case N: v_readfirstlane_b32 of a buffer load whose address is the same for every lane (offset 0, no VGPR), xored
# into each lane's own read-buffer word. Lane i stores word0 ^ word[i]. SOURCE_N_VARYING loads the readfirstlane
# source at a per-lane offset (v0), which is not provably uniform and must be refused.
SOURCE_N = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_dword v1, off, s[12:15], 0
s_waitcnt vmcnt(0)
v_readfirstlane_b32 s20, v1
v_lshlrev_b32 v4, 2, v0
buffer_load_dword v2, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
v_xor_b32 v3, s20, v2
buffer_store_dword v3, v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_N = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '000030e000010380'
    '703f8cbf' '0105287e' '82000834' '001030e004020380'
    '703f8cbf' '1404063a' '001070e004030480' '000081bf')
N_LOAD = bytes.fromhex('000030e000010380')
N_LOAD_VARYING = bytes.fromhex('001030e000010380')
WORDS_N_VARYING = WORDS_N.replace(N_LOAD, N_LOAD_VARYING, 1)

# Each group uses its ID as a low addend, overwrites that SGPR, and stores the
# low word, high word and SCC left by the high add. The fourth word stays untouched.
SOURCE_CARRY = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0
s_waitcnt lgkmcnt(0)
s_mul_i32 s20, s7, 1
s_add_u32 s20, s20, -2
s_add_u32 s21, 0, src_scc
s_cselect_b32 s22, 1, 0
s_lshl_b32 s23, s7, 4
v_mov_b32_e32 v4, s23
v_mov_b32_e32 v8, s20
v_mov_b32_e32 v9, s21
v_mov_b32_e32 v10, s22
buffer_store_dwordx3 v[8:10], v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_CARRY = bytes.fromhex(
    '06038abe 80038bbe 05030cf4000000fa 7fc08cbf 07811493 14c21480 80fd1580 81801685'
    ' 0784178f 1702087e 1402107e 1502127e 1602147e 00107ce004080480 000081bf')


SCC_EDGES = (0, 0xffffffff, 0x7fffffff, 0x80000000)
SCC_OPS = ('s_add_u32', 's_sub_u32', 's_add_i32', 's_sub_i32', 's_min_u32', 's_max_u32', 's_min_i32', 's_max_i32')


def scc_fixture():
    """Group (x, y) of the 4 x 4 grid takes S0 = edge[x], S1 = edge[y] (0, -1, INT_MAX, INT_MIN); the runner accepts at most 5 groups per axis. Each s_add/s_sub/s_min/s_max forms is run
    twice (the destination apart from S0, then the destination that is S0) and its SCC read right after; word 0 of the
    group holds the 16 SCC bits, word 1 a hash of the 16 results. RDNA 2 ISA: s_add_u32 carry out, s_sub_u32 borrow,
    s_add_i32 / s_sub_i32 signed overflow, s_min / s_max SCC = (D == S0)."""
    lines = ['s_mov_b32 s10, s6', 's_mov_b32 s11, 0', 's_load_dwordx8 s[12:19], s[10:11], 0x0', 's_waitcnt lgkmcnt(0)',
             's_lshl_b32 s24, s8, 2', 's_add_u32 s24, s24, s7', 'v_lshlrev_b32 v4, 3, s24', 's_mov_b32 s22, s7',
             's_mov_b32 s23, s8']
    for reg, index in (('s20', 's22'), ('s21', 's23')):
        lines.append('s_mov_b32 %s, 0' % reg)
        for n, value in ((1, '-1'), (2, '0x7fffffff'), (3, '0x80000000')):
            lines += ['s_cmp_eq_u32 %s, %d' % (index, n), 's_cselect_b32 %s, %s, %s' % (reg, value, reg)]
    lines += ['s_mov_b32 s27, 0', 's_mov_b32 s28, 0']
    bit = 0
    for op in SCC_OPS:
        for dst, init in (('s24', None), ('s26', 's_mov_b32 s26, s20')):
            lines += ([init] if init else []) + [
                '%s %s, %s, s21' % (op, dst, dst if init else 's20'), 's_cselect_b32 s25, 1, 0',
                's_lshl_b32 s25, s25, %d' % bit, 's_or_b32 s27, s27, s25', 's_xor_b32 s28, s28, %s' % dst,
                's_lshl_b32 s28, s28, 1']
            bit += 1
    lines += ['v_mov_b32 v5, s27', 'buffer_store_dword v5, v4, s[16:19], 0 offen', 'v_add_nc_u32 v4, 4, v4',
              'v_mov_b32 v5, s28', 'buffer_store_dword v5, v4, s[16:19], 0 offen', 's_endpgm']
    return '\n'.join(lines) + '\n'


def scc_reference(a, b):
    """(SCC bits, result hash) of the fixture for S0 = a, S1 = b, from the ISA rules (no code shared with the translator)."""
    def signed(x):
        return x - (1 << 32) if x >> 31 else x
    bits = hashed = 0
    for n, op in enumerate(SCC_OPS):
        for alias in (0, 1):
            x = a  # the aliased run starts from a copy of S0, which is S0 again
            if op.endswith('add_u32'):
                total = x + b
                result, scc = total & MASK, total > MASK
            elif op == 's_sub_u32':
                result, scc = (x - b) & MASK, x < b
            elif op == 's_add_i32':
                result = (x + b) & MASK
                scc = not -(1 << 31) <= signed(x) + signed(b) < (1 << 31)
            elif op == 's_sub_i32':
                result = (x - b) & MASK
                scc = not -(1 << 31) <= signed(x) - signed(b) < (1 << 31)
            else:
                view = signed if op.endswith('i32') else (lambda v: v)
                pick = min if 'min' in op else max
                result = pick(x, b, key=view)
                scc = result == x
            bits |= int(scc) << (2 * n + alias)
            hashed = ((hashed ^ result) << 1) & MASK
    return bits, hashed

WORDS_S = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf0882188f1807188004001ad583300000070396be080397be800394be'
    '168106bfc1141485168206bfff141485ffffff7f168306bfff14148500000080800395be178106bfc1151585178206bf'
    'ff151585ffffff7f178306bfff1515850000008080039bbe80039cbe14151880818019851980198f1b191b881c181c89'
    '1c811c8f14039abe1a151a80818019851981198f1b191b881c1a1c891c811c8f14159880818019851982198f1b191b88'
    '1c181c891c811c8f14039abe1a159a80818019851983198f1b191b881c1a1c891c811c8f14151881818019851984198f'
    '1b191b881c181c891c811c8f14039abe1a151a81818019851985198f1b191b881c1a1c891c811c8f1415988181801985'
    '1986198f1b191b881c181c891c811c8f14039abe1a159a81818019851987198f1b191b881c1a1c891c811c8f14159883'
    '818019851988198f1b191b881c181c891c811c8f14039abe1a159a83818019851989198f1b191b881c1a1c891c811c8f'
    '1415988481801985198a198f1b191b881c181c891c811c8f14039abe1a159a8481801985198b198f1b191b881c1a1c89'
    '1c811c8f1415188381801985198c198f1b191b881c181c891c811c8f14039abe1a151a8381801985198d198f1b191b88'
    '1c1a1c891c811c8f1415188481801985198e198f1b191b881c181c891c811c8f14039abe1a151a8481801985198f198f'
    '1b191b881c1a1c891c811c8f1b020a7e001070e0040504808408084a1c020a7e001070e004050480000081bf'
)

CASES = {
    # name: (source, words, rsrc2, group size, group counts)
    'A': (SOURCE_A, WORDS_A, USER7 | TGID_X | TGID_Y | TGID_Z | tidig(3), (2, 3, 4), (3, 2, 2)),
    'address-carry': (SOURCE_CARRY, WORDS_CARRY, USER7 | TGID_X, (1, 1, 1), (4, 1, 1)),
    'B': (SOURCE_B, WORDS_B, USER7 | TGID_X | TGID_Z | tidig(2), (5, 2, 1), (4, 1, 3)),
    'C': (SOURCE_C, WORDS_C, USER7 | TGID_Y | tidig(1), (8, 1, 1), (1, 5, 1)),
    'D': (SOURCE_D, WORDS_D, USER7 | TGID_X | tidig(3), (4, 2, 4), (2, 1, 1)),
    'D-wrap': (SOURCE_D_WRAP, WORDS_D_WRAP, USER7 | TGID_X | tidig(3), (4, 2, 4), (2, 1, 1)),
    'E': (*saveexec_fixture(), USER7 | TGID_X | tidig(1), (1, 1, 1), (4, 1, 1)),
    'F': (SOURCE_F, WORDS_F, USER7 | TGID_X | (1 << 15), (64, 1, 1), (2, 1, 1)),
    'F-two-fills': (SOURCE_F2, WORDS_F2, USER7 | TGID_X | (2 << 15), (64, 1, 1), (2, 1, 1)),
    'J': (SOURCE_J, WORDS_J, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'H': (SOURCE_H, WORDS_H, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'M': (SOURCE_M, WORDS_M, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'NM': (SOURCE_NM, WORDS_NM, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'O': (SOURCE_O, WORDS_O, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'K': (SOURCE_K, WORDS_K, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'P': (SOURCE_P, WORDS_P, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'Q': (SOURCE_Q, WORDS_Q, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1)),
    'G': (SOURCE_G, WORDS_G, USER7 | TGID_X | (1 << 15), (64, 1, 1), (2, 1, 1)),
    **{name: (SOURCE_L, WORDS_L, USER7, (16, 1, 1), (1, 1, 1)) for name in L_WORD3},
    'N': (SOURCE_N, WORDS_N, USER7 | tidig(1), (16, 1, 1), (1, 1, 1)),
    'S': (scc_fixture(), WORDS_S, USER7 | TGID_X | TGID_Y | tidig(1), (1, 1, 1), (4, 4, 1)),
}

# Single instructions the controls splice into case A before its store, recorded from llvm-mc 15.
STORE_A = WORDS_A.index(bytes.fromhex('001070e004050480'))
SCC_LANE_SELECT = 'v_cndmask_b32_e64 v5, 0, 1, s20'
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
    's_cbranch_scc0 2': '020084bf',
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
    SCC_LANE_SELECT: '050001d580025100',
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
    's_bfe_u32 s21, s17, 0xe0010': '11ff959310000e00',
    's_bfe_u32 s21, s29, 0xe0010': '1dff959310000e00',
    's_max_u32 s21, s21, 1': '15819584',
    's_max_u32 s21, s21, 2': '15829584',
    'v_cmp_gt_u32_e64 s21, 3, v0': '1500c4d483000200',
    's_bfe_u32 s21, s17, 0xe0011': '11ff959311000e00',
    's_mul_i32 s18, s18, s21': '12151293',
    's_mul_i32 s18, s18, 4': '12841293',
    's_and_b32 s19, s19, 0xcfffffff': '13ff1387ffffffcf',
    's_or_b32 s19, s19, 0x30000000': '13ff138800000030',
    's_or_b32 s19, s19, 0x20000000': '13ff138800000020',
    's_load_dwordx8 s[24:31], s[10:11], 0x0': '05060cf4000000fa',
    's_load_dwordx8 s[24:31], s[10:11], 0x20': '05060cf4200000fa',
    's_add_u32 s10, s10, 32': '0aa00a80',
    's_mov_b32 s11, 1': '81038bbe',
    's_waitcnt lgkmcnt(0)': '7fc08cbf',
    's_mov_b32 s19, s31': '1f0393be',
    's_mov_b32 s17, s29': '1d0391be',
    's_cbranch_scc1 2': '020085bf',
    'buffer_store_dword v5, off, s[16:19], 0': '000070e000050480',
    'buffer_store_dword v5, off, s[16:19], s20 offset:8': '080070e000050414',
    'buffer_store_dwordx2 v[5:6], off, s[16:19], 0': '000074e000050480',
    'v_lshrrev_b64 v[5:6], 4, s[20:21]': '050000d784280000',
    'v_lshrrev_b64 v[5:6], 4, v[3:4]': '050000d784060200',
    's_cbranch_execz 4': '040088bf',
}


def splice(*lines, at=STORE_A):
    return WORDS_A[:at] + b''.join(bytes.fromhex(SPLICE[line]) for line in lines) + WORDS_A[at:]


def replace_store(*lines):
    """Case A with its store replaced by `lines` (the last one is the new store)."""
    return WORDS_A[:STORE_A] + b''.join(bytes.fromhex(SPLICE[line]) for line in lines) + WORDS_A[STORE_A + 8:]


# Case I is case A with the store descriptor's stride field cleared before the store (expected words as case A).
STRIDE_CLEAR = 's_and_b32 s17, s17, 0xc000ffff'
CASES['I'] = (SOURCE_A.replace('buffer_store_dword', STRIDE_CLEAR + '\nbuffer_store_dword'), splice(STRIDE_CLEAR),
              *CASES['A'][2:])

# Case I-raw converts the structured UAV descriptor to the same byte range before the store.
STRIDE_EXTRACT = 's_bfe_u32 s21, s17, 0xe0010'
SCALE_RECORDS = 's_mul_i32 s18, s18, s21'
OOB_CLEAR = 's_and_b32 s19, s19, 0xcfffffff'
OOB_RAW = 's_or_b32 s19, s19, 0x30000000'
RAW_CONVERSION = (STRIDE_EXTRACT, STRIDE_CLEAR, SCALE_RECORDS, OOB_CLEAR, OOB_RAW)
CASES['I-raw'] = (SOURCE_A.replace('buffer_store_dword', '\n'.join((*RAW_CONVERSION, 'buffer_store_dword'))),
                  splice(*RAW_CONVERSION), *CASES['A'][2:])

# Case I-raw-mask: s21 first holds a lane mask (a compare in a 24-lane group), then the clamped stride of the same
# conversion. The descriptor-data write ends the mark, so the clamp is not a lane-mask read and the store lands as
# in case A. The translator before that change refused it: s_max_u32 reads lane mask s21 as numeric data.
MASK_S21 = 'v_cmp_gt_u32_e64 s21, 3, v0'
CLAMPED_CONVERSION = (STRIDE_EXTRACT, 's_max_u32 s21, s21, 1', *RAW_CONVERSION[1:])
CASES['I-raw-mask'] = (SOURCE_A.replace('buffer_store_dword',
                                        '\n'.join((MASK_S21, *CLAMPED_CONVERSION, 'buffer_store_dword'))),
                       splice(MASK_S21, *CLAMPED_CONVERSION), *CASES['A'][2:])

# Replace the nonzero stride constant in s5 with the first active lane of a
# fresh 24-thread group. Ignoring FF1 leaves a different word at every i > 0.
CASES['A-ff1'] = ('s_ff1_i32_b32 s5, exec_lo\n' + SOURCE_A,
                  bytes.fromhex('7e1385be') + WORDS_A, *CASES['A'][2:])

# `off` stores: the words are hand-assembled from the RDNA 2 SOP1/SOP2/VOP1/VOP3/MUBUF field layouts (the binary
# mode reassembles them with llvm-mc); check_hand_words decodes them with the repository disassembler.
OFF_PROLOGUE = 's_mov_b32 s10, s6\ns_mov_b32 s11, 0\ns_load_dwordx8 s[12:19], s[10:11], 0x0\ns_waitcnt lgkmcnt(0)\n'
OFF_PROLOGUE_WORDS = '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
SOURCE_O_DWORD = OFF_PROLOGUE + """\
s_lshl_b32 s20, s7, 2
s_xor_b32 s21, s4, s7
v_mov_b32 v5, s21
buffer_store_dword v5, off, s[16:19], s20 offset:8
s_endpgm
"""
WORDS_O_DWORD = bytes.fromhex(OFF_PROLOGUE_WORDS + '0782148f' '04071589' '15020a7e' '080070e000050414' '000081bf')
SOURCE_O_SHIFT = OFF_PROLOGUE + """\
s_mov_b32 s20, s4
s_mov_b32 s21, s5
s_lshl_b32 s22, s7, 4
v_lshrrev_b64 v[5:6], 4, s[20:21]
v_mov_b32 v4, s7
v_mov_b32 v7, 0
buffer_store_dwordx4 v[4:7], off, s[16:19], s22
s_endpgm
"""
WORDS_O_SHIFT = bytes.fromhex(OFF_PROLOGUE_WORDS + '040394be' '050395be' '0784168f' '050000d784280000' '0702087e'
                              '80020e7e' '000078e000040416' '000081bf')
CASES['O-dword'] = (SOURCE_O_DWORD, WORDS_O_DWORD, USER7 | TGID_X | tidig(1), (8, 1, 1), (4, 1, 1))
CASES['O-shift'] = (SOURCE_O_SHIFT, WORDS_O_SHIFT, USER7 | TGID_X | tidig(1), (32, 1, 1), (4, 1, 1))

# Two groups reach the same load from different branch arms. Each of their lanes must read its group's word.
SOURCE_JOIN_LOAD = OFF_PROLOGUE + """\
s_cmp_eq_u32 s7, 0
s_cbranch_scc0 1
s_nop 0
s_lshl_b32 s3, s7, 2
buffer_load_dword v7, off, s[12:15], s3
s_waitcnt vmcnt(0)
v_readfirstlane_b32 s24, v7
v_mov_b32_e32 v7, s24
v_lshl_add_u32 v4, s7, 3, v0
v_lshlrev_b32_e32 v4, 2, v4
buffer_store_dword v7, v4, s[16:19], 0 offen
s_endpgm
"""
WORDS_JOIN_LOAD = bytes.fromhex(OFF_PROLOGUE_WORDS + '078006bf 010084bf 000080bf 0782038f 000030e000070303 '
                                '703f8cbf 0705307e 18020e7e 040046d707060104 82080834 001070e004070480 000081bf')
CASES['join-load'] = (SOURCE_JOIN_LOAD, WORDS_JOIN_LOAD, USER7 | TGID_X, (8, 1, 1), (2, 1, 1))


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


# Read the shifted field after the first branch. A zero root constant skips the read and store.
FIELD_TAIL = """\
s_cmp_eq_u32 s4, 0
s_cbranch_scc1 5
v_readfirstlane_b32 s24, v1
v_mov_b32_e32 v7, s24
v_lshlrev_b32_e32 v4, 2, v0
buffer_store_dword v7, v4, s[16:19], 0 offen
s_endpgm
"""
FIELD_TAIL_WORDS = '048006bf 050085bf 0105307e 18020e7e 82000834 001070e004070480 000081bf'
for name, form, encoded, size in (
        ('masked-tid', 'v_and_or_b32 v2, v0, 3, s4\nv_lshrrev_b32_e32 v1, 2, v2',
         '020071d700071100 8204022c', 128),
        ('extracted-tid', 'v_bfe_u32 v2, v0, 2, 3\nv_lshrrev_b32_e32 v1, 1, v2',
         '020048d500050d02 8104022c', 8)):
    CASES[name] = (OFF_PROLOGUE + form + '\n' + FIELD_TAIL, bytes.fromhex(OFF_PROLOGUE_WORDS + encoded + FIELD_TAIL_WORDS),
                   USER7, (size, 1, 1), (1, 1, 1))


def formatted_input():
    # Every selected R8_UNORM field is an exact endpoint, while the UINT16 halves
    # are all distinct and the neighbouring bytes are nonzero.
    data = bytearray(b'\x7b' * (WORDS * 4))
    for lane in range(16):
        at = 6 * lane + 3
        data[at:at + 4] = bytes([255 * (lane % 2), 31 + lane, 62 + lane, 96 + lane])
    return data


def expected(name):
    if name in ('I', 'I-raw', 'I-raw-mask'):
        return expected('A')
    out = [SENTINEL] * WORDS
    if name == 'join-load':
        for i in range(16):
            out[i] = input_word(i // 8)
        return out
    if name == 'O-dword':
        for group in range(4):
            out[2 + group] = KEY ^ group
        return out
    if name == 'O-shift':
        pair = (STRIDE << 32 | KEY) >> 4
        for group in range(4):
            out[4 * group:4 * group + 4] = [group, pair & MASK, pair >> 32, 0]
        return out
    _, _, _, local, groups = CASES[name]
    if name in ('masked-tid', 'extracted-tid'):
        for lane in range(local[0]):
            out[lane] = (((lane & 3) | KEY) >> 2 if name == 'masked-tid' else (((lane >> 2) & 7) >> 1))
        return out
    if name == 'address-carry':
        for group in range(groups[0]):
            total = group + 0xfffffffe
            out[4 * group:4 * group + 3] = [total & 0xffffffff, total >> 32, 0]
        return out
    if name in L_WORD3:
        data = formatted_input()
        for lane in range(16):
            at = 6 * lane + 3
            if name == 'L-uint':
                low, high = struct.unpack_from('<HH', data, at)
                result = (high, low, 1, 0)
            else:
                one = 0x3f800000
                result = (one, one if data[at] == 255 else 0, one, 0)
            out[4 * lane:4 * lane + 4] = result
        return out
    if name == 'N':
        out[:16] = [input_word(0) ^ input_word(lane) for lane in range(16)]
        return out
    if name == 'S':
        for group in range(16):
            out[2 * group:2 * group + 2] = scc_reference(SCC_EDGES[group & 3], SCC_EDGES[group >> 2])
        return out
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
    if name in ('F', 'F-two-fills'):
        for group in range(2):
            for lane in range(64):
                index = (64 * group + lane) * 2
                broadcast = 4 * 37 if name == 'F-two-fills' else 256 * group + 37
                out[index:index + 2] = [256 * group + (lane ^ 32), broadcast]
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
                            elif name in ('P', 'Q'):
                                i = gx * 8 + tx
                                old = 1 if name == 'Q' else int(tx < 4)
                                mask = (tx & 1) if tx < 3 * gx else old  # the block's lanes take the compare, others keep M
                                out[i] = ~input_word(i) & MASK if mask else input_word(i)
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
                            elif name == 'M':
                                i = gx * 8 + tx
                                value = i ^ ((KEY + gx * STRIDE) & MASK)
                                if tx < 3:  # the then arm
                                    out[2 * i] = value
                                elif tx < 6:  # the else arm; lanes 6 and 7 skip the block
                                    out[2 * i + 1] = ~value & MASK
                            elif name == 'NM':
                                i = gx * 8 + tx
                                if tx < 3:  # vcc & exec: the compare's lanes; lanes 6 and 7 skip it with vcc set
                                    out[i] = i ^ KEY
                            elif name == 'O':
                                i = gx * 8 + tx
                                if tx >= 3:  # the saved mask less the lanes the block kept; 6 and 7 skip it
                                    out[i] = i ^ KEY
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


def check_execz_loop_break():
    # Trace the fixture once as a wave and once per singleton. Both saved masks lose
    # the same lanes. SCC tests the entire remaining mask at the early exit.
    def trace(lanes):
        remaining = {lane for lane in lanes if lane < 3}
        values = dict.fromkeys(lanes, 9)
        counter = 1
        while counter < 4:
            if remaining:  # otherwise execz goes straight to the join
                remaining = {lane for lane in remaining if lane >= counter}
                if not remaining:
                    break
                values.update(dict.fromkeys(remaining, counter))
            counter += 1
        return values, counter

    lanes = range(8)
    wave, counter = trace(lanes)
    singles = [trace([lane]) for lane in lanes]
    assert list(wave.values()) == [values[lane] for lane, (values, _) in enumerate(singles)] == [9, 1, 2, 9, 9, 9, 9, 9]
    # Make the scalar counter observable after restoring EXEC. Lane 0 leaves before
    # the wave, while lanes 3..7 follow the latch even after the wave has left.
    assert counter == 3 and [c for _, c in singles] == [1, 2, 3, 4, 4, 4, 4, 4]
    store = '001070e004050480'
    with_counter = patch(LOOP_BREAK_WORDS, store, '18020a7e' + store)
    reason = 's_cbranch_scc0 inside a block s_cbranch_execz skips per lane: loop exit crosses a backward s_cbranch_scc0'
    for program in (LOOP_BREAK_WORDS, with_counter):
        refused(reason, lambda: translate('O', code=program).hlsl_cs())
    # Remove just the latch or just the guard to isolate the control-flow diagnostic.
    refused('s_cbranch_scc0 inside a block s_cbranch_execz skips per lane: branch out of the block',
            lambda: translate('O', code=patch(LOOP_BREAK_WORDS, 'f4ff84bf', '000080bf')).hlsl_cs())
    translate('O', code=patch(LOOP_BREAK_WORDS, '060088bf', '000080bf')).hlsl_cs()


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


LANE_MASK_DATA = (
    ('v_cvt_f32_u32 v5, vcc_lo', '6a0c0a7e'),   # the llvm.amdgcn.ballot word converted to float
    ('v_add_nc_u32 v5, vcc_lo, v5', '6a0a0a4a'),
)


def check_lane_mask_data():
    """A wave mask read as a number holds every lane's bit on hardware; the one-lane model has only its own."""
    compare = bytes.fromhex('8500027d')   # v_cmp_lt_i32 vcc_lo, 5, v0
    select = bytes.fromhex('800a0a02')    # v_cndmask_b32 v5, 0, v5, vcc_lo: the mask as a per-lane select
    assert 'vcc = ' in translate('A', code=splice_words(compare, select)).hlsl_cs()
    for _, word in LANE_MASK_DATA:
        refused('as numeric data', lambda: translate('A', code=splice_words(compare, bytes.fromhex(word))))


LANE_MASK_SCALAR_DATA = (
    ('s_cmpk_gt_u32 vcc_lo, 0x64', '6400eab5'),   # c7_ballot_cmp: the ballot word compared as a number
    ('s_cmp_lg_u32 vcc_lo, 0', '6a8007bf'),
    ('s_lshr_b32 s6, vcc_lo, 1', '6a810690'),
    ('s_add_u32 s6, vcc_lo, 1', '6a810680'),
)


def check_lane_mask_scalar_data():
    """A scalar op that reads a wave mask as a number sees every lane's bit on hardware; mask-only scalar ops stay."""
    compare = bytes.fromhex('8500027d')   # v_cmp_lt_i32 vcc_lo, 5, v0
    for _, word in LANE_MASK_SCALAR_DATA:
        refused('as numeric data', lambda: translate('A', code=splice_words(compare, bytes.fromhex(word))))
    translate('A', code=splice_words(compare, bytes.fromhex('6a7e0687'))).hlsl_cs()   # s_and_b32 s6, vcc_lo, exec_lo


def check_lane_mask_derived_word():
    """A bitwise result of a lane mask is still a mask word: a later numeric read of it is refused."""
    compare = bytes.fromhex('8500027d')              # v_cmp_lt_i32 vcc_lo, 5, v0
    mask_and = bytes.fromhex('6aff0687ff000000')     # s_and_b32 s6, vcc_lo, 0xff
    for use in ('060a887d', '060a0a4a'):   # v_cmp_gt_u32 vcc_lo, s6, v5 ; v_add_nc_u32 v5, s6, v5
        refused('as numeric data', lambda: translate('A', code=splice_words(compare, mask_and, bytes.fromhex(use))))
    refused('as numeric data', lambda: translate('A', code=splice_words(compare, bytes.fromhex('6a0c0a7e'))))  # v_cvt_f32_u32 v5, vcc_lo


def check_lane_mask_single_thread_group():
    """A one-thread group has one participating lane per wave: its lane bit is the whole mask (high word 0), so the one-lane
    model is exact and a numeric read stays lowered (dispatch case E). With two threads the same program is refused."""
    compare = bytes.fromhex('8500027d')   # v_cmp_lt_i32 vcc_lo, 5, v0
    cvt = bytes.fromhex('6a0c0a7e')       # v_cvt_f32_u32 v5, vcc_lo: the mask as a number
    code = splice_words(compare, cvt)
    assert 'v5 = ' in translate('A', code=code, group=(1, 1, 1)).hlsl_cs()
    for group in ((2, 1, 1), (1, 2, 1), (1, 1, 2), (4, 1, 1)):
        refused('as numeric data', lambda: translate('A', code=code, group=group))
    xor_chunks = [bytes.fromhex(w) for w in ('000084d491000200', '8004027d', '6a008089')]
    high = splice_words(*xor_chunks, bytes.fromhex('01020a7e'))   # v_mov_b32 v5, s1: the high word
    assert 'v5 = ' in translate('A', code=high, group=(1, 1, 1)).hlsl_cs()
    refused('as numeric data', lambda: translate('A', code=high, group=(2, 1, 1)))


def check_lane_mask_high_word():
    """The high word of a wave64 mask (the ballot's upper 32 bits) is modelled as 0: a move of it is refused."""
    chunks = [bytes.fromhex(w) for w in ('0000 84d4 9100 0200'.replace(' ', ''),   # v_cmp_gt_i32 s[0:1], 17, v0
                                         '8004027d',                                 # v_cmp_lt_i32 vcc, 0, v2
                                         '6a008089')]                                # s_xor_b64 s[0:1], vcc, s[0:1]
    for word in ('01020a7e',    # v_mov_b32 v5, s1
                 '010a0a4a',    # v_add_nc_u32 v5, s1, v5
                 '010c0a7e',    # v_cvt_f32_u32 v5, s1
                 '010a0a36',    # v_and_b32 v5, s1, v5
                 '01810780',    # s_add_u32 s7, s1, 1
                 '018006bf',    # s_cmp_eq_u32 s1, 0
                 '010387be'):   # s_mov_b32 s7, s1
        refused('as numeric data', lambda: translate('A', code=splice_words(*chunks, bytes.fromhex(word))))
    # The compare alone leaves its high word unknown: refused too (earlier, by the unknown-SGPR rule).
    refused('s1', lambda: translate('A', code=splice_words(chunks[0], bytes.fromhex('010a0a4a'))))
    assert 'v5 = ud0;' in translate('A', code=splice_words(*chunks, bytes.fromhex('04020a7e'))).hlsl_cs()


def check_lane_mask_compare_pair():
    """A wave64 compare or carry-out into an even SGPR pair s[N:N+1] (the decoder names only sN) leaves the other 32 lanes in
    s(N+1). Here s[4:5] are the two root constants, so s5 is a known register: a numeric read of it after the compare must
    be refused, directly, through a copy of the pair, and through a copy of the high word."""
    reads = ('05020a7e', '050c0a7e', '050a0a36', '050a0a4a',   # v_mov / v_cvt_f32_u32 / v_and / v_add with s5
             '05810781', '058006bf', '050387be')                  # s_add_i32 s7, s5, 1 / s_cmp_eq_u32 s5, 0 / s_mov_b32 s7, s5
    for write in ('040084d491000200',                            # v_cmp_gt_i32 s[4:5], 17, v0
                  '06040fd700030200'):                           # v_add_co_u32 v6, s[4:5], v0, v1
        for read in reads:
            refused('as numeric data', lambda write=write, read=read: translate('A', code=splice_words(
                bytes.fromhex(write), bytes.fromhex(read))))
        for copy, use in (('040488be', '09810781'),             # s_mov_b64 s[8:9], s[4:5]; s_add_i32 s7, s9, 1
                          ('050388be', '08810781')):            # s_mov_b32 s8, s5; s_add_i32 s7, s8, 1
            refused('as numeric data', lambda write=write, copy=copy, use=use: translate('A', code=splice_words(
                bytes.fromhex(write), bytes.fromhex(copy), bytes.fromhex(use))))
    # before the compare s5 is the root constant: still a number the model has
    translate('A', code=splice_words(bytes.fromhex('05810781'), bytes.fromhex('040084d491000200'))).hlsl_cs()


def check_lane_mask_copy_of_data_clears_tag():
    """A copy of plain data into a register that held a lane mask on an earlier path (a loop back edge) leaves no mask tag:
    `s_load` through that pair is not a read of a mask. A copy of the mask itself is refused too."""
    loop = bytes.fromhex('0e00c4d483000200'   # v_cmp_gt_u32 s14, 3, v0
                         '800394be'           # s_mov_b32 s20, 0
                         '06038ebe'           # L: s_mov_b32 s14, s6  (the table pointer)
                         '80038fbe'           # s_mov_b32 s15, 0
                         '070608f4000000fa'   # s_load_dwordx4 s[24:27], s[14:15], 0x0
                         '7fc08cbf' '14811481' '14830abf'   # s_waitcnt; s_add_i32 s20, s20, 1; s_cmp_lt_u32 s20, 3
                         'f7ff85bf')          # s_cbranch_scc1 L
    translate('A', code=splice_words(loop)).hlsl_cs()
    mask_copy = loop.replace(bytes.fromhex('06038ebe'), bytes.fromhex('0e038ebe'))   # s_mov_b32 s14, s14: the mask itself
    refused('s_load through s14', lambda: translate('A', code=splice_words(mask_copy)).hlsl_cs())


def check_lane_mask_pair_copy_of_data_stays_refused():
    """A 64-bit copy of plain data (root constants s[4:5]) is lowered as a mask copy (the lane bit only), so its low word
    must not be read as a number in a multi-thread group: the store would write 1, not the root constant."""
    copy = bytes.fromhex('040494be')    # s_mov_b64 s[20:21], s[4:5]
    use = bytes.fromhex('14020a7e')     # v_mov_b32 v5, s20
    refused('is a lane mask', lambda: translate('A', code=splice_words(copy, use), group=(2, 1, 1)).hlsl_cs())
    # a 32-bit copy of the same plain word is a full-word copy and stays readable
    word = bytes.fromhex('040394be')    # s_mov_b32 s20, s4
    assert 'v5 = s20' in translate('A', code=splice_words(word, use), group=(2, 1, 1)).hlsl_cs()


def check_lane_mask_wqm_keeps_tag():
    """s_wqm_b32 is a no-op here (it writes nothing), so a mask it nominally rewrites from plain data keeps its tag:
    the v_mov after it still reads a lane mask in a multi-thread group."""
    cmp_ = bytes.fromhex('140084d491000200')   # v_cmp_gt_i32_e64 s20, 17, v0
    wqm = bytes.fromhex('800994be')            # s_wqm_b32 s20, 0
    use = bytes.fromhex('14020a7e')            # v_mov_b32 v5, s20
    refused('is a lane mask', lambda: translate('A', code=splice_words(cmp_, wqm, use), group=(4, 1, 1)).hlsl_cs())


def check_lane_mask_high_word_copy_and_overwrite():
    """A copy of a mask's high word is itself only a high word (no low-word tag, so no '& 1' on a numeric use in a one-thread
    group), and a wide write over a tagged pair ends the tag (no false refusal of the new value)."""
    save = bytes.fromhex('7e0494be')                      # s_mov_b64 s[20:21], exec
    copy = bytes.fromhex('150396be')                      # s_mov_b32 s22, s21: the high word
    use = bytes.fromhex('16821788')                       # s_or_b32 s23, s22, 2
    cvt = bytes.fromhex('170c0a7e')                       # v_cvt_f32_u32 v5, s23
    one = translate('A', code=splice_words(save, copy, use, cvt), group=(1, 1, 1)).hlsl_cs()
    assert 's23 = s22 | 0x00000002u;' in one, one
    refused('as numeric data', lambda: translate('A', code=splice_words(save, copy, use, cvt), group=(2, 1, 1)))
    shifted = splice_words(save, bytes.fromhex('0407948f'), bytes.fromhex('15020a7e'))   # s_lshl_b64 s[20:21], s[4:5], s7 ; v_mov_b32 v5, s21
    assert 'v5 = ' in translate('A', code=shifted, group=(4, 1, 1)).hlsl_cs()


LANE_MASK_TAG = {
    'zero_s1': '800381be',           # s_mov_b32 s1, 0: the high word the one-lane model holds as 0
    'cmp_pair': '000084d491000200',  # v_cmp_gt_i32 s[0:1], 17, v0 (writes only the low word of the pair)
    'cselect_pair': 'c1808085',      # s_cselect_b64 s[0:1], -1, 0
    'cmp_vcc': '8004027d',           # v_cmp_lt_i32 vcc, 0, v2
    'saveexec': '6a2480be',          # s_and_saveexec_b64 s[0:1], vcc
    'copy_pair': '000488be',         # s_mov_b64 s[8:9], s[0:1]
    'copy_vcc': '6a0488be',          # s_mov_b64 s[8:9], vcc
    'copy_copy': '08048abe',         # s_mov_b64 s[10:11], s[8:9]
    'xor_pair': '6a008089',          # s_xor_b64 s[0:1], vcc, s[0:1]
    'clear_s9': '800389be',          # s_mov_b32 s9, 0: a plain write ends the tag
    'read_s1': '01020a7e',           # v_mov_b32 v5, s1
    'read_s9': '09020a7e',           # v_mov_b32 v5, s9
    'read_s11': '0b020a7e',          # v_mov_b32 v5, s11
    'add_s9': '09810780',            # s_add_u32 s7, s9, 1
}


def check_lane_mask_tag_propagation():
    """The mask tag is set where a mask is written and moves with every copy: one case per way a tagged word is made, each
    read back through a numeric use (a no-op tag at any of them lets a case translate)."""
    w = {k: bytes.fromhex(v) for k, v in LANE_MASK_TAG.items()}
    cases = [
        ('cselect writes a pair', ['zero_s1', 'cselect_pair', 'read_s1']),
        ('saveexec writes a pair', ['zero_s1', 'cmp_vcc', 'saveexec', 'read_s1']),
        ('bitwise op writes a pair', ['zero_s1', 'cmp_pair', 'cmp_vcc', 'xor_pair', 'read_s1']),
        ('b64 copy of a pair', ['cmp_pair', 'cmp_vcc', 'xor_pair', 'copy_pair', 'read_s9']),
        ('b64 copy of vcc', ['cmp_vcc', 'copy_vcc', 'read_s9']),
        ('copy of a copy', ['cmp_pair', 'cmp_vcc', 'xor_pair', 'copy_pair', 'copy_copy', 'read_s11']),
        ('scalar use of a copy', ['cmp_pair', 'cmp_vcc', 'xor_pair', 'copy_pair', 'add_s9']),
    ]
    for label, steps in cases:
        try:
            translate('A', code=splice_words(*[w[s] for s in steps]))
        except G.Unsupported as reason:
            assert 'as numeric data' in str(reason), (label, str(reason))
        else:
            raise AssertionError("%s accepted, expected a refusal naming as numeric data" % label)
    # A plain write ends the tag: the copy is overwritten with a constant before the read.
    assert 'v5 = ' in translate('A', code=splice_words(*[w[s] for s in (
        'cmp_pair', 'cmp_vcc', 'xor_pair', 'copy_pair', 'clear_s9', 'read_s9')])).hlsl_cs()


def splice_words(*chunks, at=STORE_A):
    return WORDS_A[:at] + b''.join(chunks) + WORDS_A[at:]


def check_scc_modelled():
    # s_add / s_sub / s_min / s_max write SCC (carry, borrow, signed overflow, S0 chosen). The lowering used to leave the
    # emitted SCC as it was, so a later s_cbranch_scc or s_cselect read a stale value. Each form now emits its SCC from
    # its sources, ahead of the branch.
    words = {'s_add_u32 s23, s7, -1': '07c11780', 's_sub_u32 s23, s7, 1': '07819780',
             's_add_i32 s23, s7, 0x7fffffff': '07ff1781ffffff7f', 's_sub_i32 s23, s7, 1': '07819781',
             's_min_u32 s23, s7, 4': '07849783', 's_max_i32 s23, s7, 4': '07841784', 's_wqm_b32 s23, s7': '070997be',
             's_cselect_b32 s24, 1, 0': '81801885', 's_cbranch_scc1 1': '010085bf'}
    SPLICE.update(words)
    for line, word in words.items():  # the decoder prints each hand-assembled word as its line (literals in hex)
        (_, _, name, text), = G.decode(bytes.fromhex(word))
        # the decoder prints a branch with its target label and a -1 literal in hex
        assert text == line or text == line.replace('-1', '0xffffffff') or (
            name.startswith('s_cbranch') and text.startswith(name + ' ')), (line, text)
    for op, scc in (('s_add_u32 s23, s7, -1', 'scc = (s7 + 0xffffffffu) < (s7);'),
                    ('s_sub_u32 s23, s7, 1', 'scc = (s7) < (0x00000001u);'),
                    ('s_add_i32 s23, s7, 0x7fffffff',
                     'scc = (((((s7) ^ (s7 + 0x7fffffffu)) & ((0x7fffffffu) ^ (s7 + 0x7fffffffu))) >> 31) != 0);'),
                    ('s_sub_i32 s23, s7, 1', 'scc = (((((s7) ^ (0x00000001u)) & ((s7) ^ (s7 - 0x00000001u))) >> 31) != 0);'),
                    ('s_min_u32 s23, s7, 4', 'scc = ((s7) <= (0x00000004u));'),
                    ('s_max_i32 s23, s7, 4', 'scc = (int(s7) >= int(0x00000004u));')):
        text = translate('A', code=splice(op, 's_cbranch_scc1 1', 'v_mov_b32 v3, 0')).hlsl_cs()
        assert scc in text and text.index(scc) < text.index('pc = (scc)'), (op, text)
    # A reader whose last SCC writer the lowering still does not model is refused by name (s_wqm is not lowered).
    refused('scc from an SCC write that is not modelled in s_cbranch_scc1', lambda: translate(
        'A', code=splice('s_wqm_b32 s23, s7', 's_cbranch_scc1 1', 'v_mov_b32 v3, 0')))
    refused('scc from an SCC write that is not modelled in s_cselect_b32', lambda: translate(
        'A', code=splice('s_wqm_b32 s23, s7', 's_cselect_b32 s24, 1, 0')))
    translate('A', code=splice('s_wqm_b32 s23, s7')).hlsl_cs()  # nothing reads that SCC


def check_descriptor_scc():
    # Case A loads the input descriptor into s[12:15] and stores through s[16:19].
    # Read s20's lane bit into the stored value. The SCC branches skip both dwords of the lane select.
    compare = 's_cmp_lt_u32 s7, 4'
    mask = 's_and_b32 s13, s13, 0xc000ffff'
    select = 's_cselect_b32 s20, 1, 0'

    def pipeline(*lines):
        return translate('A', code=splice(*lines))

    def rejected(*lines):
        refused('scc from descriptor data', lambda: pipeline(*lines))

    for reader in (select, 's_cselect_b64 s[20:21], 1, 0', 's_cmov_b32 s20, 1', 's_cmov_b64 s[20:21], s[4:5]',
                   's_cmovk_i32 s20, 3', 's_cbranch_scc0 2', 's_cbranch_scc1 2'):
        rejected(compare, mask, reader, SCC_LANE_SELECT)
        assert 'uav 0 table 1 16' in pipeline(mask, compare, reader, SCC_LANE_SELECT).manifest()
    rejected(compare, 's_add_u32 s13, s13, 1', select, SCC_LANE_SELECT)
    assert 's20 = scc ?' in pipeline(compare, 's_mov_b32 s13, s13', select, SCC_LANE_SELECT).hlsl_cs()
    assert 's20 = scc ?' in pipeline(mask, 's_and_b32 s22, s7, 1', select, SCC_LANE_SELECT).hlsl_cs()
    assert 's20 = scc ?' in pipeline(mask, 's_and_saveexec_b32 s0, vcc_lo', select, SCC_LANE_SELECT).hlsl_cs()

    # A conditional edge can bypass the fresh comparison. A back edge can carry
    # the descriptor-derived flag to a read that comes first in program order.
    rejected(mask, 's_cbranch_vccz 1', compare, select, SCC_LANE_SELECT)
    rejected(compare, 'v_mov_b32 v3, 0', select, mask, 'v_add_nc_u32 v3, 1, v3',
             'v_cmp_gt_u32 vcc_lo, 2, v3', 's_cbranch_vccnz 65530', SCC_LANE_SELECT)
    assert 's20 = scc ?' in pipeline(compare, 's_branch 2', mask, select, SCC_LANE_SELECT).hlsl_cs()


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
        (store, store + store, 'disjoint'),                           # overlapping fill
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

    second = '400134d804040000'
    second_read = 'd401d8d808000007'
    hlsl = translate('F-two-fills').hlsl_cs()
    assert hlsl.index('lds[(v4 + 320u) >> 2] = v4;') < hlsl.index(sync) < hlsl.index('v7 = lds[')
    reversed_fills = patch(WORDS_F2, store + second, second + store)
    assert 'v7 = lds[(v8 + 468u) >> 2];' in translate('F-two-fills', code=reversed_fills).hlsl_cs()
    for old, new, reason in (
        (second, '', 'initialized span'),                             # omit the second fill
        (second, '0c0134d804040000', 'disjoint'),                     # final word of the first bank
        (second_read, '1001d8d808000007', 'initialized span'),         # first word in the gap
        (second, '040334d804040000', 'bounded'),                      # last lane exceeds allocation
        (second, '400134d805040000', 'address'),                      # unproven address VGPR
        (second, '400136d804040000', 'optional byte offset'),          # GDS
        (second, '400135d804040000', 'reserved bit'),                 # hidden encoding bit
        (second, '400134d804040100', 'unused operand field'),         # unused DATA1
        (second, second + '800234d804040000', 'two'),                 # third fill
        (second, nop + second, 'bounded'),                            # address proof no longer adjacent
    ):
        refused(reason, lambda: translate('F-two-fills', code=patch(WORDS_F2, old, new)))

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


def check_formatted_loads():
    manifest = translate('L-uint').manifest()
    assert manifest.endswith('udsrc 2 stride 0\nudsrc 3 descword3 0\nsrv 0 buf table 1 0\nuav 0 table 1 16\n'), manifest
    prefix = WORDS_L[:WORDS_L.index(LOAD_L)]
    end = bytes.fromhex('000081bf')
    for width in range(1, 5):
        load = LOAD_L[:2] + bytes([4 * (width - 1)]) + LOAD_L[3:]
        hlsl = translate('L-uint', code=prefix + load + end).hlsl_cs()
        assert 'a = v0 * ud2 + v1 + 2;' in hlsl
        assert hlsl.count(' = ((ud3 >> ') == width
    # Word 3 is overwritten in an otherwise intact native descriptor.
    refused('words were rewritten', lambda: translate('L-uint', code=prefix + bytes.fromhex('80038fbe') + LOAD_L + end))
    # At a branch target, there is no local descriptor load to prove.
    refused('local table load', lambda: translate('L-uint', code=prefix + bytes.fromhex('000082bf') + LOAD_L + end))
    local_load = bytes.fromhex('000082bf') + prefix + LOAD_L + end
    assert 'switch ((ud3 >> 12) & 127u)' in translate('L-uint', code=local_load).hlsl_cs()
    check_formatted_pointer_copy()


def check_formatted_pointer_copy():
    """The descriptor load sits past the entry block and its pointer is a copy of the table pointer made in a
    block above it. The copy must be the last write to the pointer on every path to the load."""
    copy, high, load = (bytes.fromhex(h) for h in ('06038abe', '80038bbe', '05030cf4000000fa'))
    assert WORDS_L.startswith(copy + high + load)
    tail = WORDS_L[16:WORDS_L.index(LOAD_L)] + LOAD_L + bytes.fromhex('000081bf')
    target = bytes.fromhex('000082bf')    # s_branch to the next instruction, which becomes a branch target
    skip_one = bytes.fromhex('010084bf')  # s_cbranch_scc0 over one instruction
    again = copy                          # s_mov_b32 s10, s6: a second write of the same pointer value
    same_source = bytes.fromhex('060386be')  # s_mov_b32 s6, s6: a write to the copied register

    def proof(code):
        """The copy proof for the first table load of this program, without the rest of the translation."""
        stage = G.Stage.__new__(G.Stage)
        stage.ins, stage.calculated_pc = list(G.decode(code)), {}
        at = next(i for i, ins in enumerate(stage.ins) if ins[2] == 's_load_dwordx8')
        return stage.formatted_pointer_copy(10, at)

    switch = 'switch ((ud3 >> 12) & 127u)'
    # Copy in the entry block, load and formatted read in the next block.
    assert proof(copy + high + target + load + tail) == (6, 0)
    assert switch in translate('L-uint', code=copy + high + target + load + tail).hlsl_cs()
    # Copy in a later block that every path to the load runs through.
    assert proof(target + copy + high + target + load + tail) == (6, 1)
    assert switch in translate('L-uint', code=target + copy + high + target + load + tail).hlsl_cs()
    # A branch skips the copy: the entry value of s10 reaches the load.
    assert proof(skip_one + copy + load + tail) is None
    assert proof(load + tail) is None
    refused('joined at a branch target', lambda: translate('L-uint', code=skip_one + copy + load + tail))
    # One arm writes the pointer again. The value is the same, so only this proof can refuse it.
    assert proof(copy + high + skip_one + again + load + tail) is None
    refused('local table load', lambda: translate('L-uint', code=copy + high + skip_one + again + load + tail))
    # A back edge from below the formatted read writes the pointer and re-enters at the load.
    loop = copy + high + target + load + tail[:-4] + again + bytes.fromhex('f8ff84bf') + tail[-4:]
    assert [ins[2] for ins in G.decode(loop)][-2:] == ['s_cbranch_scc0', 's_endpgm']
    assert proof(loop) is None
    refused('local table load', lambda: translate('L-uint', code=loop))
    # One path writes descriptor word 3 between the table load and the formatted read.
    word_write = (copy + high + target + load + WORDS_L[16:WORDS_L.index(LOAD_L)] + skip_one +
                  bytes.fromhex('0f038fbe') + LOAD_L + bytes.fromhex('000081bf'))  # s_mov_b32 s15, s15
    refused('local table load', lambda: translate('L-uint', code=word_write))
    # The only writer is not a register copy.
    assert proof(bytes.fromhex('80038abe') + high + target + load + tail) is None
    # The copied register is written on one path into the block that copies it.
    one_path = skip_one + same_source + copy + high + target + load + tail
    assert proof(one_path) == (6, 2)
    refused('local table load', lambda: translate('L-uint', code=one_path))
    # The same program without that write translates.
    assert switch in translate('L-uint', code=skip_one + bytes.fromhex('000080bf') + copy + high + target + load + tail).hlsl_cs()


def check_hand_words():
    """Hand-assembled words decode, with the repository disassembler, to the source line they stand for."""
    def decoded(words, at=0):
        n, _, text, _ = G.D.decode_instruction(words, at)
        return n, text.replace('_e32 ', ' ')
    for line in ('buffer_store_dword v5, off, s[16:19], 0', 'buffer_store_dword v5, off, s[16:19], s20 offset:8',
                 'buffer_store_dwordx2 v[5:6], off, s[16:19], 0', 'v_lshrrev_b64 v[5:6], 4, s[20:21]',
                 'v_lshrrev_b64 v[5:6], 4, v[3:4]'):
        assert decoded(bytes.fromhex(SPLICE[line]))[1] == line, (line, decoded(bytes.fromhex(SPLICE[line])))
    # A branch decodes to its target: 4 bytes past the branch plus 4 bytes per dword of the simm16.
    assert decoded(bytes.fromhex(SPLICE['s_cbranch_execz 4']))[1] == 's_cbranch_execz label_0x14'
    for name in ('O-dword', 'O-shift'):
        source, words = CASES[name][:2]
        at, texts = 0, []
        while at < len(words):
            n, text = decoded(words, at)
            texts.append(text)
            at += n
        assert texts == source.splitlines(), (name, texts)


def check_off_stores():
    """The `off` store (MUBUF offen = idxen = 0): every lane writes the same address, accepted with uniform data."""
    def lines(*code):
        return translate('A', code=replace_store(*code)).hlsl_cs().splitlines()
    hlsl = lines('v_mov_b32 v5, s20', 'buffer_store_dword v5, off, s[16:19], 0')
    at = hlsl.index('    a = (0) & 0xfffffffcu;')
    assert hlsl[at + 1] == '    u0.Store(a, v5);', hlsl[at:at + 2]
    hlsl = lines('v_mov_b32 v5, s20', 'buffer_store_dword v5, off, s[16:19], s20 offset:8')
    at = hlsl.index('    a = (s20 + 8) & 0xfffffffcu;')
    assert hlsl[at + 1] == '    u0.Store(a, v5);', hlsl[at:at + 2]
    # A 64-bit shift of scalar sources is uniform in both halves.
    hlsl = lines('v_lshrrev_b64 v[5:6], 4, s[20:21]', 'buffer_store_dwordx2 v[5:6], off, s[16:19], 0')
    at = hlsl.index('    a = (0) & 0xfffffffcu;')
    assert hlsl[at + 1] == '    u0.Store2(a, uint2(v5, v6));', hlsl[at:at + 2]
    # Past the entry block (inside a guard that skips per lane) the recomputable expression carries the proof.
    guard = ['v_cmp_gt_u32 vcc_lo, 1, v0', 's_and_saveexec_b32 s0, vcc_lo']
    hlsl = lines(*guard, 's_cbranch_execz 3', 'v_mov_b32 v5, s20', 'buffer_store_dword v5, off, s[16:19], 0',
                 's_or_b32 exec_lo, exec_lo, s0')
    assert 'if (exec & 1) u0.Store(a, v5);' in [line.strip() for line in hlsl], hlsl
    hlsl = lines(*guard, 's_cbranch_execz 4', 'v_lshrrev_b64 v[5:6], 4, s[20:21]',
                 'buffer_store_dwordx2 v[5:6], off, s[16:19], 0', 's_or_b32 exec_lo, exec_lo, s0')
    assert 'if (exec & 1) u0.Store2(a, uint2(v5, v6));' in [line.strip() for line in hlsl], hlsl
    # The saveexec between the definition and the store ends the proof (conservative: these lanes are a subset).
    refused('buffer store off with data not proven wave-uniform', lambda: lines(
        'v_mov_b32 v5, s20', *guard, 's_cbranch_execz 3', 's_waitcnt lgkmcnt(0)',
        'buffer_store_dword v5, off, s[16:19], 0', 's_or_b32 exec_lo, exec_lo, s0'))
    # Negative controls: per-lane data (a shift of the thread index; the per-lane load), and uniform data written
    # before an EXEC change, where the entry-block proof no longer knows which lanes store.
    for code in (('v_lshrrev_b64 v[5:6], 4, v[3:4]', 'buffer_store_dwordx2 v[5:6], off, s[16:19], 0'),
                 ('buffer_store_dword v5, off, s[16:19], 0',),
                 ('v_mov_b32 v5, s20', 'v_cmp_gt_u32 vcc_lo, 3, v0', 's_mov_b32 exec_lo, vcc_lo',
                  'buffer_store_dword v5, off, s[16:19], 0')):
        refused('buffer store off with data not proven wave-uniform', lambda code=code: lines(*code))


def check_execz_mask_exact():
    """Exhaustive 4-lane wave: the one-lane model of an s_cbranch_execz block (a lane with exec clear skips it) leaves a
    skipped lane's mask bit as M held it; the wave runs the block whenever any lane is active. Which block writes keep
    that bit: preserving ones always, a compare / exec-masked move only if M was 0, an s_orn2 with exec only if M was 1s."""
    full = 15
    writes = {'cmp': lambda m, e, c: c & e, 'mov_exec': lambda m, e, c: e, 'andn2': lambda m, e, c: m & ~e & full,
              'blend': lambda m, e, c: (m & ~e & full) | (c & e), 'or_cmp': lambda m, e, c: m | (c & e),
              'orn2': lambda m, e, c: (c | ~e) & full}
    for name, write in writes.items():
        bad = set()
        for m, e, c in itertools.product(range(16), repeat=3):
            wave = write(m, e, c) if e else m
            if any(((wave ^ m) >> lane) & 1 for lane in range(4) if not (e >> lane) & 1):
                bad.add(m)
        want = {'cmp': set(range(1, 16)), 'mov_exec': set(range(1, 16)), 'orn2': set(range(15))}.get(name, set())
        assert bad == want, (name, sorted(bad))


def check_controls():
    for name in ('masked-tid', 'extracted-tid'):
        translate(name).hlsl_cs()
    # Shift by one to retain masked bit 1, or enable extracted bit 3 in a larger group.
    masked = patch(CASES['masked-tid'][1], '8204022c', '8104022c')
    refused('requires a proven uniform source', lambda: translate('masked-tid', code=masked))
    refused('requires a proven uniform source', lambda: translate('extracted-tid', group=(16, 1, 1)))
    carry = translate('address-carry').hlsl_cs()
    assert 's21 = (scc ? 1u : 0u);' in carry
    lo, hi, end = (bytes.fromhex(x) for x in ('04051480', '80fd1580', '000081bf'))
    for near_miss in (
            bytes.fromhex('010084bf') + lo + hi,             # branch into the high add
            bytes.fromhex('06811480') + hi):                 # elided descriptor arithmetic
        refused('src_scc', lambda: translate('address-carry', code=near_miss + end))
    # Every SCC writer is modelled, so src_scc after a plain add reads the carry the add computed, whatever sits between
    # (an intervening s_sub_u32 writes the borrow) or follows: the add's own carry is emitted ahead of its use.
    for name, ok in (('s_sub_u32 between', lo + bytes.fromhex('81809e80') + hi),
                     ('non-adjacent', lo + bytes.fromhex('000080bf') + hi),
                     ('high addend is one', lo + bytes.fromhex('81fd1580')),
                     ('high register is not the next SGPR', lo + bytes.fromhex('80fd1680'))):
        text = translate('address-carry', code=ok + end).hlsl_cs()
        want = '(0x00000001u) < (0x00000000u)' if name.startswith('s_sub') else '(ud0 + ud1) < (ud0)'
        assert want in text and text.index(want) < text.index('(scc ? 1u : 0u)'), (name, text)
    refused('lane mask', lambda: translate('address-carry', code=bytes.fromhex('7e811480') + hi + end))
    ff1 = bytes.fromhex('7e1385be')
    end = bytes.fromhex('000081bf')
    # A write to EXEC can disable lane zero; a backward branch can re-enter
    # the instruction with a different mask even at code offset zero.
    for code in (bytes.fromhex('8003febe') + ff1 + end,
                 ff1 + bytes.fromhex('feff82bf') + end):
        refused('compute entry EXEC mask', lambda: translate('A', code=code))
    # vcc_lo is a lane mask; general-scalar ff1 (#2022 Item F) still refuses it.
    refused('lane mask', lambda: translate('A', code=bytes.fromhex('6a1385be') + end))
    assert 's5 = 0u;' in translate('A-ff1').hlsl_cs()
    assert expected('A-ff1') != expected('A')
    check_lds()
    check_descriptor_scc()
    check_scc_modelled()
    check_mbcnt()
    check_execz_loop_break()
    check_lane_mask_data()
    check_lane_mask_high_word()
    check_lane_mask_compare_pair()
    check_lane_mask_copy_of_data_clears_tag()
    check_lane_mask_pair_copy_of_data_stays_refused()
    check_lane_mask_wqm_keeps_tag()
    check_lane_mask_single_thread_group()
    check_lane_mask_high_word_copy_and_overwrite()
    check_lane_mask_derived_word()
    check_lane_mask_scalar_data()
    check_lane_mask_tag_propagation()
    check_formatted_loads()
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
    refused('TG_SIZE_EN', lambda: G.ComputePipeline(end, root_blob(), USER7 | TG_SIZE_EN, (8, 4, 2)))
    refused('thread-ID component count', lambda: G.ComputePipeline(end, root_blob(), USER7 | (3 << 11), (1, 1, 1)))
    refused('gives 6 user SGPRs', lambda: G.ComputePipeline(end, root_blob(), 6 << 1, (1, 1, 1)))
    refused('thread group', lambda: G.ComputePipeline(end, root_blob(), USER7, (1024, 2, 1)))
    # A user SGPR outside the root's span has no known value: s0 (s_mov_b32 s10, s0) and s7 when user is 8.
    refused('numeric use of s0', lambda: translate('A', code=patch(WORDS_A, '06038abe', '00038abe')).hlsl_cs())
    refused('numeric use of s7', lambda: translate('A', rsrc2=(8 << 1) | tidig(3)).hlsl_cs())
    # Store forms outside this slice (the store is the dword at +0x60 of WORDS_A: offen, soffset 0x80, s[16:19]).
    store = '001070e004050480'
    for word, fragment in (('000070e004050480', 'not proven wave-uniform'),  # off: one address, per-lane data
                           ('002070e004050480', 'modifier idxen'),         # idxen, no offen
                           ('005070e004050480', 'modifier glc')):          # glc offen
        # (a store through a descriptor the program also loads is lowered as one RW view: startup_compute_rw_buffer)
        refused(fragment, lambda w=word: translate('A', code=patch(WORDS_A, store, w)).hlsl_cs())
    # Graphics stages keep the unknown-op refusal text, so refusal census buckets do not move.
    for count, text in ((1, 'buffer_store_dword'), (4, 'buffer_store_dwordx4')):
        try:
            G.Stage.op_buffer_store_dword(type('P', (), {'kind': 'ps'})(), ['v5', 'v4', 's[16:19]', '0'], ['offen'], count)
        except G.Unsupported as reason:
            assert str(reason) == text, str(reason)
        else:
            raise AssertionError('graphics store accepted')
    check_hand_words()
    check_off_stores()
    check_execz_mask_exact()
    for name in ('O-dword', 'O-shift'):
        hlsl = translate(name).hlsl_cs().splitlines()
        assert any('u0.Store' in line for line in hlsl), (name, hlsl)
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
    raw = translate('I-raw')
    plain = translate('A')
    assert raw.hlsl_cs() == plain.hlsl_cs()
    assert raw.manifest() == plain.manifest()
    assert translate('A', code=splice(STRIDE_EXTRACT, clear, SCALE_RECORDS, OOB_RAW)).hlsl_cs() == raw.hlsl_cs()
    # Captured programs clamp the extracted stride to at least 1 before scaling NUM_RECORDS.
    clamped = (STRIDE_EXTRACT, 's_max_u32 s21, s21, 1', *RAW_CONVERSION[1:])
    assert translate('A', code=splice(*clamped)).hlsl_cs() == raw.hlsl_cs()
    # After a lane mask in s21 the clamped conversion still matches case I-raw apart from the compare itself; the
    # dispatch of case I-raw-mask stores through it. Control: a clamp that reads the mask itself stays refused.
    masked = translate('I-raw-mask').hlsl_cs().splitlines()
    assert [line for line in masked if line not in raw.hlsl_cs().splitlines()] == [
        '    s21 = ((0x00000003u > v0) ? 1u : 0u);'], masked
    refused('s_max_u32 reads lane mask s21 as numeric data', lambda: translate(
        'A', code=splice(MASK_S21, 's_max_u32 s21, s21, 1', *RAW_CONVERSION[1:])).hlsl_cs())
    refused('buffer store through a modified descriptor', lambda: translate(
        'A', code=splice(STRIDE_EXTRACT, 's_max_u32 s21, s21, 2', *RAW_CONVERSION[1:])).hlsl_cs())
    reload = ('s_load_dwordx8 s[24:31], s[10:11], 0x0', 's_waitcnt lgkmcnt(0)')
    other = ('s_load_dwordx8 s[24:31], s[10:11], 0x20', 's_waitcnt lgkmcnt(0)')  # the next two descriptors
    # A second load through the same untouched pointer pair reads the same words, so a word taken from it is the
    # same value. After s_add_u32 s10, s10, 32 the pointer symbol hides the addend, so that load is told apart.
    for lines in ((*reload, 's_mov_b32 s19, s31', *RAW_CONVERSION),
                  (*reload, STRIDE_EXTRACT, 's_mov_b32 s17, s29', clear, SCALE_RECORDS, OOB_RAW)):
        assert translate('A', code=splice(*lines)).hlsl_cs() == raw.hlsl_cs()
    moved = ('s_add_u32 s10, s10, 32', *reload)
    rehigh = ('s_mov_b32 s11, 1', *reload)  # another high address word: another descriptor
    for lines in (
        (*rehigh, 's_bfe_u32 s21, s29, 0xe0010', clear, SCALE_RECORDS, OOB_RAW),
        (*moved, 's_mov_b32 s19, s31', *RAW_CONVERSION),
        (*moved, 's_bfe_u32 s21, s29, 0xe0010', clear, SCALE_RECORDS, OOB_RAW),
        (*RAW_CONVERSION[:-1], 's_or_b32 s19, s19, 0x20000000'),
        (STRIDE_EXTRACT, clear, OOB_RAW),
        (STRIDE_EXTRACT, SCALE_RECORDS, OOB_RAW),
        (clear, 's_mul_i32 s18, s18, 4', OOB_RAW),
        (*other, 's_mov_b32 s19, s31', *RAW_CONVERSION),
        (*other, STRIDE_EXTRACT, 's_mov_b32 s17, s29', clear, SCALE_RECORDS, OOB_RAW),
        (*other, 's_bfe_u32 s21, s29, 0xe0010', clear, SCALE_RECORDS, OOB_RAW),  # another descriptor's stride
        ('s_bfe_u32 s21, s17, 0xe0011', clear, SCALE_RECORDS, OOB_RAW),  # not the stride field (bits 30:17)
        (*RAW_CONVERSION[:-1], 's_cmp_lt_u32 s7, 4', 's_cbranch_scc1 2', OOB_RAW),
    ):
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
    # Exec writes in a block with branches. The proof follows, over every path from the branch, which masks keep
    # the bit of a skipped lane clear; an exec write must leave exec among them. block() lists the refusals of a
    # block of one-word instructions spliced between `enter` and `leave`.
    words = {
        's_mov_b32 s0, exec_lo': '7e0380be', 's_mov_b32 exec_lo, s0': '0003febe', 's_mov_b32 s0, s21': '150380be',
        's_mov_b32 s0, 0': '800380be', 's_mov_b32 s0, 4': '840380be', 's_or_b32 s0, s0, exec_lo': '007e0088',
        's_andn2_b32 exec_lo, s0, exec_lo': '007e7e8a', 's_orn2_b32 exec_lo, s0, exec_lo': '007e7e8b',
        's_xor_b32 exec_lo, exec_lo, s0': '7e007e89', 's_and_b32 exec_lo, exec_lo, vcc_lo': '7e6a7e87',
        's_or_saveexec_b32 s0, s0': '003d80be', 's_andn2_wrexec_b32 s0, s0': '004780be',
        's_mov_b32 s22, exec_hi': '7f0396be', 's_or_b32 exec_lo, exec_lo, s22': '7e167e88',
        's_cbranch_execnz 65532': 'fcff89bf', 's_cbranch_scc1 65532': 'fcff85bf'}
    SPLICE.update(words)
    for line, word in words.items():  # the repository decoder prints each hand-assembled word as its line
        (_, _, name, text), = G.decode(bytes.fromhex(word))
        assert text == line or name == line.split()[0] and name.startswith('s_cbranch'), (line, text)

    def block(*lines):
        refusals = []
        SPLICE['s_cbranch_execz %d' % len(lines)] = '%02x0088bf' % len(lines)
        G.ComputePipeline(splice(*enter, 's_cbranch_execz %d' % len(lines), *lines, *leave), root_blob(), CASES['A'][2],
                          (2, 3, 4), refusals=refusals)
        return [reason for reason, _ in refusals]

    exec_write = '%s inside a block s_cbranch_execz skips per lane: exec write'
    nested = ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_saveexec_b32 s0, vcc_lo']
    # An if/else whose arms have their own s_cbranch_execz, switched with s_andn2 of the saved mask and restored
    # with a move of it, and the same if/else switched with s_or_saveexec and s_xor: accepted. Case M runs the
    # first on the GPU.
    assert block(*nested, 's_cbranch_execz 1', 's_add_u32 s23, s7, 1', 's_andn2_b32 exec_lo, s0, exec_lo',
                 's_cbranch_execz 1', 's_add_u32 s23, s7, 1', 's_mov_b32 exec_lo, s0') == []
    assert block(*nested, 's_or_saveexec_b32 s0, s0', 's_xor_b32 exec_lo, exec_lo, s0',
                 's_or_b32 exec_lo, exec_lo, s0') == []
    # A loop that narrows exec until no lane is left, then restores the mask saved before it: accepted.
    assert block('s_mov_b32 s0, exec_lo', 'v_mov_b32 v3, 0', 'v_add_nc_u32 v3, 1, v3', 'v_cmp_gt_u32 vcc_lo, 2, v3',
                 's_and_b32 exec_lo, exec_lo, vcc_lo', 's_cbranch_execnz 65532', 's_mov_b32 exec_lo, s0') == []
    # A loop that restores exec from s0. With s0 left alone in the body it is accepted; when the body then
    # overwrites s0 with the mask saved before the block, the back edge brings the restore, through the loop's
    # first instruction, a mask that can set a skipped lane.
    loop = ['s_mov_b32 s0, exec_lo', 's_add_u32 s23, s7, 1', 's_mov_b32 exec_lo, s0', 's_add_u32 s23, s7, 1',
            's_cbranch_scc1 65532']
    assert block(*loop) == []
    assert block(*loop[:3], 's_mov_b32 s0, s21', loop[4]) == [exec_write % 's_mov_b32']
    # A mask built from a zero literal and exec is clear for a skipped lane; one built from another literal is not.
    assert block('s_mov_b32 s0, 0', 's_or_b32 s0, s0, exec_lo', 's_mov_b32 exec_lo, s0') == []
    assert block('s_mov_b32 s0, 4', 's_mov_b32 exec_lo, s0') == [exec_write % 's_mov_b32']
    # s_or_saveexec and s_andn2_wrexec keep exec clear only when their source is: a copy of exec saved in the block
    # is accepted, s0 as it was before the block and a copy of the mask saved before the block are refused.
    assert block('s_mov_b32 s0, exec_lo', 's_andn2_wrexec_b32 s0, s0') == []
    assert block('s_mov_b32 s0, s21', 's_andn2_wrexec_b32 s0, s0') == [exec_write % 's_andn2_wrexec_b32']
    assert block('s_or_saveexec_b32 s0, s0') == [exec_write % 's_or_saveexec_b32']
    # A form the proof does not follow (s_orn2 sets the lanes its second source leaves clear), and a copy of exec_hi
    # ORed into exec_lo (the fact names the half of the wave it holds for): refused.
    assert block('s_mov_b32 s0, exec_lo', 's_orn2_b32 exec_lo, s0, exec_lo') == [exec_write % 's_orn2_b32']
    assert exec_write % 's_or_b32' in block('s_mov_b32 s22, exec_hi', 's_or_b32 exec_lo, exec_lo, s22')
    # The proof's step on its own: what a skipped lane's bit is known clear in, before and after one instruction.
    start = {('exec', 0), ('exec_hi', 1)}
    for facts, line, after in (
            (start, 's_and_saveexec_b64 s[2:3], vcc', start | {('s2', 0), ('s3', 1)}),
            (start, 's_mov_b32 s2, exec_hi', start | {('s2', 1)}),
            (start | {('s2', 0), ('s3', 0)}, 's_mov_b64 exec, s[2:3]', {('exec', 0), ('exec_hi', 0), ('s2', 0), ('s3', 0)}),
            (start | {('s2', 0)}, 's_andn2_b32 exec_lo, s2, exec_lo', start | {('s2', 0)}),
            (start | {('s2', 0)}, 's_andn2_b32 exec_lo, exec_lo, s2', start | {('s2', 0)}),
            (start | {('s2', 0)}, 's_andn2_b32 exec_lo, s4, s2', {('exec_hi', 1), ('s2', 0)}),
            (start | {('s2', 0)}, 's_add_u32 s2, s2, 1', start),
            (start | {('s2', 0)}, 'v_cmp_gt_u32_e64 s2, 1, v0', start),
            (start, 'v_cmpx_gt_u32_e32 1, v0', start)):
        name = line.split()[0]
        assert G.clear_after(name, line, G.scalar_effects(name, line)[2], facts) == after, line
    # Case M's block: every statement of the lane that enters it, and the per-lane branches around both arms.
    assert translate('M').hlsl_cs().endswith('''\
    if (pc == 1) {
        s22 = s7 * ud1;
        scc = (s22 + ud0) < (s22);
        s22 = s22 + ud0;
        if (exec & 1) v5 = s22 ^ v3;
        vcc = (exec & ((0x00000003u > v0) ? 1u : 0u));
        s0 = exec;
        exec = (exec & vcc) & 1; scc = exec != 0;
        pc = ((exec & 1) == 0) ? 3 : 2;
    }
    if (pc == 2) {
        a = (v4 + 0) & 0xfffffffcu;
        if (exec & 1) u0.Store(a, v5);
        pc = 3;
    }
    if (pc == 3) {
        exec = (s0 & ~exec) & 1;
        scc = exec != 0;
        pc = ((exec & 1) == 0) ? 5 : 4;
    }
    if (pc == 4) {
        if (exec & 1) v6 = ~v5;
        a = (v4 + 4) & 0xfffffffcu;
        if (exec & 1) u0.Store(a, v6);
        pc = 5;
    }
    if (pc == 5) {
        exec = (s0) & 1;
        pc = 6;
    }
    if (pc == 6) {
        exec = (s21) & 1;
        pc = 0xffffu;
    }
}
''')
    # A definition live where the block joins whose reads are all absorbed by exec. The block's exec writes are
    # proven, so a lane that skipped it has its exec bit clear at the join in the wave as in the model; while exec is
    # unchanged since the join, W & exec and exec & W give that lane 0 and ~(W & exec) and W | ~exec give it 1,
    # whatever W holds. Accepted at the join (case N runs the first on the GPU), and when the read overwrites W.
    SPLICE.update({
        's_and_b32 s23, vcc_lo, exec_lo': '6a7e1787', 's_and_b32 s23, exec_lo, vcc_lo': '7e6a1787',
        's_nand_b32 s23, vcc_lo, exec_lo': '6a7e178c', 's_orn2_b32 s23, vcc_lo, exec_lo': '6a7e178b',
        's_and_b32 vcc_lo, vcc_lo, exec_lo': '6a7e6a87', 's_orn2_b32 s23, exec_lo, vcc_lo': '7e6a178b',
        's_mov_b32 s23, vcc_lo': '6a0397be', 's_and_b32 s23, vcc_lo, exec_hi': '6a7f1787', 's_and_b32 s23, vcc_hi, exec_lo': '6b7e1787',
        's_cbranch_scc1 65533': 'fdff85bf'})
    compare = ['s_cbranch_execz 1', 'v_cmp_gt_u32 vcc_lo, 3, v0']
    for read in ('s_and_b32 s23, vcc_lo, exec_lo', 's_and_b32 s23, exec_lo, vcc_lo', 's_orn2_b32 s23, vcc_lo, exec_lo',
                 's_and_b32 vcc_lo, vcc_lo, exec_lo'):
        translate('A', code=splice(*enter, *compare, read, *leave, 's_cbranch_vccz 1', 'v_mov_b32 v5, s20')
                  if read.startswith('s_and_b32 vcc') else splice(*enter, *compare, read, *leave)).hlsl_cs()
    assert translate('NM').hlsl_cs().endswith('''\
    if (pc == 1) {
        vcc = (exec & ((0x00000003u > v0) ? 1u : 0u));
        pc = 2;
    }
    if (pc == 2) {
        s23 = (vcc & exec) & 1;
        scc = s23 != 0;
        exec = (exec | s0) & 1;
        scc = exec != 0;
        if (exec & 1) v5 = ud0 ^ v3;
        exec = (s23) & 1;
        a = (v4 + 0) & 0xfffffffcu;
        if (exec & 1) u0.Store(a, v5);
        exec = (s21) & 1;
        pc = 0xffffu;
    }
}
''')
    # Refused: a read once exec has been restored, a copy, exec | ~vcc (vcc decides), exec_hi in the low half, the
    # vcc branch after an absorbed read, and an absorbed read at the join that a loop re-enters after the restore
    # (exec is no longer the join's there). The reason names the first read that is not proven.
    live = 'v_cmp_gt_u32_e32 inside a block s_cbranch_execz skips per lane: definition of vcc, live where the block joins'
    for before, after, name, cause in (
            ([], ['s_and_b32 s23, vcc_lo, exec_lo'], 's_and_b32', 'exec is written between the join and the read'),
            (['s_mov_b32 s23, vcc_lo'], [], 's_mov_b32', 'lane-mask operation mov'),
            (['s_orn2_b32 s23, exec_lo, vcc_lo'], [], 's_orn2_b32', 'the word is not paired with the exec word'),
            (['s_and_b32 s23, vcc_lo, exec_hi'], [], 's_and_b32', 'the word is not paired with the exec word'),
            (['s_and_b32 s23, vcc_lo, exec_lo'], ['s_cbranch_vccz 1', 'v_mov_b32 v5, s20'], 's_cbranch_vccz',
             'a data read or a copy'),
            (['s_and_b32 s23, vcc_lo, exec_lo'], ['s_cbranch_scc1 65533'], 's_and_b32', 'another path enters the join')):
        refused('%s: read by %s after the join (cause: %s' % (live, name, cause), lambda before=before, after=after: translate(
            'A', code=splice(*enter, *compare, *before, *leave, *after)).hlsl_cs())
    # Wave64: the compare writes vcc_hi too (its decoded text names vcc_lo only). A 32-bit read of vcc_hi after the join
    # is not lane-wise against exec_lo, so it is refused where the same read of vcc_lo above is accepted.
    refused('v_cmp_gt_u32_e32 inside a block s_cbranch_execz skips per lane: definition of vcc_hi, live where the block '
            'joins: read by s_and_b32 after the join',
            lambda: translate('A', code=splice(*enter, *compare, *leave, 's_and_b32 s23, vcc_hi, exec_lo')).hlsl_cs())
    assert G.exec_absorbs('s_and_b32', 's_and_b32 s2, vcc_hi, exec_lo', 'vcc_hi') is False
    # The rule on its own: which reads exec absorbs for a word.
    for line, word, absorbed in (('s_and_b32 s2, s3, exec_lo', 's3', True), ('s_and_b64 s[2:3], exec, s[4:5]', 's5', True),
                                 ('s_nand_b32 s2, exec_lo, s3', 's3', True), ('s_orn2_b32 s2, s3, exec_lo', 's3', True),
                                 ('s_orn2_b32 s2, exec_lo, s3', 's3', False), ('s_or_b32 s2, s3, exec_lo', 's3', False),
                                 ('s_and_b32 s2, s3, s3', 's3', False), ('s_and_b64 s[2:3], s[4:5], exec', 's4', True),
                                 ('s_and_b64 s[2:3], s[4:5], s[4:5]', 's4', False), ('s_and_b32 s2, s3, exec_hi', 's3', False),
                                 ('s_and_b64 s[2:3], s[4:5], exec_lo', 's4', False)):
        assert G.exec_absorbs(line.split()[0], line, word) == absorbed, line
    # A mask the block only narrows by lanes whose exec bit is clear for a skipped lane: `s_andn2 R, R, X` with X
    # clear for that lane (the exec-write proof's fact at that instruction) keeps its bit of R, so at the join it is
    # the model's. Reads that take bit n of the result from bit n of R, for the same half, and write it to that exec
    # word or back to R are then the model's too. Accepted on the restore after the block (case O runs it on the GPU).
    SPLICE.update({'s_andn2_b32 s0, s0, exec_lo': '007e008a', 's_andn2_b32 s0, s0, s21': '0015008a',
                   's_mov_b32 s1, s0': '000381be', 's_add_u32 s23, s0, 1': '00811780'})
    save = ['s_mov_b32 s21, exec_lo', 'v_cmp_gt_u32 vcc_lo, 1, v0', 's_and_saveexec_b32 s0, vcc_lo']
    translate('A', code=splice(*save, 's_cbranch_execz 1', 's_andn2_b32 s0, s0, exec_lo', 's_mov_b32 exec_lo, s0',
                               *leave)).hlsl_cs()
    assert translate('O').hlsl_cs().endswith('''\
    if (pc == 1) {
        vcc = (exec & ((0x00000003u > v0) ? 1u : 0u));
        exec = (exec & vcc) & 1;
        scc = exec != 0;
        s0 = (s0 & ~exec) & 1;
        scc = s0 != 0;
        pc = 2;
    }
    if (pc == 2) {
        exec = (s0) & 1;
        if (exec & 1) v5 = ud0 ^ v3;
        a = (v4 + 0) & 0xfffffffcu;
        if (exec & 1) u0.Store(a, v5);
        exec = (s21) & 1;
        pc = 0xffffu;
    }
}
''')
    # Refused: X not clear (the mask saved before the block), another write of s0 in the block, alone or after the
    # narrowing, a copy and a data read after the join, and a block that a branch from outside enters (the fact is
    # not proven there).
    kept = ' inside a block s_cbranch_execz skips per lane: definition of s0, live where the block joins: read by %s ' \
           'after the join'
    for prefix, body, join, inner, name in (
            ([], ['s_andn2_b32 s0, s0, s21'], [], 's_andn2_b32', 's_mov_b32'),
            ([], ['s_mov_b32 s0, 4'], [], 's_mov_b32', 's_mov_b32'),
            ([], ['s_andn2_b32 s0, s0, exec_lo', 's_mov_b32 s0, 4'], [], 's_andn2_b32', 's_mov_b32'),
            ([], ['s_andn2_b32 s0, s0, exec_lo'], ['s_mov_b32 s1, s0'], 's_andn2_b32', 's_mov_b32'),
            ([], ['s_andn2_b32 s0, s0, exec_lo'], ['s_add_u32 s23, s0, 1'], 's_andn2_b32', 's_add_u32'),
            (['s_cbranch_scc1 1'], ['s_andn2_b32 s0, s0, exec_lo'], [], 's_andn2_b32', 's_mov_b32')):
        refused(inner + kept % name, lambda prefix=prefix, body=body, join=join: translate('A', code=splice(
            *save, *prefix, 's_cbranch_execz %d' % len(body), *body, *join, 's_mov_b32 exec_lo, s0', *leave)).hlsl_cs())
    # The rule's steps on their own: the halves a write keeps, and the reads that keep the lane's bit.
    for line, word, clear, halves in (('s_andn2_b32 s0, s0, exec_lo', 's0', start, {0}),
                                      ('s_andn2_b64 s[0:1], s[0:1], exec', 's1', start, {1}),
                                      ('s_andn2_b64 s[0:1], s[0:1], exec', 's1', {('exec', 0)}, set()),
                                      ('s_andn2_b32 s0, s0, exec_hi', 's0', start, {1}),
                                      ('s_andn2_b32 s0, s1, exec_lo', 's0', start, set()),
                                      ('s_or_b32 s0, s0, exec_lo', 's0', start, set())):
        assert G.kept_halves(line.split()[0], line, word, clear) == halves, line
    for line, word, halves, ok in (('s_mov_b32 exec_lo, s0', 's0', {0}, True), ('s_mov_b32 exec_lo, s0', 's0', {1}, False),
                                   ('s_mov_b64 exec, s[0:1]', 's1', {1}, True), ('s_mov_b64 exec, s[0:1]', 's1', {0}, False),
                                   ('s_andn2_b32 s0, s0, exec_lo', 's0', {0}, True), ('s_mov_b32 s1, s0', 's0', {0}, False),
                                   ('s_and_b32 exec_lo, s0, exec_hi', 's0', {0}, False),
                                   ('s_mov_b32 exec_hi, s0', 's0', {1}, False)):
        assert G.lane_read(line.split()[0], line, word, halves) == ok, line
    # The compiler's i1-phi lane mask: M is zeroed before the branch, set from exec in the block, and read per lane
    # after the join. For a lane the model skips, the wave's M bit is 0 either way (exec only narrows in the block),
    # which is the model's M. Accepted for vcc_lo and for an sN, with a move or an s_and in either operand order.
    SPLICE.update({'s_mov_b32 vcc_lo, 0': '8003eabe', 's_mov_b32 vcc_lo, -1': 'c103eabe', 's_mov_b32 s20, 0': '800394be',
                   's_mov_b32 s20, -1': 'c10394be', 's_mov_b32 s20, exec_lo': '7e0394be', 's_mov_b32 vcc_lo, s7': '0703eabe',
                   's_mov_b32 s20, s7': '070394be', 's_mov_b32 vcc_lo, exec_lo': '7e03eabe',
                   's_and_b32 vcc_lo, s7, exec_lo': '077e6a87', 's_and_b32 s20, exec_lo, s7': '7e071487',
                   's_and_b32 s20, s7, exec_lo': '077e1487', 's_mov_b64 vcc, exec': '7e04eabe', 'v_cndmask_b32 v5, v5, v6, vcc_lo': '050d0a02',
                   'v_cndmask_b32 v5, v5, v6, s20': '050001d5050d5200', 's_cbranch_vccz 1': '010086bf',
                   's_cmp_eq_u32 s20, 0': '148006bf', 's_and_b32 s20, vcc_lo, exec_lo': '6a7e1487', 's_mov_b32 s20, vcc_lo': '6a0394be', 's_mov_b32 s7, s20': '140387be',
                   's_branch 1': '010082bf', 's_cbranch_scc1 1': '010085bf', 's_cbranch_scc1 0': '000085bf',
                   's_cbranch_scc1 2': '020085bf', 's_and_b32 vcc_lo, s7, s7': '07076a87',
                   'v_cndmask_b32 v5, vcc_lo, v6, vcc_lo': '050001d56a0caa01'})
    def phi(zero, write, read, between=(), inside=()):
        return splice(*enter, *zero, *between, 's_cbranch_execz %d' % (len(write) + len(inside)), *write, *inside,
                      *leave, *read)
    phi_text = lambda code: translate('A', code=code).hlsl_cs()
    SPLICE.update({'s_and_b32 s20, exec_lo, vcc_lo': '7e6a1487'})
    cnd_vcc, cnd_s = ['v_cndmask_b32 v5, v5, v6, vcc_lo'], ['v_cndmask_b32 v5, v5, v6, s20']
    for zero, write, read in ((['s_mov_b32 vcc_lo, 0'], ['s_mov_b32 vcc_lo, exec_lo'], cnd_vcc),
                              (['s_mov_b32 s20, 0'], ['s_mov_b32 s20, exec_lo'], cnd_s),
                              (['s_mov_b32 vcc_lo, 0'], ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_b32 vcc_lo, vcc_lo, exec_lo'], cnd_vcc),
                              (['s_mov_b32 s20, 0'], ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_b32 s20, vcc_lo, exec_lo'], cnd_s),
                              (['s_mov_b32 s20, 0'], ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_b32 s20, exec_lo, vcc_lo'], cnd_s)):
        text = phi_text(phi(zero, write, read))
        # the lane select reads the mask's lane bit, which the block set from exec (the model runs the block for
        # every lane it models, and skips it only where exec is clear, leaving the zeroing)
        assert '& 1) ? ' in text and ('vcc = (exec) & 1;' in text or 's20 = ' in text), text
    # an s_and of exec with data (a group id) is not a lane mask: its lane bit is at bit `lane` in the wave
    for write in (['s_and_b32 vcc_lo, s7, exec_lo'], ['s_and_b32 s20, s7, exec_lo']):
        refused('definition of ' + ('vcc' if 'vcc_lo, s7' in write[0] else 's20') + ', live where the block joins',
                lambda write=write: phi_text(phi(['s_mov_b32 %s, 0' % ('vcc_lo' if 'vcc_lo, s7' in write[0] else 's20')],
                                                 write, cnd_vcc if 'vcc_lo, s7' in write[0] else cnd_s)))
    # s_and with exec reads the lane bit only (the result is a lane mask): accepted
    phi_text(phi(['s_mov_b32 vcc_lo, 0'], ['s_mov_b32 vcc_lo, exec_lo'], ['s_and_b32 s20, vcc_lo, exec_lo'] + cnd_vcc))
    # so does s_and_saveexec with the mask as its source (exec &= the lane bit); with the mask as its destination
    # the saved exec overwrites it, a whole-mask value, and the later lane select is refused
    SPLICE.update({'s_and_saveexec_b32 s22, vcc_lo': '6a3c96be', 's_and_saveexec_b32 vcc_lo, vcc_lo': '6a3ceabe'})
    phi_text(phi(['s_mov_b32 vcc_lo, 0'], ['s_mov_b32 vcc_lo, exec_lo'], ['s_and_saveexec_b32 s22, vcc_lo'] + cnd_vcc))
    refused('definition of vcc, live where the block joins', lambda: phi_text(phi(
        ['s_mov_b32 vcc_lo, 0'], ['s_mov_b32 vcc_lo, exec_lo'], ['s_and_saveexec_b32 vcc_lo, vcc_lo'] + cnd_vcc)))
    # Negative controls: each condition removed keeps the refusal of the live definition.
    live = lambda name: 'definition of %s, live where the block joins' % name
    wr, rd = ['s_mov_b32 vcc_lo, exec_lo'], cnd_vcc
    for zero in (['s_mov_b32 vcc_lo, -1'], [], ['s_mov_b32 vcc_lo, s7']):  # zeroing with all ones, absent, not zero
        refused(live('vcc'), lambda zero=zero: phi_text(phi(zero, wr, rd)))
    # a branch target between the zeroing and the branch: the path through the label did not run the zeroing
    SPLICE['s_cbranch_execz 6'] = '060088bf'
    refused(live('vcc'), lambda: phi_text(splice(*enter, 's_cbranch_scc1 1', 's_mov_b32 vcc_lo, 0', 's_cbranch_execz 1',
                                                 *wr, *leave, *rd)))
    refused(live('vcc'), lambda: phi_text(splice(*enter, 's_mov_b32 vcc_lo, 0', 's_cbranch_scc1 0', 's_cbranch_execz 1',
                                                 *wr, *leave, *rd)))
    # no zeroing of s20 at all; a block write that is an s_and without exec; the mask read as a data word
    refused(live('s20'), lambda: phi_text(phi([], ['s_mov_b32 s20, exec_lo'], cnd_s)))
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, 0'], ['s_and_b32 vcc_lo, s7, s7'], rd)))
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, 0'], wr, ['v_cndmask_b32 v5, vcc_lo, v6, vcc_lo'])))
    # a branch between the zeroing and the branch, even one to the join; and a branch inside the block (not straight)
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, 0'], wr, rd, between=['s_cbranch_scc1 2'])))
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, 0'], wr, rd, inside=['s_cbranch_scc1 0'])))
    # a write in the block that is not from exec, or a second write, or a 64-bit write
    for inside in (['s_mov_b32 vcc_lo, -1'], ['s_mov_b32 vcc_lo, s7']):
        refused(live('vcc'), lambda inside=inside: phi_text(phi(['s_mov_b32 vcc_lo, 0'], wr, rd, inside=inside)))
    # (a vcc_hi write does not translate at all, so the 64-bit move is the wide control)
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, 0'], ['s_mov_b64 vcc, exec'], rd)))
    # whole-mask reads after the join see the other lanes' bits, which the one-lane model lacks: refused
    for read in (['s_cbranch_vccz 1', 'v_mov_b32 v5, s20'] + cnd_vcc, ['s_mov_b32 s20, vcc_lo'] + cnd_vcc):
        refused(live('vcc'), lambda read=read: phi_text(phi(['s_mov_b32 vcc_lo, 0'], wr, read)))
    refused(live('s20'), lambda: phi_text(phi(['s_mov_b32 s20, 0'], ['s_mov_b32 s20, exec_lo'],
                                              ['s_cmp_eq_u32 s20, 0'] + cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(['s_mov_b32 s20, 0'], ['s_mov_b32 s20, exec_lo'],
                                              ['s_mov_b32 s7, s20'] + cnd_s)))
    # A block that leaves the mask bit of a lane it skips unchanged is exact whatever M held before the branch:
    # `s_andn2 M, M, exec_lo` and `s_or M, M, T` with T exec-masked in the block (the compiler's i1 phi with a
    # per-lane old value). A compare into M, or an s_orn2 with exec, changes a skipped lane's bit: accepted only
    # when M was zeroed (compare) or set to all ones (orn2) before the branch.
    SPLICE.update({'s_andn2_b32 s20, s20, exec_lo': '147e148a', 's_andn2_b32 vcc_lo, vcc_lo, exec_lo': '6a7e6a8a',
                   's_and_b32 s22, s7, exec_lo': '077e1687', 's_and_b32 s22, s7, s7': '07071687',
                   's_or_b32 s20, s20, s22': '14161488', 's_or_b32 vcc_lo, vcc_lo, s22': '6a166a88',
                   's_or_b32 s20, s20, s7': '14071488', 's_andn2_b32 s20, s7, exec_lo': '077e148a',
                   's_orn2_b32 s20, s7, exec_lo': '077e148b', 'v_cmp_gt_u32 vcc_lo, 3, v0': '8300887d'})
    # T and M hold lane bits (bit 0 = the lane's bit, the model's encoding): T from a compare, M zeroed before the branch.
    # Group-id data (s7) as T, or `s_mov_b32 M, 16` as M, has the wave's lane bit at bit `lane` and is refused.
    cmp_t = ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_and_b32 s22, vcc_lo, exec_lo']
    SPLICE.update({'s_and_b32 s22, vcc_lo, exec_lo': '6a7e1687', 's_mov_b32 s20, 16': '9003 94be'.replace(' ', ''),
                   's_orn2_b32 s20, vcc_lo, exec_lo': '6a7e148b'})
    blend = ['s_andn2_b32 s20, s20, exec_lo', *cmp_t, 's_or_b32 s20, s20, s22']
    z20, zv = ['s_mov_b32 s20, 0'], ['s_mov_b32 vcc_lo, 0']
    for zero, write, read in ((z20, blend, cnd_s), (z20, blend[:1], cnd_s),
                              (zv, ['s_andn2_b32 vcc_lo, vcc_lo, exec_lo', 's_and_b32 s22, vcc_lo, exec_lo',
                                    's_or_b32 vcc_lo, vcc_lo, s22'], cnd_vcc)):
        phi_text(phi(zero, write, read))
    # ANDing a numeric word with EXEC does not turn it into this lane's projected bit.
    refused(live('s20'), lambda: phi_text(phi(['s_and_b32 s20, s7, exec_lo'], blend[:1], cnd_s)))
    phi_text(phi(['s_and_b32 s20, vcc_lo, exec_lo'], blend[:1], cnd_s))
    phi_text(phi(['s_mov_b32 vcc_lo, 0'], ['v_cmp_gt_u32 vcc_lo, 3, v0'], cnd_vcc))       # compare into a zeroed M
    phi_text(phi(['s_mov_b32 s20, -1'], ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_orn2_b32 s20, vcc_lo, exec_lo'], cnd_s))  # orn2
    # controls: the same writes without their set-up, an unmasked term, M not the first source, mixed classes,
    # M or an operand that is not a lane mask, and a whole-mask read after a preserving block
    refused(live('vcc'), lambda: phi_text(phi([], ['v_cmp_gt_u32 vcc_lo, 3, v0'], cnd_vcc)))
    refused(live('vcc'), lambda: phi_text(phi(['s_mov_b32 vcc_lo, -1'], ['v_cmp_gt_u32 vcc_lo, 3, v0'], cnd_vcc)))
    for zero in ([], z20):
        refused(live('s20'), lambda zero=zero: phi_text(phi(zero, ['s_orn2_b32 s20, s7, exec_lo'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(['s_mov_b32 s20, -1'], ['s_orn2_b32 s20, s7, exec_lo'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(['s_mov_b32 s20, -1'], ['s_mov_b32 s20, exec_lo'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(z20, ['s_mov_b32 s20, exec_lo', 's_orn2_b32 s20, s7, exec_lo'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(z20, [blend[0], *cmp_t, 's_or_b32 s20, s20, s7'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(z20, [blend[0], 's_and_b32 s22, s7, exec_lo', blend[3]], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(z20, [blend[0], 's_and_b32 s22, s7, s7', blend[3]], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi(z20, ['s_andn2_b32 s20, s7, exec_lo'], cnd_s)))
    # A compare-derived operand does not make the classes mixable: a zero write (s_mov M, exec_lo) and a ones write
    # (s_orn2 with exec) in one block are refused after either setting. `s_andn2 M, M, X` keeps M's skipped bits only for
    # X == exec_lo; with a compare mask it clears bits of lanes the model skips.
    SPLICE.update({'s_andn2_b32 s20, s20, vcc_lo': '146a148a'})
    mixed_cmp = ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_mov_b32 s20, exec_lo', 's_orn2_b32 s20, vcc_lo, exec_lo']
    for setting in (z20, ['s_mov_b32 s20, -1']):
        refused(live('s20'), lambda setting=setting: phi_text(phi(setting, mixed_cmp, cnd_s)))
    for setting in (z20, [], ['s_mov_b32 s20, -1']):
        refused(live('s20'), lambda setting=setting: phi_text(
            phi(setting, ['v_cmp_gt_u32 vcc_lo, 3, v0', 's_andn2_b32 s20, s20, vcc_lo'], cnd_s)))
    refused(live('s20'), lambda: phi_text(phi([], blend, cnd_s)))                       # M unknown before the block
    refused(live('s20'), lambda: phi_text(phi(['s_mov_b32 s20, 16'], blend, cnd_s)))    # M holds data, not lane bits
    for read in (['s_cmp_eq_u32 s20, 0'] + cnd_s, ['s_mov_b32 s7, s20'] + cnd_s):
        refused(live('s20'), lambda read=read: phi_text(phi(z20, blend, read)))
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
    # A readfirstlane of a buffer load at a uniform address translates; at a per-lane address it is refused.
    assert 's20 = v1;' in translate('N').hlsl_cs()
    refused('v_readfirstlane_b32 requires a proven uniform source', lambda: translate('N', code=WORDS_N_VARYING).hlsl_cs())
    print('compute entry, numthreads, masked bit counts, manifest, offen store forms and the store, descriptor, execz and sample '
          'refusals pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    assert assemble(llvm_mc, LOOP_BREAK_SOURCE) == LOOP_BREAK_WORDS
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
    'join-load-zero': ('join-load', 's24 = ul24_7;', 's24 = 0u;'),
    # s_min_u32 sets SCC when S0 is chosen, equal operands included: a strict compare differs at S0 == S1.
    's-min-strict': ('S', '(s20) <= (s21)', '(s20) < (s21)'),
    # s_sub_u32 borrow: the first source is the smaller one.
    's-borrow-flipped': ('S', '(s20) < (s21);\n    s24 = s20 - s21;', '(s21) < (s20);\n    s24 = s20 - s21;'),
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
    'f-second-fill': ('F-two-fills', 'lds[(v4 + 320u) >> 2] = v4;', 'lds[(v4 + 320u) >> 2] = 0u;'),
    'f-broadcast': ('F', 'lds[(v8 + 164u) >> 2]', 'lds[(v8 + 160u) >> 2]'),
    'j-scalar-dropped': ('J', '        s22 = s7 * ud1;\n', ''),
    'h-inner-unpredicated': ('H', 'if (exec & 1) u0.Store(a, v5);', 'u0.Store(a, v5);'),
    'm-else-unswitched': ('M', 'exec = (s0 & ~exec) & 1;', 'exec = (s0) & 1;'),  # the then lanes run the else arm
    'n-unmasked': ('NM', 's23 = (vcc & exec) & 1;', 's23 = (vcc) & 1;'),  # lanes 6 and 7 store with their stale vcc
    'o-mask-kept': ('O', 's0 = (s0 & ~exec) & 1;', 's0 = (s0) & 1;'),  # lanes 0-2 keep their bit and store
    'k-block-entered': ('K', 'pc = ((exec & 1) == 0) ? 2 : 1;', 'pc = 1; exec = 1;'),  # skipped waves run it
    'p-dropped-andn2': ('P', 's23 = (s23 & ~exec) & 1;', 's23 = (s23) & 1;'),  # an entering lane keeps its old bit
    'p-dropped-or': ('P', 's23 = (s23 | s22) & 1;', 's23 = (s23) & 1;'),  # an entering lane loses its compare bit
    'p-block-entered': ('P', 'pc = ((exec & 1) == 0) ? 2 : 1;', 'pc = 1; exec = 1;'),  # skipping lanes run the block
    'q-block-entered': ('Q', 'pc = ((exec & 1) == 0) ? 2 : 1;', 'pc = 1; exec = 1;'),
    'q-not-all-ones': ('Q', 's23 = 0xffffffffu;', 's23 = 0u;'),  # M not set before the branch: skipped lanes read 0
    'g-swapped-offsets': ('G', 'v6 = lds[(a + 20u) >> 2];\n    v7 = lds[(a + 144u) >> 2];',
                          'v6 = lds[(a + 144u) >> 2];\n    v7 = lds[(a + 20u) >> 2];'),
    'g-overlap': ('G', 'v7 = lds[(a + 144u)', 'v7 = lds[(v6 + 144u)'),
    'mask-shift-dropped': ('masked-tid', 's24 = (ud0 >> 2u);', 's24 = ud0;'),
    'extract-shift-dropped': ('extracted-tid', 's24 = 0u;', 's24 = v2;'),
    'n-dropped-broadcast': ('N', 'v3 = s20 ^ v2;', 'v3 = v2;'),  # lane i stores word[i], not word0 ^ word[i]
    'l-byte-offset': ('L-uint', 'a = v0 * ud2 + v1 + 2;', 'a = v0 * ud2 + 2;'),
    'l-swizzle': ('L-unorm', 'v6 = ((ud3 >> 3)', 'v6 = ((0u >> 3)'),
    'l-integer-one': ('L-uint', 'c1 = w >> 16; c2 = 0u; c3 = 1u; one = 1u;', 'c1 = w >> 16; c2 = 0u; c3 = 1u;'),
}

# First bad word for controls with a pinned mismatch.
CONTROL_MISMATCH = {
    'join-load-zero': 'word=0 expected=8b02a7f1 actual=00000000',
    'f-second-fill': 'word=1 expected=00000094 actual=00000000',
    'd-mask': 'word=8 expected=00000008 actual=00000009',
    'd-whole-mask': 'word=0 expected=00000007 actual=00000011',
    'd-x-only': 'word=16 expected=00000009 actual=00000007',
    'd-base': 'word=0 expected=00000007 actual=00000000',
    'mask-shift-dropped': 'word=0 expected=0048d159 actual=01234567',
    'extract-shift-dropped': 'word=4 expected=00000000 actual=00000001',
    'd-hi': 'word=5 expected=00000008 actual=00000009',
    'd-exec': 'word=2 expected=ffffffff actual=00000000',
    'g-swapped-offsets': 'word=0 expected=00000004 actual=00000000',
    'g-overlap': 'word=1 expected=00000000 actual=00000004',
    'l-byte-offset': 'word=0 expected=0000603e actual=00003e1f',
    'l-swizzle': 'word=5 expected=3f800000 actual=00000000',
    'l-integer-one': 'word=2 expected=00000001 actual=3f800000',
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
    source = b''
    if name in L_WORD3:
        constants[2:4] = [L_STRIDE, L_WORD3[name]]
        source = formatted_input()
    case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                     + struct.pack('<%dI' % WORDS, *expected(name)) + struct.pack('<16I', *constants) + source)
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def renderer_diagnostics(stderr, adapter):
    """Echo hardware enumeration notices and retain the remaining lines for the readback check."""
    diagnostics = []
    for line in stderr.splitlines():
        if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
            print(line, file=sys.stderr)
        else:
            diagnostics.append(line)
    return diagnostics


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
            runs.append(('join-load', 'join-load-zero'))
            runs += [(case, mutation) for mutation, (case, _, _) in CONTROLS.items() if case in L_WORD3]
            runs += [('masked-tid', 'mask-shift-dropped'), ('extracted-tid', 'extract-shift-dropped')]
        for name, mutation in runs:
            run = run_case(name, dxc, runner, adapter, Path(tmp), mutation)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = renderer_diagnostics(run.stderr, adapter)
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
