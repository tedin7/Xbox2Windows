#!/usr/bin/env python3
"""Verify bounded 128-bit LDS exchanges (ds_write_b128, ds_read_b128) and mixed 32/64/128-bit ops across s_barrier.

With no arguments, checks offline RSRC2 sizing, 4-VGPR quad validation, dword alignment/bounds, mixed 32/64/128-bit
LDS interoperability, and barrier/waitcnt/branch/EXEC/1D-group/gds/non-compute refusals.
With 'binary <llvm-mc>', validates assembled gfx1013 machine-code bytes. With 'dispatch' or 'control <name>',
compiles with DXC (cs_6_0) and runs WARP / hardware GPU readback verification via offscreen_check.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_compute_check import TGID_X, USER7, patch, refused, root_blob  # noqa: E402


PREAMBLE = (
    's_mov_b32 s10, s6\n'
    's_mov_b32 s11, 0\n'
    's_load_dwordx8 s[12:19], s[10:11], 0x0\n'
    's_waitcnt lgkmcnt(0)\n'
)

# Case Q: [numthreads(32, 1, 1)], groups=(2, 1, 1), LDS_SIZE=2 (1024 bytes = 256 uints).
# 128-bit peer exchange (X ^ 17) at byte offset 16..528, including vdst[0] == vaddr aliasing (ds_read_b128 v[9:12], v9 offset:16).
SOURCE_Q = PREAMBLE + """\
s_lshl_b32 s20, s7, 8
v_add_nc_u32 v5, s20, v0
v_xor_b32 v6, 0xa5a50000, v5
v_add_nc_u32 v7, 0x5a5a0000, v5
v_not_b32 v8, v5
v_lshlrev_b32 v4, 4, v0
ds_write_b128 v4, v[5:8] offset:16
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v9, 0x110, v4
v_and_b32 v9, 0x1f0, v9
ds_read_b128 v[9:12], v9 offset:16
s_waitcnt lgkmcnt(0)
s_lshl_b32 s21, s7, 5
v_add_nc_u32 v13, s21, v0
v_lshlrev_b32 v13, 4, v13
buffer_store_dwordx4 v[9:12], v13, s[16:19], 0 offen
s_endpgm
"""

WORDS_Q = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '0788148f14000a4aff0a0c3a0000a5a5ff0a0e4a00005a5a056f107e84000834'
    '10007cdb040500007fc08cbf00008abfff08123a10010000'
    'ff121236f00100001000fcdb090000097fc08cbf'
    '0785158f15001a4a841a1a34001078e00d090480000081bf'
)

# Case B: [numthreads(64, 1, 1)], groups=(1, 1, 1), LDS_SIZE=2 (1024 bytes exact fit: 64 * 16 = 1024).
# Cross-wave peer exchange (X ^ 32) with vdst[1] == vaddr aliasing (ds_read_b128 v[9:12], v10)
# plus lane-37 128-bit broadcast at offset:592 (combined peer low qword + broadcast high qword).
SOURCE_B = PREAMBLE + """\
v_mov_b32 v5, v0
v_xor_b32 v6, 0x5a5a0000, v5
v_add_nc_u32 v7, 0x11223300, v5
v_not_b32 v8, v5
v_lshlrev_b32 v4, 4, v0
ds_write_b128 v4, v[5:8]
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v10, 0x200, v4
v_and_b32 v10, 0x3f0, v10
ds_read_b128 v[9:12], v10
s_waitcnt lgkmcnt(0)
v_mov_b32 v17, 0
ds_read_b128 v[13:16], v17 offset:592
s_waitcnt lgkmcnt(0)
v_mov_b32 v11, v15
v_mov_b32 v12, v16
v_lshlrev_b32 v18, 4, v0
buffer_store_dwordx4 v[9:12], v18, s[16:19], 0 offen
s_endpgm
"""

WORDS_B = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '00030a7eff0a0c3a00005a5aff0a0e4a00332211056f107e84000834'
    '00007cdb040500007fc08cbf00008abfff08143a00020000'
    'ff141436f00300000000fcdb0a0000097fc08cbf'
    '8002227e5002fcdb1100000d7fc08cbf0f03167e1003187e'
    '84002434001078e012090480000081bf'
)

# Case M: [numthreads(32, 1, 1)], groups=(1, 1, 1), LDS_SIZE=1 (512 bytes exact fit: 32 * 16 = 512).
# Mixed 128-bit write -> 128-bit peer read (X ^ 16, vdst[1] == vaddr) + 64-bit read (lane 18 dwords 2..3, offset:296)
# + 32-bit reads (lane 31 dword 3 at offset:508, lane 0 dword 0 at offset:0) sharing groupshared uint lds[128].
SOURCE_M = PREAMBLE + """\
v_mov_b32 v5, v0
v_xor_b32 v6, 0x55aa0000, v5
v_add_nc_u32 v7, 0x12340000, v5
v_not_b32 v8, v5
v_lshlrev_b32 v4, 4, v0
ds_write_b128 v4, v[5:8]
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v10, 0x100, v4
v_and_b32 v10, 0x1f0, v10
ds_read_b128 v[9:12], v10
s_waitcnt lgkmcnt(0)
v_mov_b32 v17, 0
ds_read_b64 v[13:14], v17 offset:296
s_waitcnt lgkmcnt(0)
v_mov_b32 v17, 0
ds_read_b32 v15, v17 offset:508
s_waitcnt lgkmcnt(0)
v_mov_b32 v17, 0
ds_read_b32 v16, v17
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v18, 5, v0
buffer_store_dwordx4 v[9:12], v18, s[16:19], 0 offen
buffer_store_dwordx4 v[13:16], v18, s[16:19], 0 offen offset:16
s_endpgm
"""

WORDS_M = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '00030a7eff0a0c3a0000aa55ff0a0e4a00003412056f107e84000834'
    '00007cdb040500007fc08cbf00008abfff08143a00010000'
    'ff141436f00100000000fcdb0a0000097fc08cbf'
    '8002227e2801d8d91100000d7fc08cbf'
    '8002227efc01d8d81100000f7fc08cbf'
    '8002227e0000d8d8110000107fc08cbf'
    '85002434001078e012090480101078e0120d0480000081bf'
)

# Case R: [numthreads(32, 1, 1)], groups=(2, 1, 1), LDS_SIZE=1 (512 bytes = 128 uints).
# Mixed 32-bit fill (offset:16..144) -> 128-bit quad-dword read of 4 consecutive peer lanes ((X ^ 16) & ~3).
SOURCE_R = PREAMBLE + """\
s_lshl_b32 s20, s7, 8
v_add_nc_u32 v5, s20, v0
v_xor_b32 v5, 0xa5000000, v5
v_lshlrev_b32 v4, 2, v0
ds_write_b32 v4, v5 offset:16
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v9, 64, v4
v_and_b32 v9, 0x70, v9
ds_read_b128 v[9:12], v9 offset:16
s_waitcnt lgkmcnt(0)
s_lshl_b32 s21, s7, 5
v_add_nc_u32 v13, s21, v0
v_lshlrev_b32 v13, 4, v13
buffer_store_dwordx4 v[9:12], v13, s[16:19], 0 offen
s_endpgm
"""

WORDS_R = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '0788148f14000a4aff0a0a3a000000a582000834100034d804050000'
    '7fc08cbf00008abfc008123aff121236700000001000fcdb09000009'
    '7fc08cbf0785158f15001a4a841a1a34001078e00d090480000081bf'
)

# Two distinct qwords per lane, with a nonzero base and mirrored reads.
SOURCE_TWO_QWORDS = PREAMBLE + """\
v_add_nc_u32 v18, 61, v0
v_xor_b32 v19, 0x13570000, v18
v_add_nc_u32 v20, 97, v0
v_not_b32 v21, v20
v_lshlrev_b32 v22, 4, v0
ds_write_b64 v22, v[18:19] offset:16
ds_write_b64 v22, v[20:21] offset:24
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v23, 31, v0
v_lshlrev_b32 v23, 4, v23
v_and_b32 v23, 0x1f0, v23
ds_read_b64 v[24:25], v23 offset:16
s_waitcnt lgkmcnt(0)
v_and_b32 v23, 0x1f0, v23
ds_read_b64 v[26:27], v23 offset:24
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v28, 4, v0
buffer_store_dwordx4 v[24:27], v28, s[16:19], 0 offen
s_endpgm
"""
SOURCE_OVERLAP = SOURCE_TWO_QWORDS.replace('v22, 4, v0', 'v22, 3, v0')
SOURCE_UNPROVEN = SOURCE_TWO_QWORDS.replace('ds_write_b64 v22, v[20:21]', 'ds_write_b64 v19, v[20:21]')

WORDS_TWO_QWORDS = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbfbd00244aff24263a00005713ff00284a'
    '61000000146f2a7e84002c34100034d916120000180034d9161400007fc08cbf00008abf'
    '9f002e3a842e2e34ff2e2e36f00100001000d8d9170000187fc08cbfff2e2e36f0010000'
    '1800d8d91700001a7fc08cbf84003834001078e01c180480000081bf'
)
WORDS_OVERLAP = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbfbd00244aff24263a00005713ff00284a'
    '61000000146f2a7e83002c34100034d916120000180034d9161400007fc08cbf00008abf'
    '9f002e3a842e2e34ff2e2e36f00100001000d8d9170000187fc08cbfff2e2e36f0010000'
    '1800d8d91700001a7fc08cbf84003834001078e01c180480000081bf'
)
WORDS_UNPROVEN = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbfbd00244aff24263a00005713ff00284a'
    '61000000146f2a7e84002c34100034d916120000180034d9131400007fc08cbf00008abf'
    '9f002e3a842e2e34ff2e2e36f00100001000d8d9170000187fc08cbfff2e2e36f0010000'
    '1800d8d91700001a7fc08cbf84003834001078e01c180480000081bf'
)

CASES = {
    'two-qwords': (SOURCE_TWO_QWORDS, WORDS_TWO_QWORDS, USER7 | TGID_X | (2 << 15), (32, 1, 1), (1, 1, 1)),
    'Q': (SOURCE_Q, WORDS_Q, USER7 | TGID_X | (2 << 15), (32, 1, 1), (2, 1, 1)),
    'B': (SOURCE_B, WORDS_B, USER7 | TGID_X | (2 << 15), (64, 1, 1), (1, 1, 1)),
    'M': (SOURCE_M, WORDS_M, USER7 | TGID_X | (1 << 15), (32, 1, 1), (1, 1, 1)),
    'R': (SOURCE_R, WORDS_R, USER7 | TGID_X | (1 << 15), (32, 1, 1), (2, 1, 1)),
}

VERIFIED_SNIPPETS = {
    'ds_write_b128 v4, v[5:8]': '00007cdb04050000',
    'ds_write_b128 v4, v[5:8] offset:16': '10007cdb04050000',
    'ds_write_b128 v2, v[8:11] offset:32': '20007cdb02080000',
    'ds_read_b128 v[24:27], v2': '0000fcdb02000018',
    'ds_read_b128 v[9:12], v9 offset:16': '1000fcdb09000009',
    'ds_read_b128 v[13:16], v17 offset:592': '5002fcdb1100000d',
}

WORDS = 320
SENTINEL = 0xc5c5c5c5


def expected_words(name):
    out = [SENTINEL] * WORDS
    if name == 'two-qwords':
        for lane in range(32):
            peer = 31 - lane
            first, second = peer + 61, peer + 97
            out[lane * 4:lane * 4 + 4] = [first, first ^ 0x13570000, second, 0xffffffff ^ second]
    elif name == 'Q':
        for g in range(2):
            for l in range(32):
                peer = (g << 8) + (l ^ 17)
                idx = (g * 32 + l) * 4
                out[idx] = peer
                out[idx + 1] = peer ^ 0xa5a50000
                out[idx + 2] = (0x5a5a0000 + peer) & 0xffffffff
                out[idx + 3] = peer ^ 0xffffffff
    elif name == 'B':
        b2 = (0x11223300 + 37) & 0xffffffff
        b3 = 37 ^ 0xffffffff
        for l in range(64):
            peer = l ^ 32
            idx = l * 4
            out[idx] = peer
            out[idx + 1] = peer ^ 0x5a5a0000
            out[idx + 2] = b2
            out[idx + 3] = b3
    elif name == 'M':
        l18_w2 = (0x12340000 + 18) & 0xffffffff
        l18_w3 = 18 ^ 0xffffffff
        l31_w3 = 31 ^ 0xffffffff
        l0_w0 = 0
        for l in range(32):
            peer = l ^ 16
            idx = l * 8
            out[idx] = peer
            out[idx + 1] = peer ^ 0x55aa0000
            out[idx + 2] = (0x12340000 + peer) & 0xffffffff
            out[idx + 3] = peer ^ 0xffffffff
            out[idx + 4] = l18_w2
            out[idx + 5] = l18_w3
            out[idx + 6] = l31_w3
            out[idx + 7] = l0_w0
    elif name == 'R':
        for g in range(2):
            for l in range(32):
                base_lane = (l ^ 16) & ~3
                idx = (g * 32 + l) * 4
                for k in range(4):
                    out[idx + k] = ((g << 8) + base_lane + k) ^ 0xa5000000
    else:
        raise KeyError(name)
    return out


def translate_case(name, code=None, rsrc2=None, group=None):
    _, words, default_rsrc2, local, _groups = CASES[name]
    return G.ComputePipeline(words if code is None else code, root_blob(),
                             default_rsrc2 if rsrc2 is None else rsrc2, local if group is None else group)



def legacy_check_case(name, code=None, rsrc2=None, group=None):
    # These refusal controls exercise the legacy proof independently of the general retry.
    _, words, default_rsrc2, local, _groups = CASES[name]
    raw = words if code is None else code
    resource = default_rsrc2 if rsrc2 is None else rsrc2
    G.check_lds_exchange(G.decode(G.strip_header(raw)), local if group is None else group,
                         ((resource >> 15) & 0x1ff) * 128)
    return translate_case(name, code=code, rsrc2=rsrc2, group=group)

def check_strided_layouts():
    def scan(source):
        ins = [(i * 4, 4, asm.split()[0], asm) for i, asm in enumerate(source.splitlines())]
        return G.check_lds_exchange(ins, (32, 1, 1), 512)

    # Three slots in a 32-byte record. Compute the middle slot's offset in ALU,
    # and leave the final qword in each record uninitialized.
    gapped = SOURCE_TWO_QWORDS.replace('v22, 4, v0', 'v22, 5, v0').replace(
        'ds_write_b64 v22, v[20:21] offset:24',
        'v_lshl_add_u32 v32, v0, 5, 24\nds_write_b64 v32, v[20:21]\n'
        'ds_write_b64 v22, v[18:19] offset:32')
    gapped = gapped.replace('v23, 4, v23', 'v23, 5, v23').replace('0x1f0', '0x3e0')
    assert scan(gapped) == 258
    refused('initialized span', lambda: scan(gapped.replace('v23 offset:24', 'v23 offset:40')))
    refused('one bounded qword', lambda: scan(gapped.replace('offset:32', 'offset:48')))
    refused('one bounded qword', lambda: scan(gapped.replace('offset:32', 'offset:28')))
    multiplied = gapped.replace('v_lshlrev_b32 v22, 5, v0', 'v_mul_u32_u24 v22, 32, v0')
    assert scan(multiplied) == 258


def check_controls():
    check_strided_layouts()
    hlsl = translate_case('two-qwords').hlsl_cs()
    assert 'groupshared uint lds[132];' in hlsl
    assert mutate(hlsl, 'qwords-overlap')
    for code in (WORDS_OVERLAP, WORDS_UNPROVEN):
        refused('LDS exchange needs one bounded qword per local X ID',
                lambda: legacy_check_case('two-qwords', code=code).hlsl_cs())
    refused('LDS exchange needs one bounded qword per local X ID',
            lambda: legacy_check_case('two-qwords', rsrc2=USER7 | TGID_X | (1 << 15)).hlsl_cs())
    # Positive HLSL lowering: declarations, single-eval alias-safe index, barrier, and all 4 dwords.
    hlsl_q = translate_case('Q').hlsl_cs()
    assert 'groupshared uint lds[256];' in hlsl_q, hlsl_q
    assert hlsl_q.count('GroupMemoryBarrierWithGroupSync();') == 1, hlsl_q
    assert ('a = (v4 + 16u) >> 2;\n    lds[a] = v5;\n    lds[a + 1u] = v6;\n'
            '    lds[a + 2u] = v7;\n    lds[a + 3u] = v8;') in hlsl_q, hlsl_q
    assert ('a = (v9 + 16u) >> 2;\n    v9 = lds[a];\n    v10 = lds[a + 1u];\n'
            '    v11 = lds[a + 2u];\n    v12 = lds[a + 3u];') in hlsl_q, hlsl_q

    hlsl_b = translate_case('B').hlsl_cs()
    assert 'groupshared uint lds[256];' in hlsl_b, hlsl_b
    assert ('a = (v4 + 0u) >> 2;\n    lds[a] = v5;\n    lds[a + 1u] = v6;\n'
            '    lds[a + 2u] = v7;\n    lds[a + 3u] = v8;') in hlsl_b, hlsl_b
    assert ('a = (v10 + 0u) >> 2;\n    v9 = lds[a];\n    v10 = lds[a + 1u];\n'
            '    v11 = lds[a + 2u];\n    v12 = lds[a + 3u];') in hlsl_b, hlsl_b
    assert ('a = (v17 + 592u) >> 2;\n    v13 = lds[a];\n    v14 = lds[a + 1u];\n'
            '    v15 = lds[a + 2u];\n    v16 = lds[a + 3u];') in hlsl_b, hlsl_b

    # Mixed 32/64/128-bit LDS lowering sharing groupshared uint lds[128].
    hlsl_m = translate_case('M').hlsl_cs()
    assert 'groupshared uint lds[128];' in hlsl_m, hlsl_m
    assert 'a = (v17 + 296u) >> 2;\n    v13 = lds[a];\n    v14 = lds[a + 1u];' in hlsl_m, hlsl_m
    assert 'v15 = lds[(v17 + 508u) >> 2];' in hlsl_m, hlsl_m
    assert 'v16 = lds[(v17 + 0u) >> 2];' in hlsl_m, hlsl_m

    hlsl_r = translate_case('R').hlsl_cs()
    assert 'groupshared uint lds[128];' in hlsl_r, hlsl_r
    assert 'lds[(v4 + 16u) >> 2] = v5;' in hlsl_r, hlsl_r
    assert ('a = (v9 + 16u) >> 2;\n    v9 = lds[a];\n    v10 = lds[a + 1u];\n'
            '    v11 = lds[a + 2u];\n    v12 = lds[a + 3u];') in hlsl_r, hlsl_r

    # vdst[3] == vaddr aliasing and nonzero dword-aligned non-16B offset (offset:4) on ds_write_b128 / ds_read_b128.
    alias_hi = patch(WORDS_B, '0000fcdb0a000009', '0000fcdb0c000009')
    alias_hi = patch(alias_hi, 'ff08143a00020000', 'ff08183a00020000')
    alias_hi = patch(alias_hi, 'ff141436f0030000', 'ff181836f0030000')
    hlsl_alias = translate_case('B', code=alias_hi).hlsl_cs()
    assert 'a = (v12 + 0u) >> 2;\n    v9 = lds[a];' in hlsl_alias, hlsl_alias

    # Dword-aligned (non-16B) offset:4 fill and read on ds_write_b128 / ds_read_b128.
    store_q = '10007cdb04050000'
    read_q = '1000fcdb09000009'
    dword_off = patch(WORDS_Q, store_q, '04007cdb04050000')
    dword_off = patch(dword_off, read_q, '0400fcdb09000009')
    hlsl_dw = translate_case('Q', code=dword_off).hlsl_cs()
    assert 'a = (v4 + 4u) >> 2;' in hlsl_dw and 'a = (v9 + 4u) >> 2;' in hlsl_dw, hlsl_dw

    # Tight capacity boundaries:
    # Case Q: offset:16 + 32*16 = 528 exceeds 1 unit (512B), fits at 2 units and at 32 KiB.
    # Case B: offset:0 + 64*16 = 1024 exact fit at 2 units; offset:4 (1028B) refuses at 2 units.
    base = USER7 | TGID_X
    refused('one bounded quad-dword', lambda: legacy_check_case('Q', rsrc2=base | (1 << 15)))
    refused('allocation', lambda: legacy_check_case('Q', rsrc2=base | (0 << 15)))
    refused('allocation', lambda: legacy_check_case('Q', rsrc2=base | (65 << 15)))
    assert 'groupshared uint lds[8192];' in translate_case('Q', rsrc2=base | (64 << 15)).hlsl_cs()
    refused('one bounded quad-dword',
            lambda: legacy_check_case('B', code=patch(WORDS_B, '00007cdb04050000', '04007cdb04050000')))

    # Mask alignment and +12/+16 boundary checks on ds_read_b128.
    mask_1f0 = 'ff121236f0010000'
    mask_1c = 'ff1212361c000000'
    mask_1e = 'ff1212361e000000'
    mask_1fc = 'ff121236fc010000'
    ok_1c = translate_case('Q', code=patch(WORDS_Q, mask_1f0, mask_1c)).hlsl_cs()
    assert 'v9 = 0x0000001cu & v9;' in ok_1c, ok_1c
    refused('aligned', lambda: legacy_check_case('Q', code=patch(WORDS_Q, mask_1f0, mask_1e)))
    refused('initialized span', lambda: legacy_check_case('Q', code=patch(WORDS_Q, mask_1f0, mask_1fc)))

    # Broadcast upper boundary: B fills [0, 1024), so offset:1008 reads [1008, 1024) (valid), offset:1012 overflows.
    bcast_b = '5002fcdb1100000d'
    ok_1008 = translate_case('B', code=patch(WORDS_B, bcast_b, 'f003fcdb1100000d')).hlsl_cs()
    assert 'a = (v17 + 1008u) >> 2;' in ok_1008, ok_1008
    refused('initialized span', lambda: legacy_check_case('B', code=patch(WORDS_B, bcast_b, 'f403fcdb1100000d')))

    # Mixed-width boundary checks:
    # - Case R fills [16, 144) via ds_write_b32; ds_read_b128 with mask 0x70 + offset:16 covers [16, 144).
    #   Mask 0x74 reaches [16, 148) -> refused.
    refused('initialized span', lambda: legacy_check_case('R', code=patch(WORDS_R, 'ff12123670000000', 'ff12123674000000')))
    # - Case M fills [0, 512) via ds_write_b128; ds_read_b64 at offset:504 fits, offset:508 refuses.
    bcast_m64 = '2801d8d91100000d'
    assert 'a = (v17 + 504u) >> 2;' in translate_case('M', code=patch(WORDS_M, bcast_m64, 'f801d8d91100000d')).hlsl_cs()
    refused('initialized span', lambda: legacy_check_case('M', code=patch(WORDS_M, bcast_m64, 'fc01d8d91100000d')))
    # - Case M with ds_read2_b32 inside [0, 512) succeeds; out-of-span offset1 refuses.
    ok_read2 = translate_case('M', code=patch(WORDS_M, bcast_m64, '047fdcd81100000d')).hlsl_cs()  # offset0:4 (16), offset1:127 (508)
    assert 'v13 = lds[(a + 16u) >> 2];\n    v14 = lds[(a + 508u) >> 2];' in ok_read2, ok_read2
    refused('initialized span', lambda: legacy_check_case('M', code=patch(WORDS_M, bcast_m64, '0480dcd81100000d')))

    # Additional mixed b64 fill -> b128 read positive + out-of-span refusal.
    b64_to_b128 = [
        (0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 3, v0'),
        (4, 8, 'ds_write_b64', 'ds_write_b64 v4, v[5:6] offset:8'),
        (12, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
        (16, 4, 's_barrier', 's_barrier 0x0'),
        (20, 4, 'v_mov_b32', 'v_mov_b32 v8, 0'),
        (24, 8, 'ds_read_b128', 'ds_read_b128 v[8:11], v8 offset:24'),
        (32, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
        (36, 4, 's_endpgm', 's_endpgm'),
    ]
    G.check_lds_exchange(b64_to_b128, (4, 1, 1), 128)  # [8, 40): read [24, 40) fits
    # Each limit has its own refusal naming the measured value (a census says which one the title exceeds).
    refused('needs a 1D group (the group is 8x4x1)', lambda: G.check_lds_exchange(b64_to_b128, (8, 4, 1), 128))
    refused('needs a 1D group (the group is 4x1x2)', lambda: G.check_lds_exchange(b64_to_b128, (4, 1, 2), 128))
    refused('needs a nonzero LDS allocation', lambda: G.check_lds_exchange(b64_to_b128, (4, 1, 1), 0))
    refused('at most 32 KiB (COMPUTE_PGM_RSRC2 allocates 33 KiB)', lambda: G.check_lds_exchange(b64_to_b128, (4, 1, 1), 8448))
    bad_b64_to_b128 = list(b64_to_b128)
    bad_b64_to_b128[5] = (24, 8, 'ds_read_b128', 'ds_read_b128 v[8:11], v8 offset:28')  # [28, 44) overflows 40
    refused('initialized span', lambda: G.check_lds_exchange(bad_b64_to_b128, (4, 1, 1), 128))

    # Unaligned offsets, wrong shift, missing barrier/waitcnt, double fill/barrier, unprovable address, gds, EXEC/branch/non-1D.
    wait = '7fc08cbf'
    barrier = '00008abf'
    nop = '000080bf'
    refused('one bounded quad-dword', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, '12007cdb04050000')))  # write offset:18
    refused('one bounded quad-dword', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, '11007cdb04050000')))  # write offset:17
    refused('aligned', lambda: legacy_check_case('Q', code=patch(WORDS_Q, read_q, '1200fcdb09000009')))                 # read offset:18
    refused('initialized span', lambda: legacy_check_case('Q', code=patch(WORDS_Q, read_q, '0c00fcdb09000009')))        # before span (offset:12)
    refused('one bounded quad-dword', lambda: legacy_check_case('Q', code=patch(WORDS_Q, '84000834', '83000834')))      # shift 3 instead of 4
    refused('one bounded quad-dword', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, store_q + store_q)))  # double fill
    refused('completed fill', lambda: legacy_check_case('Q', code=patch(WORDS_Q, wait + barrier, nop + barrier)))
    refused('after the fill barrier', lambda: legacy_check_case('Q', code=patch(WORDS_Q, barrier, nop)))
    refused('one barrier', lambda: legacy_check_case('Q', code=patch(WORDS_Q, barrier, barrier + barrier)))
    refused('completed LGKM wait', lambda: legacy_check_case('Q', code=patch(WORDS_Q, read_q + wait, read_q + nop)))
    refused('optional byte offset', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, '10007edb04050000')))  # write GDS
    refused('optional byte offset', lambda: legacy_check_case('Q', code=patch(WORDS_Q, read_q, '1000fedb09000009')))   # read GDS
    refused('literal or literal-masked',
            lambda: legacy_check_case('Q', code=patch(WORDS_Q, 'ff121236f0010000', '0403127e')))                        # unbounded v_mov v9, v4
    refused('after the fill barrier',
            lambda: legacy_check_case('Q', code=patch(WORDS_Q, 'ff121236f0010000', nop)))                               # non-producer s_nop before read
    refused('entry EXEC', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, '8003febe' + store_q)))
    refused('straight-line', lambda: legacy_check_case('Q', code=patch(WORDS_Q, store_q, '020084bf' + store_q)))
    refused('1D group', lambda: legacy_check_case('Q', group=(16, 2, 1)))
    refused('completed read', lambda: G.check_lds_exchange(b64_to_b128[:4] + [b64_to_b128[-1]], (4, 1, 1), 128))
    refused('does not cover', lambda: G.check_lds_exchange([
        (0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 4, v0'),
        (4, 8, 'ds_write_b96', 'ds_write_b96 v4, v[5:7]'),
        (12, 4, 's_endpgm', 's_endpgm'),
    ], (4, 1, 1), 128))

    # 4-VGPR quad validation and >0xffff offset refusal on ds_write_b128 / ds_read_b128.
    assert G._lds_quad('v[0:3]') == 0
    assert G._lds_quad('v[4:7]') == 4
    assert G._lds_quad('v[252:255]') == 252
    for bad_quad in ('v5', 'v[5:5]', 'v[5:6]', 'v[5:7]', 'v[5:9]', 'v[8:5]', 'v[253:256]', 's[4:7]'):
        assert G._lds_quad(bad_quad) is None, bad_quad
        decoded_w = [(0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 4, v0'),
                     (4, 8, 'ds_write_b128', f'ds_write_b128 v4, {bad_quad}'),
                     (12, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (16, 4, 's_barrier', 's_barrier 0x0'),
                     (20, 4, 'v_mov_b32', 'v_mov_b32 v8, 0'),
                     (24, 8, 'ds_read_b128', 'ds_read_b128 v[8:11], v8'),
                     (32, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (36, 4, 's_endpgm', 's_endpgm')]
        refused('4-VGPR quad', lambda d=decoded_w: G.check_lds_exchange(d, (4, 1, 1), 128))
        decoded_r = [(0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 4, v0'),
                     (4, 8, 'ds_write_b128', 'ds_write_b128 v4, v[5:8]'),
                     (12, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (16, 4, 's_barrier', 's_barrier 0x0'),
                     (20, 4, 'v_mov_b32', 'v_mov_b32 v8, 0'),
                     (24, 8, 'ds_read_b128', f'ds_read_b128 {bad_quad}, v8'),
                     (32, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (36, 4, 's_endpgm', 's_endpgm')]
        refused('4-VGPR quad', lambda d=decoded_r: G.check_lds_exchange(d, (4, 1, 1), 128))

    bad_off = list(b64_to_b128)
    bad_off[5] = (24, 8, 'ds_read_b128', 'ds_read_b128 v[8:11], v8 offset:65536')
    refused('optional byte offset', lambda: G.check_lds_exchange(bad_off, (4, 1, 1), 128))

    # Non-compute stage and missing pipe metadata refusals for b128 DS ops.
    for stage_kind in ('vs', 'ps'):
        s = G.Stage.__new__(G.Stage)
        s.kind = stage_kind
        refused('ds_write_b128', lambda st=s: st.lower('ds_write_b128', 'ds_write_b128 v4, v[5:8]', ['v4', 'v[5:8]'], []))
        refused('ds_read_b128', lambda st=s: st.lower('ds_read_b128', 'ds_read_b128 v[8:11], v8', ['v[8:11]', 'v8'], []))

    bare = G.Stage.__new__(G.Stage)
    bare.kind = 'cs'
    bare.pipe = None
    refused('metadata', lambda: bare.lower('ds_write_b128', 'ds_write_b128 v4, v[5:8]', ['v4', 'v[5:8]'], []))

    print('bounded 128-bit LDS exchange sizing, quad validation, mixed 32/64/128-bit interop, bounds and refusals pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    for name, (source, words, *_rest) in CASES.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for source, words in ((SOURCE_OVERLAP, WORDS_OVERLAP), (SOURCE_UNPROVEN, WORDS_UNPROVEN)):
        assert assemble(llvm_mc, source) == words
    for line, hex_words in VERIFIED_SNIPPETS.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == hex_words, '%s: assembled %s, recorded %s' % (line, built, hex_words)
    print('128-bit LDS fixture words match llvm-mc (%d programs, %d snippets)' % (len(CASES), len(VERIFIED_SNIPPETS)))


CONTROLS = {
    'qwords-overlap': ('two-qwords', '    v22 = v0 << (0x00000004u & 31);\n', '    v22 = v0 << (0x00000003u & 31);\n'),
    'q-same-lane': ('Q', '    v9 = 0x00000110u ^ v4;\n', '    v9 = v4;\n'),
    'q-swap-dwords': ('Q', '    v11 = lds[a + 2u];\n    v12 = lds[a + 3u];\n',
                      '    v11 = lds[a + 3u];\n    v12 = lds[a + 2u];\n'),
    'q-drop-fourth': ('Q', '    lds[a + 3u] = v8;\n', ''),
    'b-wrong-mask': ('B', '    v10 = 0x000003f0u & v10;\n', '    v10 = 0x000001f0u & v10;\n'),
    'm-wrong-b64-off': ('M', '    a = (v17 + 296u) >> 2;\n', '    a = (v17 + 288u) >> 2;\n'),
    'r-wrong-mask': ('R', '    v9 = 0x00000070u & v9;\n', '    v9 = 0x00000030u & v9;\n'),
}


def mutate(hlsl, control):
    _, old, new = CONTROLS[control]
    assert hlsl.count(old) == 1, (control, hlsl)
    hlsl = hlsl.replace(old, new)
    if control == 'qwords-overlap':
        # Order the two write rounds so the second slot deterministically overwrites
        # its adjacent lane's first slot. Read at the same reduced stride.
        second_store = '    a = (v22 + 24u) >> 2;\n'
        shift = '    v23 = v23 << (0x00000004u & 31);\n'
        mask = '    v23 = 0x000001f0u & v23;\n'
        assert hlsl.count(second_store) == 1 and hlsl.count(shift) == 1 and hlsl.count(mask) == 2
        hlsl = hlsl.replace(second_store, '    GroupMemoryBarrierWithGroupSync();\n' + second_store)
        hlsl = hlsl.replace(shift, shift.replace('4u', '3u')).replace(mask, mask.replace('01f0', '00f8'))
    return hlsl


def run_case(name, dxc, runner, adapter, directory, control=None):
    hlsl = translate_case(name).hlsl_cs()
    if control:
        hlsl = mutate(hlsl, control)
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    groups = CASES[name][4]
    constants = [0] * 16
    case.write_bytes(
        struct.pack('<I', groups[0] | (groups[1] << 8) | (groups[2] << 16))
        + struct.pack('<%dI' % WORDS, *expected_words(name))
        + struct.pack('<16I', *constants)
    )
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name in ((CONTROLS[control][0],) if control else CASES):
            run = run_case(name, dxc, runner, adapter, Path(tmp), control)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = []
            for line in run.stderr.splitlines():
                if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
                    print(line, file=sys.stderr)
                else:
                    diagnostics.append(line)
            if control:
                if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                    raise AssertionError('control %s: not rejected (exit %d) %s' % (control, run.returncode, run.stderr))
                print('control %s: rejected at %s' % (control, diagnostics[0]))
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
