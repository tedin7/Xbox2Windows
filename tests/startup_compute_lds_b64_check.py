#!/usr/bin/env python3
"""Verify bounded 64-bit LDS exchanges (ds_write_b64, ds_read_b64) across s_barrier in tools/startup/gfx2hlsl.py.

With no arguments, checks offline RSRC2 sizing, pair alignment/bounds, & 0x1c vs & 0x1e masks, +4 overflow,
mixed b32/b64 refusal, register-range refusals, barrier/waitcnt/branch/EXEC/1D-group/non-compute refusals.
With 'binary <llvm-mc>', validates assembled gfx1013 machine-code bytes. With 'dispatch' or 'control <name>',
compiles with DXC (cs_6_0) and runs WARP / hardware GPU readback verification via offscreen_check.
"""

import re
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

import gfx2hlsl as G
from startup_gfx2hlsl_compute_check import TGID_X, USER7, patch, refused, renderer_diagnostics, root_blob


def le32(w):
    return w.to_bytes(4, 'little')


def sop2(op, sdst, ssrc0, ssrc1):
    return le32(0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def sopp(op, simm=0):
    return le32(0xbf800000 | ((op & 0x7f) << 16) | (simm & 0xffff))


def vop1(op, vdst, src0, lit=None):
    b = le32(0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def vop2(op, vdst, src0, vsrc1, lit=None):
    b = le32(((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def ds_enc(op, addr, data0=0, vdst=0, offset=0, gds=0):
    w0 = 0xd8000000 | ((op & 0xff) << 18) | ((gds & 1) << 17) | (offset & 0xffff)
    w1 = ((vdst & 0xff) << 24) | ((data0 & 0xff) << 8) | (addr & 0xff)
    return le32(w0) + le32(w1)


PREAMBLE = (
    's_mov_b32 s10, s6\n'
    's_mov_b32 s11, 0\n'
    's_load_dwordx8 s[12:19], s[10:11], 0x0\n'
    's_waitcnt lgkmcnt(0)\n'
)

# Case P: [numthreads(64, 1, 1)], groups=(2, 1, 1), LDS_SIZE=2 (1024 bytes = 256 uints).
# 64-bit peer exchange (X ^ 32) across the two wave32 halves at byte offset 16..528,
# including vdst[0] == vaddr aliasing (ds_read_b64 v[8:9], v8 offset:16).
SOURCE_P = PREAMBLE + """\
s_lshl_b32 s20, s7, 8
v_add_nc_u32 v5, s20, v0
v_xor_b32 v6, 0xa5a50000, v5
v_lshlrev_b32 v4, 3, v0
ds_write_b64 v4, v[5:6] offset:16
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v8, 0x100, v4
v_and_b32 v8, 0x1f8, v8
ds_read_b64 v[8:9], v8 offset:16
s_waitcnt lgkmcnt(0)
s_lshl_b32 s21, s7, 6
v_add_nc_u32 v13, s21, v0
v_lshlrev_b32 v13, 3, v13
buffer_store_dwordx2 v[8:9], v13, s[16:19], 0 offen
s_endpgm
"""

WORDS_P = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '0788148f14000a4aff0a0c3a0000a5a583000834'
    '100034d9040500007fc08cbf00008abfff08103a00010000'
    'ff101036f80100001000d8d9080000087fc08cbf'
    '0786158f15001a4a831a1a34001074e00d080480000081bf'
)

# Case B: [numthreads(64, 1, 1)], groups=(1, 1, 1), LDS_SIZE=1 (512 bytes exact fit: 64 * 8 = 512).
# Outputs full 4-dword result (peer qword v[8:9] + lane-37 broadcast qword v[10:11] at offset 296).
SOURCE_B = PREAMBLE + """\
s_lshl_b32 s20, s7, 8
v_add_nc_u32 v5, s20, v0
v_xor_b32 v6, 0x5a5a0000, v5
v_lshlrev_b32 v4, 3, v0
ds_write_b64 v4, v[5:6]
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v8, 0x100, v4
v_and_b32 v8, 0x1f8, v8
ds_read_b64 v[8:9], v8
s_waitcnt lgkmcnt(0)
v_mov_b32 v12, 0
ds_read_b64 v[10:11], v12 offset:296
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v13, 4, v0
buffer_store_dwordx4 v[8:11], v13, s[16:19], 0 offen
s_endpgm
"""

WORDS_B = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf'
    '0788148f14000a4aff0a0c3a00005a5a83000834'
    '000034d9040500007fc08cbf00008abfff08103a00010000'
    'ff101036f80100000000d8d9080000087fc08cbf'
    '8002187e2801d8d90c00000a7fc08cbf84001a34'
    '001078e00d080480000081bf'
)

# 32 lanes of adjacent pairs. The fixture stores both packed conversion results.
SOURCE_NEIGHBOUR = PREAMBLE + """\
v_add_nc_u32 v21, 11, v0
v_add_nc_u32 v22, 43, v0
v_swap_b32 v21, v22
v_cvt_f32_u32 v23, v21
v_cvt_f32_u32 v24, v22
v_cvt_pkrtz_f16_f32 v29, v23, v24
v_cmp_lt_u32_e32 vcc_lo, 14, v0
v_cndmask_b32_e32 v28, v21, v22, vcc_lo
v_lshlrev_b32 v26, 3, v0
ds_write_b64 v26, v[28:29]
s_waitcnt lgkmcnt(0)
s_barrier
v_xor_b32 v30, 8, v26
v_and_b32 v30, 0xf8, v30
ds_read_b64 v[30:31], v30
s_waitcnt lgkmcnt(0)
buffer_store_dwordx2 v[30:31], v26, s[16:19], 0 offen
s_endpgm
"""
SAVE_AND_CLOBBER = "v_mov_b32_e32 v25, v0\nv_xor_b32 v0, 1, v0\n"
SOURCE_SAVED = SOURCE_NEIGHBOUR.replace(
    'v_lshlrev_b32 v26, 3, v0', SAVE_AND_CLOBBER + 'v_lshlrev_b32 v26, 3, v25')
# Keep the observation address tied to the original lane, even in the GPU mutant.
SOURCE_SAVED = SOURCE_SAVED.replace(SAVE_AND_CLOBBER, 'v_lshlrev_b32 v27, 3, v0\n' + SAVE_AND_CLOBBER)
SOURCE_SAVED = SOURCE_SAVED.replace('v30, 8, v26', 'v30, 8, v27').replace('v[30:31], v26', 'v[30:31], v27')

SOURCE_LATE = SOURCE_SAVED.replace(
    SAVE_AND_CLOBBER, 'v_xor_b32 v0, 1, v0\nv_mov_b32_e32 v25, v0\n')
SOURCE_NO_COPY = SOURCE_SAVED.replace(
    SAVE_AND_CLOBBER, 'v_xor_b32 v0, 1, v0\n').replace('3, v25', '3, v0')

WORDS_NEIGHBOUR = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf8b002a4aab002c4a16cb2a7e150d2e7e'
    '160d307e17313a5e8e00827d152d380283003434000034d91a1c00007fc08cbf00008abf'
    '88343c3aff3c3c36f80000000000d8d91e00001e7fc08cbf001074e01a1e0480000081bf'
)
WORDS_SAVED = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf8b002a4aab002c4a16cb2a7e150d2e7e'
    '160d307e17313a5e8e00827d152d3802830036340003327e8100003a83323434000034d9'
    '1a1c00007fc08cbf00008abf88363c3aff3c3c36f80000000000d8d91e00001e7fc08cbf'
    '001074e01b1e0480000081bf'
)
WORDS_LATE = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf8b002a4aab002c4a16cb2a7e150d2e7e'
    '160d307e17313a5e8e00827d152d3802830036348100003a0003327e83323434000034d9'
    '1a1c00007fc08cbf00008abf88363c3aff3c3c36f80000000000d8d91e00001e7fc08cbf'
    '001074e01b1e0480000081bf'
)
WORDS_NO_COPY = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf8b002a4aab002c4a16cb2a7e150d2e7e'
    '160d307e17313a5e8e00827d152d3802830036348100003a83003434000034d91a1c0000'
    '7fc08cbf00008abf88363c3aff3c3c36f80000000000d8d91e00001e7fc08cbf001074e0'
    '1b1e0480000081bf'
)

CASES = {
    'neighbour': (SOURCE_NEIGHBOUR, WORDS_NEIGHBOUR, USER7 | TGID_X | (1 << 15), (32, 1, 1), (1, 1, 1)),
    'saved': (SOURCE_SAVED, WORDS_SAVED, USER7 | TGID_X | (1 << 15), (32, 1, 1), (1, 1, 1)),
    'P': (SOURCE_P, WORDS_P, USER7 | TGID_X | (2 << 15), (64, 1, 1), (2, 1, 1)),
    'B': (SOURCE_B, WORDS_B, USER7 | TGID_X | (1 << 15), (64, 1, 1), (1, 1, 1)),
}

VERIFIED_SNIPPETS = {
    'ds_write_b64 v4, v[5:6]': '000034d904050000',
    'ds_write_b64 v4, v[5:6] offset:16': '100034d904050000',
    'ds_read_b64 v[8:9], v8': '0000d8d908000008',
    'ds_read_b64 v[24:25], v2 offset:16': '1000d8d902000018',
    'ds_read_b64 v[10:11], v12 offset:296': '2801d8d90c00000a',
}

WORDS = 320
SENTINEL = 0xc5c5c5c5


def expected_words(name):
    out = [SENTINEL] * WORDS
    if name in ('neighbour', 'saved'):
        for lane in range(32):
            peer = lane ^ 1
            out[lane * 2] = peer + (11 if peer > 14 else 43)
            out[lane * 2 + 1], = struct.unpack('<I', struct.pack('<ee', peer + 43, peer + 11))
    elif name == 'P':
        for g in range(2):
            for l in range(64):
                peer = (g << 8) + (l ^ 32)
                idx = (g * 64 + l) * 2
                out[idx] = peer
                out[idx + 1] = peer ^ 0xa5a50000
    elif name == 'B':
        b0 = 37
        b1 = b0 ^ 0x5a5a0000
        for l in range(64):
            peer = l ^ 32
            idx = l * 4
            out[idx] = peer
            out[idx + 1] = peer ^ 0x5a5a0000
            out[idx + 2] = b0
            out[idx + 3] = b1
    else:
        raise KeyError(name)
    return out


def translate_case(name, code=None, rsrc2=None, group=None):
    _, words, default_rsrc2, local, _groups = CASES[name]
    return G.ComputePipeline(words if code is None else code, root_blob(),
                             default_rsrc2 if rsrc2 is None else rsrc2, local if group is None else group)


def check_saved_id_writes():
    def scan(source):
        ins = [(i * 4, 4, asm.split()[0], asm) for i, asm in enumerate(source.splitlines())]
        G.check_lds_exchange(ins, (32, 1, 1), 128)

    swapped = SOURCE_SAVED.replace('v_xor_b32 v0, 1, v0', 'v_swap_b32 v21, v0')
    scan(swapped)
    bad_swap = swapped.replace('3, v25', '3, v0')
    refused('unchanged entry local X ID: v_swap_b32', lambda: scan(bad_swap))
    lost_copy = SOURCE_SAVED.replace('ds_write_b64 v26', 'v_mov_b32_e32 v25, 0\nds_write_b64 v26')
    refused('unchanged entry local X ID: v_xor_b32', lambda: scan(lost_copy))


def check_controls():
    check_saved_id_writes()
    for name in ('neighbour', 'saved'):
        translate_case(name).hlsl_cs()
    for code in (WORDS_LATE, WORDS_NO_COPY):
        refused('LDS fill needs the unchanged entry local X ID: v_xor_b32',
                lambda: translate_case('saved', code=code).hlsl_cs())
    assert mutate(translate_case('saved').hlsl_cs(), 'saved-late-copy')
    # Positive HLSL lowering: declarations, single-eval alias-safe index, barrier, and both dwords.
    hlsl_p = translate_case('P').hlsl_cs()
    assert 'groupshared uint lds[256];' in hlsl_p, hlsl_p
    assert hlsl_p.count('GroupMemoryBarrierWithGroupSync();') == 1, hlsl_p
    assert 'a = (v4 + 16u) >> 2;\n    lds[a] = v5;\n    lds[a + 1u] = v6;' in hlsl_p, hlsl_p
    assert 'a = (v8 + 16u) >> 2;\n    v8 = lds[a];\n    v9 = lds[a + 1u];' in hlsl_p, hlsl_p

    hlsl_b = translate_case('B').hlsl_cs()
    assert 'groupshared uint lds[128];' in hlsl_b, hlsl_b
    assert 'a = (v4 + 0u) >> 2;\n    lds[a] = v5;\n    lds[a + 1u] = v6;' in hlsl_b, hlsl_b
    assert 'a = (v12 + 296u) >> 2;\n    v10 = lds[a];\n    v11 = lds[a + 1u];' in hlsl_b, hlsl_b

    # Tight capacity boundary: offset:16 + 64*8 = 528 exceeds 1 unit (512 bytes), fits at offset:0 and at 32 KiB.
    base = USER7 | TGID_X
    refused('one bounded qword', lambda: translate_case('P', rsrc2=base | (1 << 15)))
    refused('allocation', lambda: translate_case('P', rsrc2=base | (0 << 15)))
    refused('allocation', lambda: translate_case('P', rsrc2=base | (65 << 15)))
    assert 'groupshared uint lds[8192];' in translate_case('P', rsrc2=base | (64 << 15)).hlsl_cs()

    # Exact aligned mask (& 0x1c ok inside span) vs unaligned (& 0x1e refused) and +4 overflow (& 0x1fc refused).
    mask_1f8 = 'ff101036f8010000'
    mask_1c = 'ff1010361c000000'
    mask_1e = 'ff1010361e000000'
    mask_1fc = 'ff101036fc010000'
    ok_1c = translate_case('P', code=patch(WORDS_P, mask_1f8, mask_1c)).hlsl_cs()
    assert 'v8 = 0x0000001cu & v8;' in ok_1c, ok_1c
    refused('aligned', lambda: translate_case('P', code=patch(WORDS_P, mask_1f8, mask_1e)))
    refused('initialized span', lambda: translate_case('P', code=patch(WORDS_P, mask_1f8, mask_1fc)))

    # Broadcast +4 boundary: last initialized dword starts at 524, so reading b64 at offset 524 overflows (+4 -> 528).
    bcast = '2801d8d90c00000a'
    ok_504 = translate_case('B', code=patch(WORDS_B, bcast, 'f801d8d90c00000a')).hlsl_cs()
    assert 'a = (v12 + 504u) >> 2;' in ok_504, ok_504
    refused('initialized span', lambda: translate_case('B', code=patch(WORDS_B, bcast, 'fc01d8d90c00000a')))  # 508+4=512 >= 512

    # Pitched fill, unaligned offsets, wrong shift, missing barrier/waitcnt, EXEC/branch/non-1D/non-compute refusals.
    store = '100034d904050000'
    read_p = '1000d8d908000008'
    wait = '7fc08cbf'
    barrier = '00008abf'
    nop = '000080bf'
    refused('one bounded qword', lambda: translate_case('P', code=patch(WORDS_P, store, '120034d904050000')))  # odd write offset:18
    refused('aligned', lambda: translate_case('P', code=patch(WORDS_P, read_p, '1200d8d908000008')))           # odd read offset:18
    refused('initialized span', lambda: translate_case('P', code=patch(WORDS_P, read_p, '0c00d8d908000008')))  # before span (offset:12)
    refused('one bounded qword', lambda: translate_case('P', code=patch(WORDS_P, '83000834', '82000834')))    # shift 2 instead of 3
    refused('completed fill', lambda: translate_case('P', code=patch(WORDS_P, wait + barrier, nop + barrier)))
    refused('after the fill barrier', lambda: translate_case('P', code=patch(WORDS_P, barrier, nop)))
    refused('completed LGKM wait', lambda: translate_case('P', code=patch(WORDS_P, read_p + wait, read_p + nop)))
    refused('optional byte offset', lambda: translate_case('P', code=patch(WORDS_P, store, '100036d904050000')))  # GDS
    refused('entry EXEC', lambda: translate_case('P', code=patch(WORDS_P, store, '8003febe' + store)))
    refused('straight-line', lambda: translate_case('P', code=patch(WORDS_P, store, '020084bf' + store)))
    refused('1D group', lambda: translate_case('P', group=(32, 2, 1)))

    # Mixed-width interoperability: in-span b32 after b64 succeeds, while out-of-span b32/read2_b32/b64 stay refused.
    mixed_b32 = translate_case('P', code=patch(WORDS_P, read_p, '1000d8d808000008')).hlsl_cs()
    assert 'v8 = lds[(v8 + 16u) >> 2];' in mixed_b32, mixed_b32
    refused('initialized span',
            lambda: translate_case('P', code=patch(WORDS_P, read_p, '0408dcd808000008')))  # ds_read2_b32 offset1:8 past 528
    refused('initialized span',
            lambda: translate_case('P', code=patch(WORDS_P, '83000834' + store, '82000834' + '100034d804050000')))  # ds_write_b32 + ds_read_b64 past 272

    # Register-range validation: reject 1-VGPR, 3-VGPR, >v255, and SGPR operands on ds_write_b64 / ds_read_b64.
    assert G._lds_pair('v[5:6]') == 5
    for bad_pair in ('v5', 'v[5:5]', 'v[5:7]', 'v[5:8]', 'v[5:4]', 'v[255:256]', 's[4:5]'):
        assert G._lds_pair(bad_pair) is None, bad_pair
        decoded_w = [(0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 3, v0'),
                     (4, 8, 'ds_write_b64', f'ds_write_b64 v4, {bad_pair}'),
                     (12, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (16, 4, 's_barrier', 's_barrier 0x0'),
                     (20, 4, 'v_mov_b32', 'v_mov_b32 v8, 0'),
                     (24, 8, 'ds_read_b64', 'ds_read_b64 v[8:9], v8'),
                     (32, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (36, 4, 's_endpgm', 's_endpgm')]
        refused('2-VGPR pair', lambda d=decoded_w: G.check_lds_exchange(d, (4, 1, 1), 128))
        decoded_r = [(0, 4, 'v_lshlrev_b32', 'v_lshlrev_b32 v4, 3, v0'),
                     (4, 8, 'ds_write_b64', 'ds_write_b64 v4, v[5:6]'),
                     (12, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (16, 4, 's_barrier', 's_barrier 0x0'),
                     (20, 4, 'v_mov_b32', 'v_mov_b32 v8, 0'),
                     (24, 8, 'ds_read_b64', f'ds_read_b64 {bad_pair}, v8'),
                     (32, 4, 's_waitcnt', 's_waitcnt lgkmcnt(0)'),
                     (36, 4, 's_endpgm', 's_endpgm')]
        refused('2-VGPR pair', lambda d=decoded_r: G.check_lds_exchange(d, (4, 1, 1), 128))

    # Non-compute stage refusal for b64 DS ops.
    for stage_kind in ('vs', 'ps'):
        s = G.Stage.__new__(G.Stage)
        s.kind = stage_kind
        refused('ds_write_b64', lambda st=s: st.lower('ds_write_b64', 'ds_write_b64 v4, v[5:6]', ['v4', 'v[5:6]'], []))
        refused('ds_read_b64', lambda st=s: st.lower('ds_read_b64', 'ds_read_b64 v[8:9], v8', ['v[8:9]', 'v8'], []))

    # Missing pipe metadata refusal on bare Stage probe.
    bare = G.Stage.__new__(G.Stage)
    bare.kind = 'cs'
    bare.pipe = None
    refused('metadata', lambda: bare.lower('ds_write_b64', 'ds_write_b64 v4, v[5:6]', ['v4', 'v[5:6]'], []))

    print('bounded 64-bit LDS exchange sizing, pair validation, aligned masks, bounds and refusals pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    for name, (source, words, *_rest) in CASES.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for source, words in ((SOURCE_LATE, WORDS_LATE), (SOURCE_NO_COPY, WORDS_NO_COPY)):
        assert assemble(llvm_mc, source) == words
    for line, hex_words in VERIFIED_SNIPPETS.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == hex_words, '%s: assembled %s, recorded %s' % (line, built, hex_words)
    print('64-bit LDS fixture words match llvm-mc (%d programs, %d snippets)' % (len(CASES), len(VERIFIED_SNIPPETS)))


CONTROLS = {
    'saved-late-copy': ('saved', '    v25 = v0;\n    v0 = 0x00000001u ^ v0;\n',
                        '    v0 = 0x00000001u ^ v0;\n    v25 = v0;\n'),
    'p-same-lane': ('P', '    v8 = 0x00000100u ^ v4;\n', '    v8 = v4;\n'),
    'p-swap-dwords': ('P', '    a = (v8 + 16u) >> 2;\n    v8 = lds[a];\n    v9 = lds[a + 1u];\n',
                      '    a = (v8 + 16u) >> 2;\n    v8 = lds[a + 1u];\n    v9 = lds[a];\n'),
    'p-drop-high': ('P', '    lds[a + 1u] = v6;\n', ''),
    'b-wrong-mask': ('B', '    v8 = 0x000001f8u & v8;\n', '    v8 = 0x000000f8u & v8;\n'),
    'b-wrong-lane': ('B', '    a = (v12 + 296u) >> 2;\n', '    a = (v12 + 288u) >> 2;\n'),
}


def mutate(hlsl, control):
    _, old, new = CONTROLS[control]
    assert hlsl.count(old) == 1, (control, hlsl)
    return hlsl.replace(old, new)


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
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if control:
                if run.returncode != 1 or not diagnostics or not re.fullmatch(
                        r'word=\d+ expected=[0-9a-f]{8} actual=[0-9a-f]{8}', diagnostics[0]):
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
