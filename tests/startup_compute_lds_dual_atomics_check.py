#!/usr/bin/env python3
"""Verify dual-offset/stride-64 32-bit LDS ops and 32-bit LDS atomics in tools/startup/gfx2hlsl.py (Refs #1541, #2274).

Covers:
  - Dual-offset & stride-64 32-bit LDS ops: ds_write2_b32 (* 4), ds_write2st64_b32 (* 256), ds_read2st64_b32 (* 256)
  - 32-bit unsigned/bitwise LDS atomics (non-rtn & _rtn): ds_add_u32, ds_add_rtn_u32, ds_min_u32, ds_min_rtn_u32,
    ds_max_u32, ds_max_rtn_u32, ds_and_b32, ds_and_rtn_b32, ds_or_b32, ds_or_rtn_b32, ds_wrxchg_rtn_b32

Modes:
  (no arguments)                                     pure-Python unit, encoding, HLSL and refusal controls
  binary <llvm-mc>                                   reassemble every fixture and snippet with llvm-mc (gfx1013)
  dispatch <dxc> <offscreen_check> <adapter>         compile DXC cs_6_0, dispatch and verify GPU readback
  control <name> <dxc> <offscreen_check> <adapter>   verify mutated shader/reference control rejects at pinned word
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

import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_compute_check import MASK, SENTINEL, TGID_X, USER7, WORDS, patch, refused, renderer_diagnostics, root_blob  # noqa: E402


def le32(w):
    return (w & MASK).to_bytes(4, 'little')


def sop2(op, sdst, ssrc0, ssrc1):
    return le32(0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def sop1(op, sdst, ssrc0):
    return le32(0xbe800000 | ((sdst & 0x7f) << 16) | ((op & 0xff) << 8) | (ssrc0 & 0xff))


def sopp(op, simm=0):
    return le32(0xbf800000 | ((op & 0x7f) << 16) | (simm & 0xffff))


def vop1(op, vdst, src0, lit=None):
    b = le32(0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def v_mov_imm(vdst, val):
    if 0 <= val <= 64:
        return vop1(0x01, vdst, 0x80 + val)
    if -16 <= val <= -1:
        return vop1(0x01, vdst, 0xc0 - val)
    return vop1(0x01, vdst, 0xff, val)


def vop2(op, vdst, src0, vsrc1, lit=None):
    b = le32(((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def vopc(op, src0, vsrc1, lit=None):
    b = le32(0x7c000000 | ((op & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def mubuf_store(op, vdata, vaddr, srsrc=16, soffset_enc=0x80, offset=0):
    w0 = 0xe0001000 | ((op & 0x7f) << 18) | (offset & 0xfff)
    w1 = ((soffset_enc & 0xff) << 24) | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return le32(w0) + le32(w1)


def ds_single(op, addr, data0=0, vdst=0, offset=0, gds=0, bit16=0, data1=0):
    w0 = 0xd8000000 | ((op & 0xff) << 18) | ((gds & 1) << 17) | ((bit16 & 1) << 16) | (offset & 0xffff)
    w1 = ((vdst & 0xff) << 24) | ((data1 & 0xff) << 16) | ((data0 & 0xff) << 8) | (addr & 0xff)
    return le32(w0) + le32(w1)


def ds_dual(op, addr, data0=0, data1=0, vdst=0, offset0=0, offset1=0, gds=0, bit16=0):
    off16 = ((offset1 & 0xff) << 8) | (offset0 & 0xff)
    return ds_single(op, addr, data0=data0, vdst=vdst, offset=off16, gds=gds, bit16=bit16, data1=data1)


PREAMBLE_LINES = [
    ('s_mov_b32 s10, s6', bytes.fromhex('06038abe')),
    ('s_mov_b32 s11, 0', bytes.fromhex('80038bbe')),
    ('s_load_dwordx8 s[12:19], s[10:11], 0x0', bytes.fromhex('05030cf4000000fa')),
    ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
]


def build_program(items):
    return '\n'.join(asm for asm, _ in items) + '\n', b''.join(raw for _, raw in items)


# ---------------------------------------------------------------------------
# Case D: [numthreads(64, 1, 1)], groups=(1, 1, 1), LDS_SIZE=4 (2048 B = 8 st64 banks).
# Exercises ds_write2_b32, ds_write2st64_b32, ds_read2st64_b32, omitted offset0=0,
# vdst[0]==vaddr aliasing, EXEC-guarded dual write + static OOB offset1:8,
# runtime misaligned write no-op, and runtime dynamic OOB read returning 0.
# ---------------------------------------------------------------------------
def make_case_d():
    p = list(PREAMBLE_LINES)
    p += [
        ('v_lshlrev_b32 v4, 2, v0', vop2(0x1a, 4, 0x82, 0)),
        ('v_xor_b32 v5, 0x11000000, v0', vop2(0x1d, 5, 0xff, 0, 0x11000000)),
        ('v_lshlrev_b32 v6, 8, v0', vop2(0x1a, 6, 0x88, 0)),
        ('v_xor_b32 v6, 0x22000000, v6', vop2(0x1d, 6, 0xff, 6, 0x22000000)),
        # Bank 0 (+0) and Bank 2 (+512) via ds_write2st64_b32 (omitted offset0 defaults to 0).
        ('ds_write2st64_b32 v4, v5, v6 offset1:2', ds_dual(0x0f, 4, 5, 6, offset0=0, offset1=2)),
        # Bank 6 (+1536) and Bank 7 (+1792) via ds_write2st64_b32.
        ('ds_write2st64_b32 v4, v5, v6 offset0:6 offset1:7', ds_dual(0x0f, 4, 5, 6, offset0=6, offset1=7)),
        # Bank 4..5 (+1024..1535) via ds_write2_b32 (8 bytes per lane: offset0=0, offset1=1 -> +4B).
        ('v_lshlrev_b32 v7, 3, v0', vop2(0x1a, 7, 0x83, 0)),
        ('v_add_nc_u32 v7, 1024, v7', vop2(0x25, 7, 0xff, 7, 1024)),
        ('v_xor_b32 v8, 0x33000000, v0', vop2(0x1d, 8, 0xff, 0, 0x33000000)),
        ('v_xor_b32 v9, 0x44000000, v0', vop2(0x1d, 9, 0xff, 0, 0x44000000)),
        ('ds_write2_b32 v7, v8, v9 offset1:1', ds_dual(0x0e, 7, 8, 9, offset0=0, offset1=1)),
        # EXEC-guarded dual st64 write: only lanes 0..31 overwrite Bank 6; offset1:8 (2048 B) is static OOB no-op.
        ('v_or_b32 v10, 0x55000000, v0', vop2(0x1c, 10, 0xff, 0, 0x55000000)),
        ('v_mov_b32 v11, 0xdead0001', v_mov_imm(11, 0xdead0001)),
        ('v_cmp_gt_u32 vcc_lo, 32, v0', vopc(0xc4, 0x80 + 32, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('ds_write2st64_b32 v4, v10, v11 offset0:6 offset1:8', ds_dual(0x0f, 4, 10, 11, offset0=6, offset1=8)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        # Runtime misaligned address (4*l + 2): ds_write2_b32 must be a no-op and leave Bank 0 intact.
        ('v_add_nc_u32 v12, 2, v4', vop2(0x25, 12, 0x82, 4)),
        ('ds_write2_b32 v12, v11, v11 offset1:1', ds_dual(0x0e, 12, 11, 11, offset0=0, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('s_barrier', bytes.fromhex('00008abf')),
        # Cross-wave peer read with vdst[0] == vaddr aliasing (v[14:15], v14): Bank 2 and EXEC-masked Bank 6.
        ('v_xor_b32 v14, 0x80, v4', vop2(0x1d, 14, 0xff, 4, 0x80)),
        ('ds_read2st64_b32 v[14:15], v14 offset0:2 offset1:6', ds_dual(0x38, 14, vdst=14, offset0=2, offset1=6)),
        # Bank 4 (even dwords from ds_write2_b32) + Bank 7 (lanes 0..31 in-bounds, lanes 32..63 dynamic OOB -> 0).
        ('v_lshlrev_b32 v16, 3, v0', vop2(0x1a, 16, 0x83, 0)),
        ('ds_read2st64_b32 v[16:17], v16 offset0:4 offset1:7', ds_dual(0x38, 16, vdst=16, offset0=4, offset1=7)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('v_lshlrev_b32 v13, 4, v0', vop2(0x1a, 13, 0x84, 0)),
        ('buffer_store_dwordx4 v[14:17], v13, s[16:19], 0 offen', mubuf_store(0x1e, 14, 13)),
        # Lanes 0..23 also read Bank 0 (proving misaligned write did not clobber it) and Bank 4 odd dwords (+4).
        ('v_lshlrev_b32 v18, 3, v0', vop2(0x1a, 18, 0x83, 0)),
        ('v_add_nc_u32 v18, 4, v18', vop2(0x25, 18, 0x84, 18)),
        ('ds_read2st64_b32 v[18:19], v18 offset1:4', ds_dual(0x38, 18, vdst=18, offset0=0, offset1=4)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('v_cmp_gt_u32 vcc_lo, 24, v0', vopc(0xc4, 0x80 + 24, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('v_lshlrev_b32 v20, 3, v0', vop2(0x1a, 20, 0x83, 0)),
        ('buffer_store_dwordx2 v[18:19], v20, s[16:19], 0 offen offset:1024', mubuf_store(0x1d, 18, 20, offset=1024)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        ('s_endpgm', bytes.fromhex('000081bf')),
    ]
    return build_program(p)


# ---------------------------------------------------------------------------
# Case A: [numthreads(64, 1, 1)], groups=(2, 1, 1), LDS_SIZE=1 (512 B = 128 dwords).
# 2 groups x 64 threads contending on shared LDS slots with ds_add_u32, ds_add_rtn_u32,
# ds_min_u32, ds_max_u32, ds_or_b32, ds_and_b32, ds_wrxchg_rtn_b32, plus bijective ticket
# histogram proof, wrxchg conservation sum, EXEC-masked atomics + inactive vdst preservation,
# runtime OOB _rtn -> 0, and runtime misaligned _rtn -> 0.
# ---------------------------------------------------------------------------
def make_case_a():
    p = list(PREAMBLE_LINES)
    p += [
        ('v_mov_b32 v1, 0', v_mov_imm(1, 0)),
        ('v_mov_b32 v2, 1', v_mov_imm(2, 1)),
        # Lanes 0..31 zero lds[64..127] (bytes 256..508) via ds_write2_b32.
        ('v_lshlrev_b32 v3, 3, v0', vop2(0x1a, 3, 0x83, 0)),
        ('v_add_nc_u32 v3, 256, v3', vop2(0x25, 3, 0xff, 3, 256)),
        ('v_cmp_gt_u32 vcc_lo, 32, v0', vopc(0xc4, 0x80 + 32, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('ds_write2_b32 v3, v1, v1 offset1:1', ds_dual(0x0e, 3, 1, 1, offset0=0, offset1=1)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        # Lane 0 initializes lds[0..11] via ds_write2_b32.
        ('v_cmp_gt_u32 vcc_lo, 1, v0', vopc(0xc4, 0x81, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('v_mov_b32 v4, 0x10000000', v_mov_imm(4, 0x10000000)),
        ('v_mov_b32 v5, 0x20000000', v_mov_imm(5, 0x20000000)),
        ('ds_write2_b32 v1, v4, v5 offset1:1', ds_dual(0x0e, 1, 4, 5, offset0=0, offset1=1)),
        ('v_mov_b32 v4, -2', v_mov_imm(4, -2)),
        ('v_mov_b32 v5, 1', v_mov_imm(5, 1)),
        ('ds_write2_b32 v1, v4, v5 offset0:2 offset1:3', ds_dual(0x0e, 1, 4, 5, offset0=2, offset1=3)),
        ('v_mov_b32 v4, 0', v_mov_imm(4, 0)),
        ('v_mov_b32 v5, -1', v_mov_imm(5, -1)),
        ('ds_write2_b32 v1, v4, v5 offset0:4 offset1:5', ds_dual(0x0e, 1, 4, 5, offset0=4, offset1=5)),
        ('v_mov_b32 v4, 0', v_mov_imm(4, 0)),
        ('v_mov_b32 v5, 0xdeadbeef', v_mov_imm(5, 0xdeadbeef)),
        ('ds_write2_b32 v1, v4, v5 offset0:6 offset1:7', ds_dual(0x0e, 1, 4, 5, offset0=6, offset1=7)),
        ('v_mov_b32 v4, 0', v_mov_imm(4, 0)),
        ('v_mov_b32 v5, 0x77770000', v_mov_imm(5, 0x77770000)),
        ('ds_write2_b32 v1, v4, v5 offset0:8 offset1:9', ds_dual(0x0e, 1, 4, 5, offset0=8, offset1=9)),
        ('v_mov_b32 v4, 0x100', v_mov_imm(4, 0x100)),
        ('v_mov_b32 v5, 0', v_mov_imm(5, 0)),
        ('ds_write2_b32 v1, v4, v5 offset0:10 offset1:11', ds_dual(0x0e, 1, 4, 5, offset0=10, offset1=11)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('s_barrier', bytes.fromhex('00008abf')),
        # All 64 lanes execute contended 32-bit LDS atomics.
        ('s_lshl_b32 s20, s7, 8', sop2(0x1e, 20, 7, 0x88)),
        ('s_lshl_b32 s22, s7, 16', sop2(0x1e, 22, 7, 0x90)),
        ('v_add_nc_u32 v6, s20, v0', vop2(0x25, 6, 20, 0)),
        ('v_add_nc_u32 v6, 1, v6', vop2(0x25, 6, 0x81, 6)),
        ('ds_add_u32 v1, v6', ds_single(0x00, 1, 6, offset=0)),
        ('ds_add_rtn_u32 v7, v1, v6 offset:4', ds_single(0x20, 1, 6, vdst=7, offset=4)),
        # Unsigned min/max values crossing 0x7fffffff / 0x80000000 signed boundary.
        ('v_add_nc_u32 v8, 0x7ffffff0, v0', vop2(0x25, 8, 0xff, 0, 0x7ffffff0)),
        ('v_add_nc_u32 v8, s22, v8', vop2(0x25, 8, 22, 8)),
        ('v_and_b32 v9, 3, v0', vop2(0x1b, 9, 0x83, 0)),
        ('v_lshlrev_b32 v9, 30, v9', vop2(0x1a, 9, 0x80 + 30, 9)),
        ('v_xor_b32 v8, v9, v8', vop2(0x1d, 8, 0x109, 8)),
        ('ds_min_u32 v1, v8 offset:8', ds_single(0x07, 1, 8, offset=8)),
        ('ds_max_u32 v1, v8 offset:12', ds_single(0x08, 1, 8, offset=12)),
        # Bitwise OR and AND reductions.
        ('v_and_b32 v10, 15, v0', vop2(0x1b, 10, 0x80 + 15, 0)),
        ('v_lshlrev_b32 v10, v10, v2', vop2(0x1a, 10, 0x10a, 2)),
        ('v_or_b32 v11, s22, v10', vop2(0x1c, 11, 22, 10)),
        ('ds_or_b32 v1, v11 offset:16', ds_single(0x0a, 1, 11, offset=16)),
        ('v_not_b32 v12, v10', vop1(0x37, 12, 0x10a)),
        ('ds_and_b32 v1, v12 offset:20', ds_single(0x09, 1, 12, offset=20)),
        # Bijective ticket proof: 64 threads fetch unique 0..63 tickets and mark lds[64 + ticket].
        ('ds_add_rtn_u32 v13, v1, v2 offset:24', ds_single(0x20, 1, 2, vdst=13, offset=24)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('v_lshlrev_b32 v14, 2, v13', vop2(0x1a, 14, 0x82, 13)),
        ('v_add_nc_u32 v14, 256, v14', vop2(0x25, 14, 0xff, 14, 256)),
        ('ds_add_u32 v14, v2', ds_single(0x00, 14, 2, offset=0)),
        # wrxchg conservation proof: sum(returned_old) + final_lds7 == init + sum(v15).
        ('v_or_b32 v15, 0xaa000000, v6', vop2(0x1c, 15, 0xff, 6, 0xaa000000)),
        ('ds_wrxchg_rtn_b32 v16, v1, v15 offset:28', ds_single(0x2d, 1, 15, vdst=16, offset=28)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('ds_add_u32 v1, v16 offset:32', ds_single(0x00, 1, 16, offset=32)),
        # EXEC-guarded atomics (lanes 0..15) + inactive-lane vdst preservation (v17 stays 0x12345678 for l>=16).
        ('v_mov_b32 v17, 0x12345678', v_mov_imm(17, 0x12345678)),
        ('v_cmp_gt_u32 vcc_lo, 16, v0', vopc(0xc4, 0x80 + 16, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('ds_add_u32 v1, v2 offset:36', ds_single(0x00, 1, 2, offset=36)),
        ('ds_add_rtn_u32 v17, v1, v2 offset:40', ds_single(0x20, 1, 2, vdst=17, offset=40)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('ds_add_u32 v1, v17 offset:44', ds_single(0x00, 1, 17, offset=44)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        # Runtime OOB and misaligned _rtn atomics: LDS untouched, destination receives 0.
        ('v_mov_b32 v18, 512', v_mov_imm(18, 512)),
        ('v_mov_b32 v19, 0xbad00001', v_mov_imm(19, 0xbad00001)),
        ('ds_add_rtn_u32 v19, v18, v2', ds_single(0x20, 18, 2, vdst=19, offset=0)),
        ('v_mov_b32 v20, 2', v_mov_imm(20, 2)),
        ('v_mov_b32 v21, 0xbad00002', v_mov_imm(21, 0xbad00002)),
        ('ds_wrxchg_rtn_b32 v21, v20, v2', ds_single(0x2d, 20, 2, vdst=21, offset=0)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('s_barrier', bytes.fromhex('00008abf')),
        # Read back lds[l] and lds[64 + l] in one ds_read2st64_b32, store 64 ticket words per group.
        ('v_lshlrev_b32 v22, 2, v0', vop2(0x1a, 22, 0x82, 0)),
        ('ds_read2st64_b32 v[22:23], v22 offset1:1', ds_dual(0x38, 22, vdst=22, offset0=0, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        # Per-group output byte base s21 = gid * 320 (80 dwords per group).
        ('s_mul_i32 s21, s7, 320', sop2(0x26, 21, 7, 0xff) + le32(320)),
        ('v_lshlrev_b32 v24, 2, v0', vop2(0x1a, 24, 0x82, 0)),
        ('v_add_nc_u32 v24, s21, v24', vop2(0x25, 24, 21, 24)),
        ('buffer_store_dword v23, v24, s[16:19], 0 offen offset:64', mubuf_store(0x1c, 23, 24, offset=64)),
        # Lanes 0..6 store lds[0..6] to words 0..6 of the group's 80-word slice.
        ('v_cmp_gt_u32 vcc_lo, 7, v0', vopc(0xc4, 0x87, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('buffer_store_dword v22, v24, s[16:19], 0 offen', mubuf_store(0x1c, 22, 24, offset=0)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        # Lane 0 computes wrxchg conservation sum (lds[7] + lds[8]) and stores words 7..12.
        ('v_cmp_gt_u32 vcc_lo, 1, v0', vopc(0xc4, 0x81, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('ds_add_rtn_u32 v25, v1, v1 offset:28', ds_single(0x20, 1, 1, vdst=25, offset=28)),
        ('ds_add_rtn_u32 v26, v1, v1 offset:32', ds_single(0x20, 1, 1, vdst=26, offset=32)),
        ('ds_add_rtn_u32 v27, v1, v1 offset:36', ds_single(0x20, 1, 1, vdst=27, offset=36)),
        ('ds_add_rtn_u32 v28, v1, v1 offset:40', ds_single(0x20, 1, 1, vdst=28, offset=40)),
        ('ds_add_rtn_u32 v29, v1, v1 offset:44', ds_single(0x20, 1, 1, vdst=29, offset=44)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        ('v_add_nc_u32 v25, v25, v26', vop2(0x25, 25, 0x119, 26)),
        ('v_mov_b32 v26, v27', vop1(0x01, 26, 0x11b)),
        ('v_mov_b32 v27, v28', vop1(0x01, 27, 0x11c)),
        ('v_mov_b32 v28, v29', vop1(0x01, 28, 0x11d)),
        ('buffer_store_dwordx4 v[25:28], v24, s[16:19], 0 offen offset:28', mubuf_store(0x1e, 25, 24, offset=28)),
        ('v_mov_b32 v29, v19', vop1(0x01, 29, 0x113)),
        ('v_mov_b32 v30, v21', vop1(0x01, 30, 0x115)),
        ('buffer_store_dwordx2 v[29:30], v24, s[16:19], 0 offen offset:44', mubuf_store(0x1d, 29, 24, offset=44)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        # Lane 16 stores v17 (0x12345678, preserved while inactive during l < 16 guard) at word 13 (offset 52).
        ('v_cmp_eq_u32 vcc_lo, 16, v0', vopc(0xc2, 0x80 + 16, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('v_mov_b32 v31, s21', vop1(0x01, 31, 21)),
        ('buffer_store_dword v17, v31, s[16:19], 0 offen offset:52', mubuf_store(0x1c, 17, 31, offset=52)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        ('s_endpgm', bytes.fromhex('000081bf')),
    ]
    return build_program(p)


# ---------------------------------------------------------------------------
# Case R: [numthreads(32, 1, 1)], groups=(1, 1, 1), LDS_SIZE=2 (1024 B = 256 dwords).
# Per-lane private slots exercising all 6 _rtn 32-bit LDS atomics (ds_add_rtn_u32,
# ds_min_rtn_u32, ds_max_rtn_u32, ds_or_rtn_b32, ds_and_rtn_b32, ds_wrxchg_rtn_b32)
# with vdst==vaddr, vdst==vdata, and vdst==vaddr==vdata aliasing, verifying both
# pre-op return values and post-op LDS words deterministically.
# ---------------------------------------------------------------------------
def make_case_r():
    p = list(PREAMBLE_LINES)
    p += [
        ('v_lshlrev_b32 v4, 2, v0', vop2(0x1a, 4, 0x82, 0)),  # 0..124 (slot per lane in each 128B region)
        # Initialize 6 regions of 32 dwords (offsets 0, 128, 256, 384, 512, 640) via ds_write2st64_b32.
        ('v_add_nc_u32 v5, 0x7ffffff0, v0', vop2(0x25, 5, 0xff, 0, 0x7ffffff0)),
        ('v_xor_b32 v6, 0x80000010, v0', vop2(0x1d, 6, 0xff, 0, 0x80000010)),
        ('ds_write2st64_b32 v4, v5, v6 offset1:1', ds_dual(0x0f, 4, 5, 6, offset0=0, offset1=1)),  # +0 and +256
        ('v_add_nc_u32 v7, 128, v4', vop2(0x25, 7, 0xff, 4, 128)),
        ('ds_write2st64_b32 v7, v6, v5 offset1:1', ds_dual(0x0f, 7, 6, 5, offset0=0, offset1=1)),  # +128 and +384
        ('v_xor_b32 v8, 0x0f0f0000, v0', vop2(0x1d, 8, 0xff, 0, 0x0f0f0000)),
        ('v_xor_b32 v9, 0xf0f000ff, v0', vop2(0x1d, 9, 0xff, 0, 0xf0f000ff)),
        ('ds_write2st64_b32 v4, v8, v9 offset0:2 offset1:3', ds_dual(0x0f, 4, 8, 9, offset0=2, offset1=3)),  # +512 and +768
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        # 1. ds_add_rtn_u32 at +0 with vdst == vaddr aliasing (v10 holds addr, receives old).
        ('v_mov_b32 v10, v4', vop1(0x01, 10, 0x104)),
        ('v_add_nc_u32 v11, 0x100, v0', vop2(0x25, 11, 0xff, 0, 0x100)),
        ('ds_add_rtn_u32 v10, v10, v11', ds_single(0x20, 10, 11, vdst=10, offset=0)),
        # 2. ds_min_rtn_u32 at +128 with vdst == vdata aliasing (v11 holds data, receives old).
        ('v_add_nc_u32 v11, 0x7ffffff8, v0', vop2(0x25, 11, 0xff, 0, 0x7ffffff8)),
        ('ds_min_rtn_u32 v11, v4, v11 offset:128', ds_single(0x27, 4, 11, vdst=11, offset=128)),
        # 3. ds_max_rtn_u32 at +256.
        ('v_add_nc_u32 v12, 0x7ffffff8, v0', vop2(0x25, 12, 0xff, 0, 0x7ffffff8)),
        ('ds_max_rtn_u32 v12, v4, v12 offset:256', ds_single(0x28, 4, 12, vdst=12, offset=256)),
        # 4. ds_or_rtn_b32 at +512.
        ('v_mov_b32 v13, 0x00f000aa', v_mov_imm(13, 0x00f000aa)),
        ('ds_or_rtn_b32 v13, v4, v13 offset:512', ds_single(0x2a, 4, 13, vdst=13, offset=512)),
        # 5. ds_and_rtn_b32 at +768.
        ('v_mov_b32 v14, 0xff0000f0', v_mov_imm(14, 0xff0000f0)),
        ('ds_and_rtn_b32 v14, v4, v14 offset:768', ds_single(0x29, 4, 14, vdst=14, offset=768)),
        # 6. ds_wrxchg_rtn_b32 at +384 with triple aliasing vdst == vaddr == vdata (v15 = 4*l + 384).
        ('v_add_nc_u32 v15, 384, v4', vop2(0x25, 15, 0xff, 4, 384)),
        ('ds_wrxchg_rtn_b32 v15, v15, v15', ds_single(0x2d, 15, 15, vdst=15, offset=0)),
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        # Read back post-op values of all 6 slots using ds_read2st64_b32.
        ('ds_read2st64_b32 v[16:17], v4 offset1:1', ds_dual(0x38, 4, vdst=16, offset0=0, offset1=1)),      # post +0, +256
        ('ds_read2st64_b32 v[18:19], v7 offset1:1', ds_dual(0x38, 7, vdst=18, offset0=0, offset1=1)),      # post +128, +384
        ('ds_read2st64_b32 v[20:21], v4 offset0:2 offset1:3', ds_dual(0x38, 4, vdst=20, offset0=2, offset1=3)),  # post +512, +768
        ('s_waitcnt lgkmcnt(0)', bytes.fromhex('7fc08cbf')),
        # Pack 8 words per lane (256 words total; words 256..319 remain SENTINEL):
        # [old_add^post_add, old_min, post_min, old_max, post_max, old_or^post_or, old_and^post_and, old_xchg^post_xchg]
        # Store exact distinct words for lanes 0..31 by folding old/post with rotations or storing 8 words:
        #   word 0: old_add (v10), word 1: post_add (v16), word 2: old_min^post_min, word 3: old_max^post_max
        #   Wait: storing 8 direct words (with old+post verified cleanly):
        #   [old_add, post_add, post_min, post_max, post_or, post_and, old_xchg, post_xchg] +
        #   XOR old_min/old_max/old_or/old_and into a second pass? Better: just mix old_min/max/or/and into lanes 0..31
        #   without losing any bit: 10 words per lane = 320? No, keep 32 sentinel words! 9 words * 32 = 288 (32 sentinels)!
        #   Even cleaner: 8 words per lane using exact non-symmetric checksums? No, direct words are cleaner for mutants!
        #   With 24 lanes? No, 32 lanes x 8 words = 256 words:
        #   [old_add, post_add, old_min, post_min, old_max, post_max, post_or ^ (old_or << 1), post_and ^ (old_and << 1)]
        #   Wait: we still have words 256..319 (64 words)! Lanes 0..15 can store [old_xchg, post_xchg, old_or, old_and]
        #   at words 256..319? Keep 16 sentinel words: lanes 0..11 store [old_xchg, post_xchg, old_or, old_and] = 48 words (256..303),
        #   leaving words 304..319 as 16 untouched SENTINEL words!
        ('v_lshlrev_b32 v22, 5, v0', vop2(0x1a, 22, 0x85, 0)),  # 32 * l (8 dwords per lane)
        ('v_mov_b32 v24, v10', vop1(0x01, 24, 0x10a)),  # old_add
        ('v_mov_b32 v25, v16', vop1(0x01, 25, 0x110)),  # post_add
        ('v_mov_b32 v26, v11', vop1(0x01, 26, 0x10b)),  # old_min
        ('v_mov_b32 v27, v18', vop1(0x01, 27, 0x112)),  # post_min
        ('buffer_store_dwordx4 v[24:27], v22, s[16:19], 0 offen', mubuf_store(0x1e, 24, 22, offset=0)),
        ('v_mov_b32 v24, v12', vop1(0x01, 24, 0x10c)),  # old_max
        ('v_mov_b32 v25, v17', vop1(0x01, 25, 0x111)),  # post_max
        ('v_mov_b32 v26, v20', vop1(0x01, 26, 0x114)),  # post_or
        ('v_mov_b32 v27, v21', vop1(0x01, 27, 0x115)),  # post_and
        ('buffer_store_dwordx4 v[24:27], v22, s[16:19], 0 offen offset:16', mubuf_store(0x1e, 24, 22, offset=16)),
        # Lanes 0..11 store [old_xchg, post_xchg, old_or, old_and] at byte offset 1024 + 16*l (words 256..303).
        ('v_cmp_gt_u32 vcc_lo, 12, v0', vopc(0xc4, 0x80 + 12, 0)),
        ('s_and_saveexec_b32 s0, vcc_lo', bytes.fromhex('6a3c80be')),
        ('v_lshlrev_b32 v23, 4, v0', vop2(0x1a, 23, 0x84, 0)),
        ('v_mov_b32 v24, v15', vop1(0x01, 24, 0x10f)),  # old_xchg
        ('v_mov_b32 v25, v19', vop1(0x01, 25, 0x113)),  # post_xchg
        ('v_mov_b32 v26, v13', vop1(0x01, 26, 0x10d)),  # old_or
        ('v_mov_b32 v27, v14', vop1(0x01, 27, 0x10e)),  # old_and
        ('buffer_store_dwordx4 v[24:27], v23, s[16:19], 0 offen offset:1024', mubuf_store(0x1e, 24, 23, offset=1024)),
        ('s_or_b32 exec_lo, exec_lo, s0', bytes.fromhex('7e007e88')),
        ('s_endpgm', bytes.fromhex('000081bf')),
    ]
    return build_program(p)


def make_case_wide():
    p = list(PREAMBLE_LINES)
    p += [
        ('v_lshlrev_b32 v3, 4, v0', vop2(0x1a, 3, 0x84, 0)),
        ('v_lshlrev_b32 v4, 3, v0', vop2(0x1a, 4, 0x83, 0)),
        ('v_add_nc_u32 v4, 1024, v4', vop2(0x25, 4, 0xff, 4, 1024)),
    ]
    for reg, tag in enumerate((0x105b0000, 0xfa040000, 0x437e0000, 0xcb230000), 8):
        p.append((f'v_xor_b32 v{reg}, 0x{tag:x}, v0', vop2(0x1d, reg, 0xff, 0, tag)))
    p += [
        # Initialize the ordinary second slot so the wrong-offset control reads defined data.
        ('ds_write2_b64 v3, v[8:9], v[8:9] offset1:1', ds_dual(0x4e, 3, 8, 8, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', sopp(0x0c, 0xc07f)),
        ('s_barrier', sopp(0x0a)),
        ('ds_write2_b64 v3, v[8:9], v[10:11] offset1:1', ds_dual(0x4e, 3, 8, 10, offset1=1)),
        ('ds_write2st64_b64 v4, v[10:11], v[8:9] offset1:1', ds_dual(0x4f, 4, 10, 8, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', sopp(0x0c, 0xc07f)),
        ('s_barrier', sopp(0x0a)),
        ('v_add_nc_u32 v5, 1, v0', vop2(0x25, 5, 0x81, 0)),
        ('v_and_b32 v5, 31, v5', vop2(0x1b, 5, 0x9f, 5)),
        ('v_lshlrev_b32 v6, 4, v5', vop2(0x1a, 6, 0x84, 5)),
        # Destination aliases the incoming address in each form.
        ('ds_read2_b64 v[6:9], v6 offset1:1', ds_dual(0x77, 6, vdst=6, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', sopp(0x0c, 0xc07f)),
        ('buffer_store_dwordx4 v[6:9], v3, s[16:19], 0 offen', mubuf_store(0x1e, 6, 3)),
        ('v_lshlrev_b32 v7, 3, v5', vop2(0x1a, 7, 0x83, 5)),
        ('v_add_nc_u32 v7, 1024, v7', vop2(0x25, 7, 0xff, 7, 1024)),
        ('ds_read2st64_b64 v[7:10], v7 offset1:1', ds_dual(0x78, 7, vdst=7, offset1=1)),
        ('s_waitcnt lgkmcnt(0)', sopp(0x0c, 0xc07f)),
        ('buffer_store_dwordx4 v[7:10], v3, s[16:19], 0 offen offset:512', mubuf_store(0x1e, 7, 3, offset=512)),
        ('s_endpgm', sopp(0x01)),
    ]
    return build_program(p)


SOURCE_D, WORDS_D = make_case_d()
SOURCE_A, WORDS_A = make_case_a()
SOURCE_R, WORDS_R = make_case_r()
SOURCE_WIDE, WORDS_WIDE = make_case_wide()

CASES = {
    'wide': (SOURCE_WIDE, WORDS_WIDE, USER7 | (4 << 15), (32, 1, 1), (1, 1, 1)),
    'D': (SOURCE_D, WORDS_D, USER7 | TGID_X | (4 << 15), (64, 1, 1), (1, 1, 1)),
    'A': (SOURCE_A, WORDS_A, USER7 | TGID_X | (1 << 15), (64, 1, 1), (2, 1, 1)),
    'R': (SOURCE_R, WORDS_R, USER7 | TGID_X | (2 << 15), (32, 1, 1), (1, 1, 1)),
}

# Verified single-instruction gfx1013 DS snippets for all 14 lowered opcodes.
VERIFIED_SNIPPETS = {
    'ds_write2_b32 v2, v8, v16 offset0:4 offset1:8': ds_dual(0x0e, 2, 8, 16, offset0=4, offset1=8).hex(),
    'ds_write2st64_b32 v4, v5, v6 offset0:2 offset1:3': ds_dual(0x0f, 4, 5, 6, offset0=2, offset1=3).hex(),
    'ds_read2st64_b32 v[10:11], v5 offset0:3 offset1:2': ds_dual(0x38, 5, vdst=10, offset0=3, offset1=2).hex(),
    'ds_add_u32 v2, v8 offset:8': ds_single(0x00, 2, 8, offset=8).hex(),
    'ds_min_u32 v2, v8 offset:12': ds_single(0x07, 2, 8, offset=12).hex(),
    'ds_max_u32 v2, v8 offset:16': ds_single(0x08, 2, 8, offset=16).hex(),
    'ds_and_b32 v2, v8 offset:20': ds_single(0x09, 2, 8, offset=20).hex(),
    'ds_or_b32 v2, v8 offset:24': ds_single(0x0a, 2, 8, offset=24).hex(),
    'ds_add_rtn_u32 v24, v2, v8 offset:8': ds_single(0x20, 2, 8, vdst=24, offset=8).hex(),
    'ds_min_rtn_u32 v24, v2, v8 offset:12': ds_single(0x27, 2, 8, vdst=24, offset=12).hex(),
    'ds_max_rtn_u32 v24, v2, v8 offset:16': ds_single(0x28, 2, 8, vdst=24, offset=16).hex(),
    'ds_and_rtn_b32 v24, v2, v8 offset:20': ds_single(0x29, 2, 8, vdst=24, offset=20).hex(),
    'ds_or_rtn_b32 v24, v2, v8 offset:24': ds_single(0x2a, 2, 8, vdst=24, offset=24).hex(),
    'ds_wrxchg_rtn_b32 v24, v2, v8 offset:28': ds_single(0x2d, 2, 8, vdst=24, offset=28).hex(),
}


def expected_words(name):
    out = [SENTINEL] * WORDS
    if name == 'wide':
        for lane in range(32):
            peer = (lane + 1) % 32
            words = [tag ^ peer for tag in (0x105b0000, 0xfa040000, 0x437e0000, 0xcb230000)]
            out[4 * lane:4 * lane + 4] = words
            out[128 + 4 * lane:128 + 4 * lane + 4] = words[2:] + words[:2]
    elif name == 'D':
        for l in range(64):
            peer = l ^ 32
            bank2_peer = 0x22000000 ^ (peer << 8)
            bank6_peer = (0x55000000 | peer) if peer < 32 else (0x11000000 ^ peer)
            bank4_even = 0x33000000 ^ l
            bank7_word = (0x22000000 ^ ((2 * l) << 8)) if l < 32 else 0
            out[4 * l:4 * l + 4] = [bank2_peer, bank6_peer, bank4_even, bank7_word]
        for l in range(24):
            bank0_odd = 0x11000000 ^ (2 * l + 1)
            bank4_odd = 0x44000000 ^ l
            out[256 + 2 * l:256 + 2 * l + 2] = [bank0_odd, bank4_odd]
    elif name == 'A':
        for g in range(2):
            base = g * 80
            adds = [((g << 8) + l + 1) & MASK for l in range(64)]
            minmax = [(((0x7ffffff0 + l + (g << 16)) & MASK) ^ ((l & 3) << 30)) for l in range(64)]
            ors = [(g << 16) | (1 << (l & 15)) for l in range(64)]
            ands = [(~(1 << (l & 15))) & MASK for l in range(64)]
            xvals = [0xaa000000 | v for v in adds]
            or_acc = 0
            for v in ors:
                or_acc |= v
            and_acc = MASK
            for v in ands:
                and_acc &= v
            out[base + 0] = (0x10000000 + sum(adds)) & MASK
            out[base + 1] = (0x20000000 + sum(adds)) & MASK
            out[base + 2] = min([0xfffffffe] + minmax)
            out[base + 3] = max([1] + minmax)
            out[base + 4] = or_acc
            out[base + 5] = and_acc
            out[base + 6] = 64
            out[base + 7] = (0xdeadbeef + sum(xvals)) & MASK
            out[base + 8] = 0x77770010
            out[base + 9] = 0x110
            out[base + 10] = sum(0x100 + i for i in range(16)) & MASK
            out[base + 11] = 0
            out[base + 12] = 0
            out[base + 13] = 0x12345678
            for l in range(64):
                out[base + 16 + l] = 1
    elif name == 'R':
        for l in range(32):
            v5 = (0x7ffffff0 + l) & MASK
            v6 = (0x80000010 ^ l) & MASK
            v8 = (0x0f0f0000 ^ l) & MASK
            v9 = (0xf0f000ff ^ l) & MASK
            old_add = v5
            post_add = (old_add + 0x100 + l) & MASK
            old_min = v6
            post_min = min(old_min, (0x7ffffff8 + l) & MASK)
            old_max = v6
            post_max = max(old_max, (0x7ffffff8 + l) & MASK)
            old_or = v8
            post_or = old_or | 0x00f000aa
            old_and = v9
            post_and = old_and & 0xff0000f0
            old_xchg = v5
            post_xchg = (384 + 4 * l) & MASK
            out[8 * l:8 * l + 8] = [old_add, post_add, old_min, post_min, old_max, post_max, post_or, post_and]
            if l < 12:
                out[256 + 4 * l:256 + 4 * l + 4] = [old_xchg, post_xchg, old_or, old_and]
    else:
        raise KeyError(name)
    return out


def translate_case(name, code=None, rsrc2=None, group=None):
    _, words, default_rsrc2, local, _groups = CASES[name]
    return G.ComputePipeline(words if code is None else code, root_blob(),
                             default_rsrc2 if rsrc2 is None else rsrc2, local if group is None else group)


def single_ds_code(ds_bytes, *, with_exec=False, with_barrier=False, with_branch=False):
    code = bytearray(b''.join(raw for _, raw in PREAMBLE_LINES))
    code += v_mov_imm(1, 0) + v_mov_imm(2, 7) + v_mov_imm(3, 11)
    if with_branch:
        code += sopp(0x04, 2)  # s_cbranch_scc0 +2 (skip 8-byte DS instruction cleanly)
    if with_exec:
        code += vopc(0xc4, 0x81, 0) + bytes.fromhex('6a3c80be')  # v_cmp_gt_u32 vcc_lo, 1, v0; s_and_saveexec_b32 s0, vcc_lo
    code += ds_bytes
    if with_exec:
        code += bytes.fromhex('7e007e88')
    if with_barrier:
        code += bytes.fromhex('7fc08cbf00008abf')
    code += bytes.fromhex('000081bf')
    return bytes(code)


def check_controls():
    # 1. Decode every fixture and single-instruction snippet through disasm_rdna2 / G.decode.
    for line, hex_words in VERIFIED_SNIPPETS.items():
        (_, size, op, asm), = G.decode(bytes.fromhex(hex_words))
        assert size == 8 and op == line.split()[0] and asm == line, (line, op, asm)
    for name, (source, words, *_rest) in CASES.items():
        lines = source.splitlines()
        decoded = G.decode(words)
        assert len(decoded) == len(lines), (name, len(decoded), len(lines))
        for src_line, (_, _, dec_op, _dec_asm) in zip(lines, decoded):
            assert dec_op.removesuffix('_e32').removesuffix('_e64') == src_line.split()[0], (name, src_line, dec_op)

    hlsl_wide = translate_case('wide').hlsl_cs()
    assert mutate(hlsl_wide, 'wide-offset-times-four') != hlsl_wide
    for raw, diagnostic in (
        (ds_dual(0x4e, 3, 8, 10, offset1=1, gds=1), 'ds_write2_b64 needs two ordinary 8-bit offsets'),
        (ds_dual(0x77, 3, vdst=8), 'ds_read2_b64 needs two distinct offsets'),
        (ds_dual(0x4f, 3, 8, 10, offset1=1, vdst=1), 'ds_write2st64_b64 unused operand field set'),
    ):
        refused(diagnostic, lambda: translate_case('wide', code=single_ds_code(raw)).hlsl_cs())
    refused('ds_write2_b64 needs a nonzero LDS allocation',
            lambda: translate_case('wide', rsrc2=USER7).hlsl_cs())

    # 2. Verify HLSL lowering for Case D (dual-offset & stride-64 ops, EXEC guard, alignment & bounds).
    hlsl_d = translate_case('D').hlsl_cs()
    assert 'groupshared uint lds[512];' in hlsl_d, hlsl_d
    assert hlsl_d.count('GroupMemoryBarrierWithGroupSync();') == 1, hlsl_d
    assert 'if (((a & 3u) == 0u) && a < 2048u) lds[(a + 0u) >> 2] = v5;' in hlsl_d, hlsl_d
    assert 'if (((a & 3u) == 0u) && a < 1536u) lds[(a + 512u) >> 2] = v6;' in hlsl_d, hlsl_d
    assert 'if (((a & 3u) == 0u) && a < 2044u) lds[(a + 4u) >> 2] = v9;' in hlsl_d, hlsl_d
    assert 'if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) lds[(a + 1536u) >> 2] = v10;' in hlsl_d, hlsl_d
    assert 'lds[(a + 2048u) >> 2]' not in hlsl_d, hlsl_d  # static OOB offset1:8 omitted
    assert 'if ((exec & 1) && ((a & 3u) == 0u) && a < 256u) w = lds[(a + 1792u) >> 2];' in hlsl_d, hlsl_d

    # 3. Verify HLSL lowering for Case A (all 7 atomic ops, Interlocked*, EXEC guard, dual barriers).
    hlsl_a = translate_case('A').hlsl_cs()
    assert 'groupshared uint lds[128];' in hlsl_a, hlsl_a
    assert 'uint ao = 0;' in hlsl_a, hlsl_a
    assert hlsl_a.count('GroupMemoryBarrierWithGroupSync();') == 2, hlsl_a
    for snippet in (
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) InterlockedAdd(lds[(a + 0u) >> 2], v6);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 508u) InterlockedAdd(lds[(a + 4u) >> 2], v6, ao);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 504u) InterlockedMin(lds[(a + 8u) >> 2], v8);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 500u) InterlockedMax(lds[(a + 12u) >> 2], v8);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 496u) InterlockedOr(lds[(a + 16u) >> 2], v11);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 492u) InterlockedAnd(lds[(a + 20u) >> 2], v12);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 484u) InterlockedExchange(lds[(a + 28u) >> 2], v15, ao);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 476u) InterlockedAdd(lds[(a + 36u) >> 2], v2);',
        'if ((exec & 1) && ((a & 3u) == 0u) && a < 472u) InterlockedAdd(lds[(a + 40u) >> 2], v2, ao);',
        'if (exec & 1) v17 = ao;',
    ):
        assert snippet in hlsl_a, (snippet, hlsl_a)

    # 4. Verify HLSL lowering for Case R (all 6 _rtn atomics + aliasing, entry-full EXEC without exec guard).
    hlsl_r = translate_case('R').hlsl_cs()
    assert 'groupshared uint lds[256];' in hlsl_r, hlsl_r
    for snippet in (
        'if (((a & 3u) == 0u) && a < 1024u) InterlockedAdd(lds[(a + 0u) >> 2], v11, ao);',
        'if (((a & 3u) == 0u) && a < 896u) InterlockedMin(lds[(a + 128u) >> 2], v11, ao);',
        'if (((a & 3u) == 0u) && a < 768u) InterlockedMax(lds[(a + 256u) >> 2], v12, ao);',
        'if (((a & 3u) == 0u) && a < 512u) InterlockedOr(lds[(a + 512u) >> 2], v13, ao);',
        'if (((a & 3u) == 0u) && a < 256u) InterlockedAnd(lds[(a + 768u) >> 2], v14, ao);',
        'if (((a & 3u) == 0u) && a < 1024u) InterlockedExchange(lds[(a + 0u) >> 2], v15, ao);',
    ):
        assert snippet in hlsl_r, (snippet, hlsl_r)

    # 5. Non-rtn atomic shader must NOT declare unused `uint ao = 0;`.
    non_rtn_only = G.ComputePipeline(single_ds_code(ds_single(0x00, 1, 2, offset=4)), root_blob(),
                                     USER7 | (1 << 15), (32, 1, 1)).hlsl_cs()
    assert 'InterlockedAdd(lds[(a + 4u) >> 2], v2);' in non_rtn_only
    assert 'uint ao = 0;' not in non_rtn_only, non_rtn_only

    # 6. LDS_SIZE boundaries: 0 refused, 65 (>32 KiB) refused, 1 (512 B) and 64 (32 KiB) accepted, 2D/3D groups accepted.
    base = USER7 | TGID_X
    for name in ('D', 'A', 'R'):
        refused('at most 32 KiB', lambda n=name: translate_case(n, rsrc2=base | (0 << 15)))
        refused('at most 32 KiB', lambda n=name: translate_case(n, rsrc2=base | (65 << 15)))
    assert 'groupshared uint lds[8192];' in translate_case('D', rsrc2=base | (64 << 15)).hlsl_cs()
    assert 'groupshared uint lds[8192];' in translate_case('A', rsrc2=base | (64 << 15)).hlsl_cs()
    assert '[numthreads(8, 8, 1)]' in translate_case('D', group=(8, 8, 1)).hlsl_cs()

    # 7. Static OOB immediate offset on _rtn atomic sets dst = 0u without emitting OOB Interlocked*.
    oob_rtn = G.ComputePipeline(single_ds_code(ds_single(0x20, 1, 2, vdst=3, offset=512)), root_blob(),
                                USER7 | (1 << 15), (32, 1, 1)).hlsl_cs()
    assert 'Interlocked' not in oob_rtn and 'v3 = 0u;' in oob_rtn, oob_rtn
    oob_rtn_exec = G.ComputePipeline(single_ds_code(ds_single(0x20, 1, 2, vdst=3, offset=512), with_exec=True),
                                     root_blob(), USER7 | (1 << 15), (32, 1, 1)).hlsl_cs()
    assert 'if (exec & 1) v3 = 0u;' in oob_rtn_exec, oob_rtn_exec

    # 8. Dual aliasing refusals: offset0 == offset1 (both equal nonzero and both omitted = 0) on write2/write2st64/read2st64.
    for op, kw in ((0x0e, {'data0': 2, 'data1': 3}), (0x0f, {'data0': 2, 'data1': 3}), (0x38, {'vdst': 2})):
        for off in (0, 5):
            raw = ds_dual(op, 1, offset0=off, offset1=off, **kw)
            refused('distinct offsets', lambda r=raw: G.ComputePipeline(
                single_ds_code(r), root_blob(), USER7 | (4 << 15), (32, 1, 1)))

    # 9. GDS refusal on all 14 opcodes.
    for line, hex_words in VERIFIED_SNIPPETS.items():
        w0, w1 = struct.unpack('<2I', bytes.fromhex(hex_words))
        gds_raw = struct.pack('<2I', w0 | (1 << 17), w1)
        refused('offset', lambda r=gds_raw: G.ComputePipeline(
            single_ds_code(r), root_blob(), USER7 | (4 << 15), (32, 1, 1)))

    # 10. Binary encoding anomalies (reserved bit 16 and unused w1 operand fields) refused at decode/check_encoding.
    for line, hex_words in VERIFIED_SNIPPETS.items():
        op_name = line.split()[0]
        w0, w1 = struct.unpack('<2I', bytes.fromhex(hex_words))
        bit16_raw = struct.pack('<2I', w0 | (1 << 16), w1)
        refused('reserved bit 16', lambda r=bit16_raw: G.ComputePipeline(
            single_ds_code(r), root_blob(), USER7 | (4 << 15), (32, 1, 1)))
        unused_mask = G.LDS_UNUSED_W1_MASK[op_name]
        # Set the lowest bit in the unused w1 field mask.
        bad_bit = unused_mask & -unused_mask
        unused_raw = struct.pack('<2I', w0, w1 | bad_bit)
        refused('unused operand field', lambda r=unused_raw: G.ComputePipeline(
            single_ds_code(r), root_blob(), USER7 | (4 << 15), (32, 1, 1)))

    # 11. Unaligned immediate offsets (1, 2, 3) refused on single-offset LDS atomics.
    for op_code in (0x00, 0x07, 0x08, 0x09, 0x0a, 0x20, 0x27, 0x28, 0x29, 0x2a, 0x2d):
        vdst = 3 if op_code >= 0x20 else 0
        for bad_off in (1, 2, 3, 6):
            raw = ds_single(op_code, 1, 2, vdst=vdst, offset=bad_off)
            refused('4-byte aligned immediate offset', lambda r=raw: G.ComputePipeline(
                single_ds_code(r), root_blob(), USER7 | (1 << 15), (32, 1, 1)))

    # 12. Branched shader + s_barrier refused (while branched shader WITHOUT s_barrier translates!).
    branch_no_bar = single_ds_code(ds_single(0x00, 1, 2, offset=4), with_branch=True, with_barrier=False)
    assert 'InterlockedAdd(' in G.ComputePipeline(branch_no_bar, root_blob(), USER7 | (1 << 15), (32, 1, 1)).hlsl_cs()
    branch_with_bar = single_ds_code(ds_single(0x00, 1, 2, offset=4), with_branch=True, with_barrier=True)
    refused('straight-line', lambda: G.ComputePipeline(branch_with_bar, root_blob(), USER7 | (1 << 15), (32, 1, 1)))
    # Nonzero reserved immediate on s_barrier in a dual/atomic shader is refused.
    bad_bar = patch(WORDS_A, '00008abf', '01008abf') if WORDS_A.count(bytes.fromhex('00008abf')) == 1 else (
        WORDS_A.replace(bytes.fromhex('00008abf'), bytes.fromhex('01008abf'), 1))
    refused('s_barrier with unexpected', lambda: translate_case('A', code=bad_bar))

    # 13. Direct Stage operand, modifier, non-compute stage, and missing-metadata refusals.
    stage = translate_case('D').cs
    for bad_ops in ([], ['v1', 'v2'], ['v1', 'v2', 'v3', 'v4'], ['v[1:2]', 'v2', 'v3'], ['s1', 'v2', 'v3'], ['v1', 'v256', 'v3']):
        refused('three single VGPRs', lambda o=bad_ops: stage.lds_dual('ds_write2_b32', o, ['offset1:1']))
        refused('three single VGPRs', lambda o=bad_ops: stage.lds_dual('ds_write2st64_b32', o, ['offset1:1']))
    for bad_ops in ([], ['v[1:2]'], ['v1', 'v2'], ['v[1:3]', 'v2'], ['v[255:256]', 'v2'], ['v[1:2]', 's2']):
        refused('2-VGPR pair', lambda o=bad_ops: stage.lds_dual('ds_read2st64_b32', o, ['offset1:1']))
    refused('ordinary dword offsets', lambda: stage.lds_dual('ds_write2_b32', ['v1', 'v2', 'v3'], ['offset0:256', 'offset1:1']))
    refused('ordinary dword offsets', lambda: stage.lds_dual('ds_write2_b32', ['v1', 'v2', 'v3'], ['offset0:1', 'offset0:2']))
    refused('ordinary dword offsets', lambda: stage.lds_dual('ds_read2st64_b32', ['v[1:2]', 'v3'], ['offset:4']))
    refused('two single VGPRs', lambda: stage.lds_atomic('ds_add_u32', ['v1', 'v2', 'v3'], []))
    refused('two single VGPRs', lambda: stage.lds_atomic('ds_min_u32', ['v[1:2]', 'v3'], []))
    refused('three single VGPRs', lambda: stage.lds_atomic('ds_add_rtn_u32', ['v1', 'v2'], []))
    refused('three single VGPRs', lambda: stage.lds_atomic('ds_wrxchg_rtn_b32', ['v[1:2]', 'v2', 'v3'], []))
    refused('optional byte offset', lambda: stage.lds_atomic('ds_add_u32', ['v1', 'v2'], ['offset:65536']))
    refused('optional byte offset', lambda: stage.lds_atomic('ds_or_b32', ['v1', 'v2'], ['slc']))

    for stage_kind in ('vs', 'ps'):
        s = G.Stage.__new__(G.Stage)
        s.kind = stage_kind
        for op in G.LDS_DUAL_OPS:
            refused(op, lambda st=s, o=op: st.lower(o, o, ['v1', 'v2', 'v3'], ['offset1:1']))
        for op in G.LDS_ATOMIC_OPS:
            refused(op, lambda st=s, o=op: st.lower(o, o, ['v1', 'v2'], []))

    bare = G.Stage.__new__(G.Stage)
    bare.kind = 'cs'
    bare.pipe = None
    refused('metadata', lambda: bare.lds_dual('ds_write2_b32', ['v1', 'v2', 'v3'], ['offset1:1']))
    refused('metadata', lambda: bare.lds_atomic('ds_add_u32', ['v1', 'v2'], []))

    # 14. Out-of-scope DS ops stay refused by name.
    for op_code, op_name in ((0x05, 'ds_min_i32'), (0x06, 'ds_max_i32'), (0x0b, 'ds_xor_b32'),
                             (0x15, 'ds_add_f32'), (0x2e, 'ds_wrxchg2_rtn_b32')):
        raw = ds_dual(op_code, 1, 2, 4, vdst=8, offset0=1, offset1=2)
        refused(op_name, lambda r=raw: G.ComputePipeline(
            single_ds_code(r), root_blob(), USER7 | (4 << 15), (32, 1, 1)))

    # 15. D3D12 debug_layer status handling: enabled clean, enabled nonzero/missing, unavailable, missing status.
    assert check_debug_layer('D', 'backend=warp debug_layer=enabled\ndebug_errors=0 debug_warnings=0\n') == 'enabled'
    assert check_debug_layer('D', 'backend=hardware debug_layer=unavailable\ngroups=1x1x1 compared_words=320 mismatches=0\n') == 'unavailable'
    for bad_out, expected_msg in (
        ('backend=warp debug_layer=enabled\ndebug_errors=1 debug_warnings=0\n', 'missing zero debug errors/warnings'),
        ('backend=warp debug_layer=enabled\ndebug_errors=0 debug_warnings=2\n', 'missing zero debug errors/warnings'),
        ('backend=warp debug_layer=enabled\ndebug_errors=0 debug_warnings=01\n', 'missing zero debug errors/warnings'),
        ('backend=warp debug_layer=enabled\ngroups=1x1x1 compared_words=320 mismatches=0\n', 'missing zero debug errors/warnings'),
        ('backend=hardware\ngroups=1x1x1 compared_words=320 mismatches=0\n', 'missing debug_layer status'),
    ):
        try:
            check_debug_layer('D', bad_out)
        except AssertionError as err:
            assert expected_msg in str(err), (bad_out, str(err))
        else:
            raise AssertionError('expected debug_layer rejection for %r' % (bad_out,))

    print('dual-offset/st64 LDS and 32-bit LDS atomic controls (18 ops, encodings, bounds, aliasing, EXEC, debug_layer, refusals) pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    for name, (source, words, *_rest) in CASES.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for line, hex_words in VERIFIED_SNIPPETS.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == hex_words, '%s: assembled %s, recorded %s' % (line, built, hex_words)
    print('LDS dual/atomic fixture words match llvm-mc (%d programs, %d snippets)'
          % (len(CASES), len(VERIFIED_SNIPPETS)))


# Shader mutant controls: each must fail on WARP and hardware at the expected word with debug_errors=0.
CONTROLS = {
    'wide-offset-times-four': ('wide', 'lds[(a + 8u) >> 2] = v10;\n'
                               '    if (((a & 3u) == 0u) && a < 2036u) lds[(a + 12u) >> 2] = v11;',
                               'lds[(a + 4u) >> 2] = v10;\n'
                               '    if (((a & 3u) == 0u) && a < 2036u) lds[(a + 8u) >> 2] = v11;'),
    'd-wrong-st64-stride': ('D', 'if (((a & 3u) == 0u) && a < 1536u) lds[(a + 512u) >> 2] = v6;\n',
                            'if (((a & 3u) == 0u) && a < 2040u) lds[(a + 8u) >> 2] = v6;\n'),
    'd-drop-oob-zero': ('D', '    w = 0u;\n    if ((exec & 1) && ((a & 3u) == 0u) && a < 256u) w = lds[(a + 1792u) >> 2];\n',
                        '    w = 1u;\n    if ((exec & 1) && ((a & 3u) == 0u) && a < 256u) w = lds[(a + 1792u) >> 2];\n'),
    'd-drop-align': ('D', 'if ((exec & 1) && ((a & 3u) == 0u) && a < 2048u) lds[(a + 0u) >> 2] = v11;\n',
                     'if ((exec & 1) && a < 2048u) lds[(a + 0u) >> 2] = v11;\n'),
    'd-drop-exec': ('D', 'if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) lds[(a + 1536u) >> 2] = v10;\n',
                    'if (((a & 3u) == 0u) && a < 512u) lds[(a + 1536u) >> 2] = v10;\n'),
    'd-alias-overwrite': ('D', '    if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) w = lds[(a + 1536u) >> 2];\n',
                          '    if ((exec & 1) && ((v14 & 3u) == 0u) && v14 < 512u) w = lds[(v14 + 1536u) >> 2];\n'),
    'a-swap-minmax': ('A', 'InterlockedMin(lds[(a + 8u) >> 2], v8);',
                      'InterlockedMax(lds[(a + 8u) >> 2], v8);'),
    'a-drop-exec': ('A', 'if ((exec & 1) && ((a & 3u) == 0u) && a < 476u) InterlockedAdd(lds[(a + 36u) >> 2], v2);',
                    'if (((a & 3u) == 0u) && a < 476u) InterlockedAdd(lds[(a + 36u) >> 2], v2);'),
    'a-drop-wrxchg-rtn': ('A', 'InterlockedExchange(lds[(a + 28u) >> 2], v15, ao);\n    if (exec & 1) v16 = ao;',
                          'InterlockedExchange(lds[(a + 28u) >> 2], v15, ao);\n    if (exec & 1) v16 = 0u;'),
    'a-oob-nonzero': ('A', '    a = v18;\n    ao = 0u;\n    if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) InterlockedAdd(lds[(a + 0u) >> 2], v2, ao);',
                      '    a = v18;\n    ao = 1u;\n    if ((exec & 1) && ((a & 3u) == 0u) && a < 512u) InterlockedAdd(lds[(a + 0u) >> 2], v2, ao);'),
    'r-drop-add-rtn': ('R', 'InterlockedAdd(lds[(a + 0u) >> 2], v11, ao);\n    v10 = ao;',
                       'InterlockedAdd(lds[(a + 0u) >> 2], v11, ao);\n    v10 = v11;'),
}


def mutate(hlsl, control):
    _, old, new = CONTROLS[control]
    assert hlsl.count(old) == 1, (control, hlsl)
    return hlsl.replace(old, new)


def run_case(name, dxc, runner, adapter, directory, control=None):
    hlsl = translate_case(name).hlsl_cs()
    if control:
        hlsl = mutate(hlsl, control)
    return run_shader(name, dxc, runner, adapter, directory, hlsl, expected_words(name), CASES[name][4])


def run_shader(name, dxc, runner, adapter, directory, hlsl, expected, groups=(1, 1, 1)):
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-Gis', '-WX', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    constants = [0] * 16
    case.write_bytes(
        struct.pack('<I', groups[0] | (groups[1] << 8) | (groups[2] << 16))
        + struct.pack('<%dI' % WORDS, *expected)
        + struct.pack('<16I', *constants)
    )
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_debug_layer(name, stdout, stderr=''):
    if 'debug_layer=enabled' in stdout:
        if not re.search(r'\bdebug_errors=0 debug_warnings=0\b', stdout):
            raise AssertionError('case %s: missing zero debug errors/warnings: %s %s' % (name, stdout, stderr))
        return 'enabled'
    if 'debug_layer=unavailable' in stdout:
        return 'unavailable'
    raise AssertionError('case %s: missing debug_layer status: %s %s' % (name, stdout, stderr))


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
            if check_debug_layer(name, run.stdout, run.stderr) == 'unavailable':
                print('case %s: debug_layer=unavailable (word readback still verified)' % name)
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if control:
                if run.returncode != 1 or not diagnostics or not re.fullmatch(
                        r'word=\d+ expected=[0-9a-f]{8} actual=[0-9a-f]{8}', diagnostics[0]):
                    raise AssertionError('control %s: not rejected (exit %d) %s' % (control, run.returncode, run.stderr))
                if control == 'wide-offset-times-four':
                    assert diagnostics[0] == 'word=1 expected=fa040001 actual=437e0001', diagnostics
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
