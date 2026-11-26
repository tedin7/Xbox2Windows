#!/usr/bin/env python3
"""Verify compute buffer_store_format_d16_x lowering in tools/startup/gfx2hlsl.py (Refs #2274).

  (no arguments)                                     pure-Python unit, format, bounds and refusal checks
  binary <llvm-mc>                                   reassemble fixture sources and verify machine words
  dispatch <dxc> <offscreen_check> <warp|hardware>   translate, compile DXC cs_6_0, dispatch and verify readback
  control <name> <dxc> <offscreen_check> <adapter>   dispatch a mutated translation that must mismatch

The fixtures are source-built gfx1013 compute programs:
- Case F: 16 lanes read poisoned-high-half inputs from t0 and store via `buffer_store_format_d16_x ... idxen offset:4`.
  Dispatched across all six supported formats (32_FLOAT, 16_FLOAT, 16_UNORM, 16_SNORM legacy, 16_UINT, 16_SINT legacy)
  with OOB_SELECT=0, stride=8, num_records=12 (so lanes 12..15 are OOB no-ops preserving sentinel), plus no-op checks
  for unsupported FORMAT, OOB_SELECT=1/2, OOB_SELECT=0 with stride=0, OOB_SELECT=3 with stride=8, and DW1 swizzle bits.
- Case P: 32 lanes concurrently write adjacent low/high 16-bit halves of the same dwords using `offen`, SGPR `soffset`
  (`s4`), `offset:2`, OOB_SELECT=3, `stride=0` and `num_records=50` (raw byte bounds; lanes 23..31 are OOB no-ops).
- Case S: 16 lanes store via structured OOB_SELECT=0 with SGPR `soffset` (`s4=16 > stride=8`) and `offset:4`, proving
  RDNA 2 ISA separation of `soffset` from the record offset checked against `stride`.
- Case E: 16 lanes narrow EXEC to tid.x < 10, branch with `s_cbranch_execz`, locally reload the UAV descriptor, narrow
  EXEC again to tid.x < 6 via `s_and_saveexec_b32`, and store via `idxen offen` to `BUF_FMT_16_FLOAT` (OOB_SELECT=0);
  inactive lanes 6..15 must preserve sentinel.
- Case C: 256 concurrent threads (4 groups x 64 lanes) repeatedly write competing values ({0x1111, 0x2222} to byte 0,
  {0x4444, 0x8888} to byte 2) to the same 16-bit halfwords, verifying atomic CAS prevents bitwise-OR or torn corruption.
- Case B / B-short: 4 lanes write 3 in-bounds 16-bit halves at addr = 0, 2, 4 under guest OOB_SELECT=3, stride=0,
  num_records=6. Case B binds a padded host raw UAV (uav_words=2, 8 bytes) so DWORD 0 updates and DWORD 1 updates only
  its low half while preserving the high-half sentinel; Case B-short binds the same shader/inputs to a short host raw
  UAV (uav_words=1, 4 bytes) proving the host backing guard exits before Load/InterlockedCompareExchange on DWORD 1.
"""

import math
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
from startup_gfx2hlsl_compute_check import SENTINEL, USER7, WORDS, patch, refused, root_blob  # noqa: E402

MUBUF_STORE_FORMAT_D16_X = 0x84


def encode_mubuf(op, vdata, vaddr, srsrc, soffset_enc, *, idxen=False, offen=False, offset=0,
                 glc=False, slc=False, dlc=False, tfe=False, lds=False, w0_rsvd=False, w1_rsvd=False):
    w0 = (0xe0000000 | ((op & 0xff) << 18) | (int(w0_rsvd) << 17) | (int(lds) << 16)
          | (int(dlc) << 15) | (int(glc) << 14) | (int(idxen) << 13) | (int(offen) << 12) | (offset & 0xfff))
    w1 = (((soffset_enc & 0xff) << 24) | (int(tfe) << 23) | (int(slc) << 22) | (int(w1_rsvd) << 21)
          | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff))
    return struct.pack('<II', w0, w1)


def desc_word3(fmt_code, *, legacy=False, oob_select=0, dst_sel=0xFAC):
    if legacy:
        field = next(k for k, v in G.STORE_D16_X_LEGACY_FORMATS.items() if v == fmt_code)
        rsrc_level = 0
    else:
        field = fmt_code
        rsrc_level = 1
    return ((oob_select & 3) << 28) | (rsrc_level << 24) | ((field & 0x7f) << 12) | (dst_sel & 0xfff)


PREAMBLE_X8 = (
    's_mov_b32 s10, s6\n'
    's_mov_b32 s11, 0\n'
    's_load_dwordx8 s[12:19], s[10:11], 0x0\n'
    's_waitcnt lgkmcnt(0)\n'
)

SOURCE_F = PREAMBLE_X8 + """\
v_lshlrev_b32 v4, 2, v0
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen offset:4
s_endpgm
"""
WORDS_F = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
    '82000834' '001030e004050380' '703f8cbf'
    '042010e200050480' '000081bf'
)

SOURCE_P = PREAMBLE_X8 + """\
v_lshlrev_b32 v4, 2, v0
buffer_load_dword v5, v4, s[12:15], 0 offen
v_lshlrev_b32 v1, 1, v0
s_waitcnt vmcnt(0)
buffer_store_format_d16_x v5, v1, s[16:19], s4 offen offset:2
s_endpgm
"""
WORDS_P = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
    '82000834' '001030e004050380' '81000234' '703f8cbf'
    '021010e201050404' '000081bf'
)

SOURCE_S = PREAMBLE_X8 + """\
v_lshlrev_b32 v4, 2, v0
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
buffer_store_format_d16_x v5, v0, s[16:19], s4 idxen offset:4
s_endpgm
"""
WORDS_S = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
    '82000834' '001030e004050380' '703f8cbf'
    '042010e200050404' '000081bf'
)

SOURCE_E = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[12:15], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v4, 2, v0
s_mov_b32 s21, exec_lo
v_cmp_gt_u32 vcc_lo, 10, v0
s_mov_b32 exec_lo, vcc_lo
s_cbranch_execz 14
s_mov_b32 s22, s6
s_mov_b32 s23, 0
s_load_dwordx4 s[16:19], s[22:23], 0x10
buffer_load_dword v5, v4, s[12:15], 0 offen
v_mov_b32 v2, v0
v_mov_b32 v3, 2
s_waitcnt vmcnt(0) lgkmcnt(0)
v_cmp_gt_u32 vcc_lo, 6, v0
s_and_saveexec_b32 s0, vcc_lo
buffer_store_format_d16_x v5, v[2:3], s[16:19], 0 idxen offen
s_or_b32 exec_lo, exec_lo, s0
s_mov_b32 exec_lo, s21
s_endpgm
"""
WORDS_E = bytes.fromhex(
    '06038abe' '80038bbe' '050308f4000000fa' '7fc08cbf'
    '82000834' '7e0395be' '8a00887d' '6a03febe' '0e0088bf'
    '060396be' '800397be' '0b0408f4100000fa' '001030e004050380'
    '0003047e' '8202067e' '70008cbf' '8600887d' '6a3c80be'
    '003010e202050480' '7e007e88' '1503febe' '000081bf'
)

SOURCE_C = PREAMBLE_X8 + """\
v_and_b32 v1, 1, v0
v_lshlrev_b32 v1, 2, v1
v_add_nc_u32 v6, 8, v1
buffer_load_dword v2, v1, s[12:15], 0 offen
buffer_load_dword v3, v6, s[12:15], 0 offen
s_waitcnt vmcnt(0)
buffer_store_format_d16_x v2, off, s[16:19], 0
buffer_store_format_d16_x v3, off, s[16:19], 0 offset:2
buffer_store_format_d16_x v2, off, s[16:19], 0
buffer_store_format_d16_x v3, off, s[16:19], 0 offset:2
buffer_store_format_d16_x v2, off, s[16:19], 0
buffer_store_format_d16_x v3, off, s[16:19], 0 offset:2
buffer_store_format_d16_x v2, off, s[16:19], 0
buffer_store_format_d16_x v3, off, s[16:19], 0 offset:2
s_endpgm
"""
WORDS_C = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
    '81000236' '82020234' '88020c4a'
    '001030e001020380' '001030e006030380' '703f8cbf'
    + ('000010e200020480' '020010e200030480') * 4
    + '000081bf'
)

SOURCE_B = PREAMBLE_X8 + """\
v_lshlrev_b32 v4, 2, v0
buffer_load_dword v5, v4, s[12:15], 0 offen
v_lshlrev_b32 v1, 1, v0
s_waitcnt vmcnt(0)
buffer_store_format_d16_x v5, v1, s[16:19], 0 offen
s_endpgm
"""
WORDS_B = bytes.fromhex(
    '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf'
    '82000834' '001030e004050380' '81000234' '703f8cbf'
    '001010e201050480' '000081bf'
)

VERIFIED_SNIPPETS = {
    'buffer_store_format_d16_x v5, off, s[16:19], 0 offset:4': '040010e200050480',
    'buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen offset:4': '042010e200050480',
    'buffer_store_format_d16_x v5, v0, s[16:19], s4 idxen offset:4': '042010e200050404',
    'buffer_store_format_d16_x v5, v1, s[16:19], 0 offen': '001010e201050480',
    'buffer_store_format_d16_x v5, v1, s[16:19], s4 offen offset:2': '021010e201050404',
    'buffer_store_format_d16_x v5, v[2:3], s[16:19], 0 idxen offen': '003010e202050480',
    'buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen glc': '006010e200050480',
    'buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen slc': '002010e200054480',
    'buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen dlc': '00a010e200050480',
    'buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen tfe': '002010e200058480',
}

# 16 FP16 patterns covering 0, -0, normal values, subnormal, max, +/-inf, NaN, out-of-range norm inputs,
# each paired with distinct non-zero poison upper 16 bits that buffer_store_format_d16_x must ignore.
FP16_VALUES = (
    0x0000,  # +0.0
    0x8000,  # -0.0
    0x3400,  # +0.25
    0x3800,  # +0.5  (exact UNORM16/SNORM16 tie)
    0xb800,  # -0.5  (exact negative SNORM16 tie)
    0x3c00,  # +1.0
    0xbc00,  # -1.0
    0x4000,  # +2.0  (clamps to 1.0 for UNORM/SNORM)
    0xc000,  # -2.0  (clamps to 0.0 UNORM / -1.0 SNORM)
    0x0001,  # smallest positive binary16 subnormal (2^-24)
    0x7bff,  # max finite binary16 (65504.0)
    0x7e00,  # quiet NaN (converts to 0 for UNORM/SNORM; bit-preserved for 16_FLOAT)
    0x3666,  # lane 12 (OOB in F / S)
    0x3a66,  # lane 13 (OOB in F / S)
    0x7c00,  # lane 14 (OOB in F / S)
    0xfc00,  # lane 15 (OOB in F / S)
)
INT16_VALUES = (
    0x0000, 0x0001, 0x1234, 0x7fff, 0x8000, 0x8001, 0xabcd, 0xfedc,
    0xfffe, 0xffff, 0x4321, 0x0555, 0x1111, 0x2222, 0x3333, 0x4444,
)

# F-series subcases all share WORDS_F and vary only entry constants (word3, uav_stride = DW1 >> 16, num_records).
F_VARIANTS = {
    'F-f32':           (desc_word3(22, oob_select=0, dst_sel=0x204), 8,          12, 'fp16',  True),
    'F-f16':           (desc_word3(13, oob_select=0, dst_sel=0xFAC), 8,          12, 'fp16',  True),
    'F-unorm16':       (desc_word3(7,  oob_select=0, dst_sel=0x204), 8,          12, 'fp16',  True),
    'F-snorm16':       (desc_word3(8,  legacy=True, oob_select=0),   8,          12, 'fp16',  True),
    'F-uint16':        (desc_word3(11, oob_select=0, dst_sel=0xFAC), 8,          12, 'int16', True),
    'F-sint16':        (desc_word3(12, legacy=True, oob_select=0),   8,          12, 'int16', True),
    # Refused / no-op descriptor configurations (must preserve all 320 sentinel words):
    'F-unsupported':   (desc_word3(29, oob_select=0, dst_sel=0xFAC), 8,          12, 'fp16',  False),  # 16_16_FLOAT
    'F-oob1':          (desc_word3(22, oob_select=1, dst_sel=0x204), 8,          12, 'fp16',  False),  # INDEX_ONLY
    'F-oob2':          (desc_word3(22, oob_select=2, dst_sel=0x204), 8,          12, 'fp16',  False),  # DISABLED
    'F-oob0-stride0':  (desc_word3(22, oob_select=0, dst_sel=0x204), 0,          64, 'fp16',  False),  # structured + stride=0
    'F-oob3-stride8':  (desc_word3(22, oob_select=3, dst_sel=0x204), 8,          64, 'fp16',  False),  # raw + stride!=0
    'F-swizzle-en':    (desc_word3(22, oob_select=0, dst_sel=0x204), 8 | 0x8000, 12, 'fp16',  False),  # DW1 bit 31
    'F-cache-swizzle': (desc_word3(22, oob_select=0, dst_sel=0x204), 8 | 0x4000, 12, 'fp16',  False),  # DW1 bit 30
}

PROGRAMS = {
    'F': (SOURCE_F, WORDS_F, USER7, (16, 1, 1), (1, 1, 1)),
    'P': (SOURCE_P, WORDS_P, USER7, (32, 1, 1), (1, 1, 1)),
    'S': (SOURCE_S, WORDS_S, USER7, (16, 1, 1), (1, 1, 1)),
    'E': (SOURCE_E, WORDS_E, USER7, (16, 1, 1), (1, 1, 1)),
    'C': (SOURCE_C, WORDS_C, USER7, (64, 1, 1), (4, 1, 1)),
    'B': (SOURCE_B, WORDS_B, USER7, (4, 1, 1), (1, 1, 1)),
}

DISPATCH_CASES = (*F_VARIANTS.keys(), 'P', 'S', 'E', 'B', 'B-short')
CONTENTION_VALID_LO = (0x1111, 0x2222)
CONTENTION_VALID_HI = (0x4444, 0x8888)


def f16_to_f32_bits(h):
    val = struct.unpack('<e', struct.pack('<H', h & 0xffff))[0]
    return struct.unpack('<I', struct.pack('<f', val))[0]


def f16_to_unorm16(h):
    val = struct.unpack('<e', struct.pack('<H', h & 0xffff))[0]
    if math.isnan(val):
        return 0
    clamped = min(1.0, max(0.0, float(val)))
    scaled = struct.unpack('<f', struct.pack('<f', clamped * 65535.0))[0]
    return int(math.floor(struct.unpack('<f', struct.pack('<f', scaled + 0.5))[0])) & 0xffff


def f16_to_snorm16(h):
    val = struct.unpack('<e', struct.pack('<H', h & 0xffff))[0]
    if math.isnan(val):
        return 0
    clamped = min(1.0, max(-1.0, float(val)))
    scaled = struct.unpack('<f', struct.pack('<f', clamped * 32767.0))[0]
    rounded = (math.floor(struct.unpack('<f', struct.pack('<f', scaled + 0.5))[0]) if scaled >= 0.0
               else math.ceil(struct.unpack('<f', struct.pack('<f', scaled - 0.5))[0]))
    return int(rounded) & 0xffff


def poisoned_source(halves):
    words = [0x00000000] * WORDS
    for i, h in enumerate(halves):
        poison = (0xa500 + i * 0x111) & 0xffff
        words[i] = (poison << 16) | (h & 0xffff)
    return struct.pack('<%dI' % WORDS, *words)


def put_half(buf, byte_addr, half):
    base = byte_addr & ~3
    shift = (byte_addr & 2) << 3
    dword = struct.unpack_from('<I', buf, base)[0]
    dword = (dword & ~(0xffff << shift)) | ((half & 0xffff) << shift)
    struct.pack_into('<I', buf, base, dword)


def case_inputs(name):
    """Return (program_key, source_bytes, constants_16, expected_words_320, uav_words)."""
    buf = bytearray(struct.pack('<%dI' % WORDS, *([SENTINEL] * WORDS)))
    if name in F_VARIANTS:
        word3, stride, num_records, kind, active = F_VARIANTS[name]
        halves = FP16_VALUES if kind == 'fp16' else INT16_VALUES
        if name == 'F-f32':
            halves = tuple(0x7c00 if i == 11 else h for i, h in enumerate(halves))
        source = poisoned_source(halves)
        constants = [0, 0, word3, stride, num_records] + [0] * 11
        if active:
            code, _, _ = G.validate_store_d16_x_word3(word3)
            for lane in range(num_records):
                addr = lane * stride + 4
                h = halves[lane]
                if code == 22:
                    struct.pack_into('<I', buf, addr, f16_to_f32_bits(h))
                elif code == 7:
                    put_half(buf, addr, f16_to_unorm16(h))
                elif code == 8:
                    put_half(buf, addr, f16_to_snorm16(h))
                else:
                    put_half(buf, addr, h)
        return 'F', source, constants, list(struct.unpack('<%dI' % WORDS, buf)), None

    if name == 'P':
        # 32 lanes, s4 = 2 (soffset), offset:2 -> base byte offset 4; lane i writes at 4 + 2*i.
        # OOB_SELECT=3 (raw), stride = 0, num_records = 50 -> lanes 0..22 write bytes 4..49,
        # lanes 23..31 are OOB no-ops preserving sentinel (including high half of dword 12).
        halves = tuple(((0x1000 + i * 0x0137) & 0xffff) for i in range(32))
        source = poisoned_source(halves)
        word3 = desc_word3(11, oob_select=3, dst_sel=0xFAC)  # BUF_FMT_16_UINT, OOB_SELECT=3 (RAW)
        constants = [2, 0, word3, 0, 50] + [0] * 11
        for lane in range(32):
            addr = 4 + 2 * lane
            if addr + 2 <= 50:
                put_half(buf, addr, halves[lane])
        return 'P', source, constants, list(struct.unpack('<%dI' % WORDS, buf)), None

    if name == 'S':
        # Structured OOB_SELECT=0 with soffset (s4=16) > stride (8) and imm_off=4.
        # RDNA 2 ISA: rec_off=4 <= stride-4 (in-bounds), addr = 16 + idx*8 + 4 for idx < 12.
        halves = tuple(0x7c00 if i == 11 else h for i, h in enumerate(FP16_VALUES))
        source = poisoned_source(halves)
        word3 = desc_word3(22, oob_select=0, dst_sel=0x204)  # BUF_FMT_32_FLOAT, OOB_SELECT=0
        constants = [16, 0, word3, 8, 12] + [0] * 11
        for lane in range(12):
            struct.pack_into('<I', buf, 16 + lane * 8 + 4, f16_to_f32_bits(halves[lane]))
        return 'S', source, constants, list(struct.unpack('<%dI' % WORDS, buf)), None

    if name == 'E':
        halves = FP16_VALUES
        source = poisoned_source(halves)
        word3 = desc_word3(13, oob_select=0, dst_sel=0x204)  # BUF_FMT_16_FLOAT, OOB_SELECT=0
        constants = [0, 0, word3, 8, 16] + [0] * 11
        for lane in range(6):  # Outer EXEC tid.x < 10, inner s_and_saveexec tid.x < 6
            put_half(buf, lane * 8 + 2, halves[lane])
        return 'E', source, constants, list(struct.unpack('<%dI' % WORDS, buf)), None

    if name in ('B', 'B-short'):
        # Host-backing boundary check separate from guest num_records/stride:
        # guest OOB_SELECT=3, stride=0, num_records=6 lets lanes 0..2 pass (addr = 0, 2, 4; addr + 2 <= 6).
        # Case B binds a padded 2-word (8-byte) raw UAV -> word 0 updated, word 1 low half updated (high half sentinel).
        # Case B-short binds a short 1-word (4-byte) raw UAV -> word 0 updated, word 1 untouched sentinel.
        halves = (0x1234, 0x5678, 0x9abc, 0xdef0)
        source = poisoned_source(halves)
        word3 = desc_word3(11, oob_select=3, dst_sel=0xFAC)  # BUF_FMT_16_UINT, OOB_SELECT=3 (RAW)
        constants = [0, 0, word3, 0, 6] + [0] * 11
        uav_words = 2 if name == 'B' else 1
        uav_bytes = uav_words * 4
        for lane in range(4):
            addr = 2 * lane
            if addr + 2 <= 6 and (addr & ~3) + 4 <= uav_bytes:
                put_half(buf, addr, halves[lane])
        return 'B', source, constants, list(struct.unpack('<%dI' % WORDS, buf)), uav_words

    if name == 'C':
        # Even lanes read t0[0]=0x1111, t0[2]=0x4444; odd lanes read t0[1]=0x2222, t0[3]=0x8888.
        source = poisoned_source((0x1111, 0x2222, 0x4444, 0x8888))
        word3 = desc_word3(11, oob_select=3, dst_sel=0xFAC)  # BUF_FMT_16_UINT, OOB_SELECT=3 (RAW)
        constants = [0, 0, word3, 0, 4] + [0] * 11
        # Sentinel 0xdeadbeef at word 0 forces offscreen_check to report actual word 0 while asserting words 1..319 == SENTINEL.
        exp = [SENTINEL] * WORDS
        exp[0] = 0xdeadbeef
        return 'C', source, constants, exp, None

    raise KeyError(name)


def translate(prog_key, code=None, rsrc2=None, group=None):
    _, words, default_rsrc2, local, _ = PROGRAMS[prog_key]
    return G.ComputePipeline(words if code is None else code, root_blob(),
                             default_rsrc2 if rsrc2 is None else rsrc2, local if group is None else group)


def check_controls():
    # 1. Verify decoder round-trip for every verified snippet.
    for line, hex_words in VERIFIED_SNIPPETS.items():
        (_, size, op, asm), = G.decode(bytes.fromhex(hex_words))
        assert (size, op, asm) == (8, 'buffer_store_format_d16_x', line), (line, size, op, asm)

    # 2. Verify validate_store_d16_x_word3 accepts all 6 unified + 6 legacy formats, OOB 0/3, 0xFAC and 0x204.
    for code, (fmt_name, elem_bytes) in G.STORE_D16_X_FORMATS.items():
        for legacy in (False, True):
            for oob in (0, 3):
                for sel in (0xFAC, 0x204):
                    w3 = desc_word3(code, legacy=legacy, oob_select=oob, dst_sel=sel)
                    assert G.validate_store_d16_x_word3(w3) == (code, fmt_name, elem_bytes), hex(w3)

    # 3. Verify validate_store_d16_x_word3 refuses unsupported formats, flags, DST_SEL_X and OOB_SELECT 1/2.
    for bad_w3, frag in (
        (desc_word3(0), 'unsupported descriptor FORMAT'),
        (desc_word3(1), 'unsupported descriptor FORMAT'),
        (desc_word3(5), 'unsupported descriptor FORMAT'),
        (desc_word3(20), 'unsupported descriptor FORMAT'),
        (desc_word3(23), 'unsupported descriptor FORMAT'),
        (desc_word3(29), 'unsupported descriptor FORMAT'),
        (desc_word3(56), 'unsupported descriptor FORMAT'),
        (desc_word3(77), 'unsupported descriptor FORMAT'),
        ((0 << 24) | (1 << 12) | 0xFAC, 'unsupported descriptor FORMAT'),
        (desc_word3(22, dst_sel=0xFA0), 'non-identity DST_SEL_X'),
        (desc_word3(22, dst_sel=0xFA5), 'non-identity DST_SEL_X'),
        (desc_word3(22, oob_select=1), 'OOB_SELECT 1'),
        (desc_word3(22, oob_select=2), 'OOB_SELECT 2'),
        (desc_word3(22) | (1 << 23), 'flags'),
        (desc_word3(22) | (1 << 21), 'flags'),
        (desc_word3(22) | (1 << 30), 'flags'),
        (desc_word3(22) | (1 << 19), 'flags'),
        (desc_word3(22) | (1 << 25), 'flags'),
        (-1, 'invalid descriptor word 3'),
        (True, 'invalid descriptor word 3'),
        (1 << 32, 'invalid descriptor word 3'),
    ):
        refused(frag, lambda w=bad_w3: G.validate_store_d16_x_word3(w))

    # 4. Verify manifests, CAS atomic helper (not InterlockedAnd/Or), and soffset/imm_offset separation.
    pipe_f = translate('F')
    mf_f = pipe_f.manifest()
    hlsl_f = pipe_f.hlsl_cs()
    assert mf_f == (
        'x2w-manifest 2\nstage cs\ngroup 16 1 1\nud 5\n'
        'udsrc 0 rootconst 0 0\nudsrc 1 rootconst 0 1\n'
        'udsrc 2 uav_descword3 0\nudsrc 3 uav_stride 0\nudsrc 4 uav_numrecords 0\n'
        'srv 0 buf table 1 0\nuav 0 table 1 16\n'
    ), mf_f
    assert 'void store_format_d16_x(' in hlsl_f
    assert 'buf.InterlockedCompareExchange(' in hlsl_f
    assert 'InterlockedAnd' not in hlsl_f and 'InterlockedOr' not in hlsl_f
    assert '    store_format_d16_x(u0, ud2, ud3, ud4, v0, 0u, 0u, 4u, v5);' in hlsl_f, hlsl_f
    dim_pos = hlsl_f.index('buf.GetDimensions(uav_bytes);')
    store_guard_pos = hlsl_f.index('if ((addr & 3u) != 0u || uav_bytes < 4u || addr > uav_bytes - 4u) return;')
    store_pos = hlsl_f.index('buf.Store(addr, asuint(f16tof32(h)));')
    cas_guard_pos = hlsl_f.index('if (uav_bytes < 4u || base > uav_bytes - 4u) return;')
    load_pos = hlsl_f.index('buf.Load(base)')
    cas_pos = hlsl_f.index('buf.InterlockedCompareExchange(')
    assert dim_pos < store_guard_pos < store_pos < cas_guard_pos < load_pos < cas_pos, (
        dim_pos, store_guard_pos, store_pos, cas_guard_pos, load_pos, cas_pos)

    hlsl_p = translate('P').hlsl_cs()
    assert '    store_format_d16_x(u0, ud2, ud3, ud4, 0u, v1, ud0, 2u, v5);' in hlsl_p, hlsl_p

    hlsl_s = translate('S').hlsl_cs()
    assert '    store_format_d16_x(u0, ud2, ud3, ud4, v0, 0u, ud0, 4u, v5);' in hlsl_s, hlsl_s

    hlsl_e = translate('E').hlsl_cs()
    assert 'if (exec & 1) store_format_d16_x(u0, ud2, ud3, ud4, v2, v3, 0u, 0u, v5);' in hlsl_e, hlsl_e

    hlsl_b = translate('B').hlsl_cs()
    assert '    store_format_d16_x(u0, ud2, ud3, ud4, 0u, v1, 0u, 0u, v5);' in hlsl_b, hlsl_b

    # 'off' addressing mode with inline soffset=8 and offset:4 must keep soff=8u and imm_off=4u SEPARATE!
    store_f = '042010e200050480'
    off_store = encode_mubuf(MUBUF_STORE_FORMAT_D16_X, 5, 0, 16, 0x80 + 8, offset=4)
    hlsl_off = translate('F', code=patch(WORDS_F, store_f, off_store.hex())).hlsl_cs()
    assert 'store_format_d16_x(u0, ud2, ud3, ud4, 0u, 0u, 8u, 4u, v5);' in hlsl_off, hlsl_off

    # Shaders without buffer_store_format_d16_x must NOT emit the helper.
    plain = G.ComputePipeline(bytes.fromhex('000081bf'), root_blob(), USER7, (1, 1, 1)).hlsl_cs()
    assert 'store_format_d16_x' not in plain

    # 5. Refusal checks: unsupported modifiers (glc, slc, dlc, tfe) AND binary lds=1 / reserved bits.
    for mod_name, hex_word in (
        ('modifier glc', VERIFIED_SNIPPETS['buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen glc']),
        ('modifier slc', VERIFIED_SNIPPETS['buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen slc']),
        ('modifier dlc', VERIFIED_SNIPPETS['buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen dlc']),
        ('modifier tfe', VERIFIED_SNIPPETS['buffer_store_format_d16_x v5, v0, s[16:19], 0 idxen tfe']),
        ('illegal MUBUF encoding bits', encode_mubuf(MUBUF_STORE_FORMAT_D16_X, 5, 0, 16, 0x80, idxen=True, offset=4, lds=True).hex()),
        ('illegal MUBUF encoding bits', encode_mubuf(MUBUF_STORE_FORMAT_D16_X, 5, 0, 16, 0x80, idxen=True, offset=4, w0_rsvd=True).hex()),
        ('illegal MUBUF encoding bits', encode_mubuf(MUBUF_STORE_FORMAT_D16_X, 5, 0, 16, 0x80, idxen=True, offset=4, w1_rsvd=True).hex()),
    ):
        refused(mod_name, lambda hw=hex_word: translate('F', code=patch(WORDS_F, store_f, hw)).hlsl_cs())

    # Lane mask in soffset: patching Case E store soffset from 0 (0x80) to s0 (0x00, written by s_and_saveexec_b32 s0, vcc_lo).
    refused('is a lane mask',
            lambda: translate('E', code=patch(WORDS_E, '003010e202050480', '003010e202050400')).hlsl_cs())

    # Direct Stage handler checks for non-compute stages, operand arity, lds/duplicate/out-of-range modifiers,
    # bad vdata/vaddr/soffset/descriptor forms.
    for stage_kind in ('ps', 'vs'):
        st = G.Stage.__new__(G.Stage)
        st.kind = stage_kind
        try:
            st.op_buffer_store_format_d16_x(['v5', 'v0', 's[16:19]', '0'], ['idxen'])
        except G.Unsupported as reason:
            assert str(reason) == 'buffer_store_format_d16_x', str(reason)
        else:
            raise AssertionError(f'{stage_kind} stage accepted buffer_store_format_d16_x')

    cs = pipe_f.cs
    cs.raw_code = b''  # unit probe calls with synthetic operand lists
    for bad_ops, bad_mods, frag in (
        ([], [], 'with 0 operands, expected 4'),
        (['v5'], [], 'with 1 operands, expected 4'),
        (['v5', 'v0', 's[16:19]'], ['idxen'], 'with 3 operands, expected 4'),
        (['v5', 'v0', 's[16:19]', '0', '0'], ['idxen'], 'with 5 operands, expected 4'),
        (['v5', 'v0', 's[16:19]', '0'], ['lds'], 'modifier lds'),
        (['v5', 'v0', 's[16:19]', '0'], ['idxen', 'idxen'], 'modifier idxen'),
        (['v5', 'v0', 's[16:19]', '0'], ['offen', 'offen'], 'modifier offen'),
        (['v5', 'v0', 's[16:19]', '0'], ['offset:4', 'offset:8'], 'modifier offset:8'),
        (['v5', 'v0', 's[16:19]', '0'], ['offset:4096'], 'modifier offset:4096'),
        (['v[5:6]', 'v0', 's[16:19]', '0'], ['idxen'], 'data v[5:6]'),
        (['s4', 'v0', 's[16:19]', '0'], ['idxen'], 'data s4'),
        (['v256', 'v0', 's[16:19]', '0'], ['idxen'], 'data v256'),
        (['v5', 'v0', 's[16:19]', '0'], ['idxen', 'offen'], 'vaddr v0 for idxen offen'),
        (['v5', 'v[0:1]', 's[16:19]', '0'], ['idxen'], 'vaddr v[0:1] for idxen'),
        (['v5', 'off', 's[16:19]', '0'], ['offen'], 'vaddr off for offen'),
        (['v5', 'v0', 's[16:19]', '0'], [], 'vaddr v0 without idxen or offen'),
        (['v5', 'v0', 's[16:19]', '-1'], ['idxen'], 'soffset -1'),
        (['v5', 'v0', 's[16:19]', 'v1'], ['idxen'], 'soffset v1'),
        (['v5', 'v0', 's[16:19]', 'exec_lo'], ['idxen'], 'is a lane mask'),
        (['v5', 'v0', 's[16:19]', 's106'], ['idxen'], 'soffset s106'),
        (['v5', 'v0', 's[16:17]', '0'], ['idxen'], 'descriptor s[16:17]'),
        (['v5', 'v0', 's[104:107]', '0'], ['idxen'], 'descriptor s[104:107]'),
    ):
        refused(frag, lambda o=bad_ops, m=bad_mods: cs.op_buffer_store_format_d16_x(o, m))

    # 6. Descriptor validation, mixed buffer binding and missing local reload.
    for inserted, frag in (
        ('800392be', 'modified descriptor'),                 # s_mov_b32 s18, 0
        ('800393be', 'modified descriptor'),                 # s_mov_b32 s19, 0
        ('11ff1187ffff00c0', 'modified descriptor'),         # s_and_b32 s17, s17, 0xc000ffff (stride clear)
        ('10841080', 'modified descriptor'),                 # s_add_u32 s16, s16, 4
        ('010082bf000080bf', 'local table load'),                    # s_branch +0 without local reload
    ):
        refused(frag, lambda ins=inserted: translate('F', code=patch(WORDS_F, store_f, ins + store_f)).hlsl_cs())

    store_on_read = encode_mubuf(MUBUF_STORE_FORMAT_D16_X, 5, 0, 12, 0x80, idxen=True, offset=4).hex()
    alias = translate('F', code=patch(WORDS_F, store_f, store_on_read))
    hlsl = alias.hlsl_cs()
    assert 'u0.Load(' in hlsl and 'store_format_d16_x(u0,' in hlsl, hlsl
    assert 'uav 0 table 1 0' in alias.manifest() and 'srv ' not in alias.manifest(), alias.manifest()

    # Negative self-check: disabling the handler refuses Case F cleanly.
    saved = G.Stage.op_buffer_store_format_d16_x
    try:
        delattr(G.Stage, 'op_buffer_store_format_d16_x')
        refused('buffer_store_format_d16_x', lambda: translate('F').hlsl_cs())
    finally:
        G.Stage.op_buffer_store_format_d16_x = saved

    # 7. Debug-layer status handling (with and without Windows Graphics Tools).
    assert _check_debug_layer('backend=hardware debug_layer=enabled\ndebug_errors=0 debug_warnings=0\n', 'unit-ok') == 'debug_errors=0 debug_warnings=0'
    assert _check_debug_layer('backend=hardware debug_layer=unavailable\n', 'unit-unavail') == 'debug_layer=unavailable'
    for bad_out in (
        'backend=hardware debug_layer=enabled\ndebug_errors=1 debug_warnings=0\n',
        'backend=hardware debug_layer=enabled\n',
        'backend=hardware\ndebug_errors=0 debug_warnings=0\n',
    ):
        try:
            _check_debug_layer(bad_out, 'unit-fail')
        except AssertionError:
            pass
        else:
            raise AssertionError('unexpectedly accepted debug output: %r' % (bad_out,))

    print('buffer_store_format_d16_x unit, format, bounds, CAS, provenance and refusal checks pass')


def assemble(llvm_mc, source):
    from startup_flat_param_check import assemble as run
    return run(llvm_mc, source)


def check_binary(llvm_mc):
    for name, (source, words, *_rest) in PROGRAMS.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for line, hex_words in VERIFIED_SNIPPETS.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == hex_words, '%s: assembled %s, recorded %s' % (line, built, hex_words)
    print('buffer_store_format_d16_x fixture words match llvm-mc (%d programs, %d snippets)'
          % (len(PROGRAMS), len(VERIFIED_SNIPPETS)))


CONTROLS = {
    'f32-drop-store': (
        'F-f32',
        '    store_format_d16_x(u0, ud2, ud3, ud4, v0, 0u, 0u, 4u, v5);\n',
        '',
        None,
    ),
    'f32-wrong-half': (
        'F-f32',
        '    uint h = val & 0xffffu;\n',
        '    uint h = (val >> 16u) & 0xffffu;\n',
        None,
    ),
    'f32-drop-oob': (
        'F-f32',
        '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;\n',
        '        if (elem_bytes > stride || rec_off > stride - elem_bytes) return;\n',
        None,
    ),
    'unorm16-trunc': (
        'F-unorm16',
        '    return (uint)floor(scaled + 0.5f);\n',
        '    return (uint)floor(scaled);\n',
        None,
    ),
    'p-drop-soffset': (
        'P',
        '    store_format_d16_x(u0, ud2, ud3, ud4, 0u, v1, ud0, 2u, v5);\n',
        '    store_format_d16_x(u0, ud2, ud3, ud4, 0u, v1, 0u, 2u, v5);\n',
        None,
    ),
    'p-drop-keep': (
        'P',
        '        uint next = (cur & keep) | ins;\n',
        '        uint next = ins;\n',
        None,
    ),
    's-fold-soff-stride': (
        'S',
        '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;\n',
        '        if (idx >= num_records || elem_bytes > stride || (rec_off + soff) > stride - elem_bytes) return;\n',
        None,
    ),
    'e-drop-exec': (
        'E',
        'if (exec & 1) store_format_d16_x(',
        'store_format_d16_x(',
        None,
    ),
    'b-short-binding': (
        'B',
        None,
        None,
        1,
    ),
    'b-tight-host-guard': (
        'B',
        '    if (uav_bytes < 4u || base > uav_bytes - 4u) return;\n',
        '    if (uav_bytes < 4u || base > uav_bytes - 8u) return;\n',
        None,
    ),
}


def mutate(hlsl, control):
    _, old, new, _ = CONTROLS[control]
    if old is None:
        return hlsl
    assert hlsl.count(old) == 1, (control, hlsl)
    return hlsl.replace(old, new)


def run_case(name, dxc, runner, adapter, directory, control=None, uav_words=None):
    prog_key, source, constants, exp, case_uav_words = case_inputs(name)
    hlsl = translate(prog_key).hlsl_cs()
    if control:
        hlsl = mutate(hlsl, control)
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    groups = PROGRAMS[prog_key][4]
    case.write_bytes(
        struct.pack('<I', groups[0] | (groups[1] << 8) | (groups[2] << 16))
        + struct.pack('<%dI' % WORDS, *exp)
        + struct.pack('<16I', *constants)
        + source
    )
    bound_uav_words = case_uav_words if uav_words is None else uav_words
    cmd = [runner, adapter, '--compute-store-sized-case', str(cso), str(case)]
    if bound_uav_words is not None:
        cmd.append(str(bound_uav_words))
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_debug_layer(stdout, label):
    if 'debug_layer=enabled' in stdout:
        assert 'debug_errors=0 debug_warnings=0' in stdout, (label, stdout)
        return 'debug_errors=0 debug_warnings=0'
    if 'debug_layer=unavailable' in stdout:
        return 'debug_layer=unavailable'
    raise AssertionError((label, 'missing debug_layer status', stdout))


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        runs = [(CONTROLS[control][0], control, CONTROLS[control][3])] if control else [
            (name, None, None) for name in DISPATCH_CASES
        ]
        for name, mutation, uav_words in runs:
            run = run_case(name, dxc, runner, adapter, Path(tmp), mutation, uav_words=uav_words)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = [line for line in run.stderr.splitlines()
                           if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
            if mutation:
                if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                    raise AssertionError('control %s: not rejected (exit %d) %s' % (mutation, run.returncode, run.stderr))
                suffix = ''
                if mutation in ('b-short-binding', 'b-tight-host-guard'):
                    assert diagnostics[0].startswith('word=1 '), (mutation, diagnostics)
                    suffix = ' (%s)' % _check_debug_layer(run.stdout, mutation)
                print('control %s: rejected at %s%s' % (mutation, diagnostics[0], suffix))
            elif run.returncode or diagnostics:
                raise AssertionError('case %s: %s' % (name, run.stderr or 'exit %d' % run.returncode))
            else:
                debug_note = _check_debug_layer(run.stdout, name)
                print('case %s: all %d words match (%s)' % (name, WORDS, debug_note))

        if not control:
            # Case C: same-halfword concurrent contention (256 threads x 4 writes to both halves of dword 0).
            # Every contending lane writes a valid candidate; CAS guarantees no bitwise-OR (0x3333/0xcccc) or torn 0x0000.
            run = run_case('C', dxc, runner, adapter, Path(tmp))
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = [line for line in run.stderr.splitlines()
                           if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
            debug_note = _check_debug_layer(run.stdout, 'C')
            assert 'mismatches=1' in run.stdout, (run.stdout, diagnostics)
            m = re.fullmatch(r'word=0 expected=deadbeef actual=([0-9a-f]{8})', diagnostics[0] if diagnostics else '')
            assert m, (diagnostics, run.stdout)
            actual = int(m.group(1), 16)
            lo, hi = actual & 0xffff, (actual >> 16) & 0xffff
            assert lo in CONTENTION_VALID_LO and hi in CONTENTION_VALID_HI, (
                'case C: atomic CAS contention corrupted word 0: 0x%08x (lo=0x%04x hi=0x%04x)' % (actual, lo, hi))
            print('case C: atomic CAS contention ok (word0=0x%08x, 319 sentinels untouched, %s)' % (actual, debug_note))
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
