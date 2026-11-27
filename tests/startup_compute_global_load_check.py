"""Compute 32-bit GLOBAL saddr load (`global_load_dword`) in tools/startup/gfx2hlsl.py (Refs #2618, #2022).

  (no arguments) or controls                    translation, encoding and refusal controls, pure Python
  binary <llvm-mc>                              reassemble every fixture source and compare machine words
  decode <libLLVM>                              disassemble GLOBAL test words with LLVM 15 and compare strings
  dispatch <dxc> <offscreen_check> <adapter>    translate, compile DXC cs_6_0, dispatch and verify GPU readback
  image-dispatch <dxc> <offscreen_check> <adapter>  run the program-image readbacks and mutations

All instruction bytes below are constructed from the RDNA 2 / tools/disasm_rdna2.py bit layouts (no game bytes).
"""
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))

import disasm_rdna2 as D  # noqa: E402
from startup_gfx2hlsl_compute_check import G, MASK, SENTINEL, TGID_X, WORDS, assemble, refused  # noqa: E402

GLOBAL_LOAD_UBYTE = 0x08
GLOBAL_LOAD_SBYTE = 0x09
GLOBAL_LOAD_USHORT = 0x0a
GLOBAL_LOAD_SSHORT = 0x0b
GLOBAL_LOAD_DWORD = 0x0c
GLOBAL_LOAD_DWORDX2 = 0x0d
GLOBAL_LOAD_DWORDX4 = 0x0e
GLOBAL_LOAD_DWORDX3 = 0x0f
GLOBAL_STORE_DWORD = 0x1c
GLOBAL_ATOMIC_ADD = 0x32

# s0-s3 unmodelled, s4-s5 = two root constants (ud0, ud1), s[6:7] = root SRV (param 1), s8 = table (param 2);
# s9 = gid.x (TGID_X), v0 = tid.x.
USER9 = 9 << 1
RSRC2_A = USER9 | TGID_X
GROUP_A = (8, 1, 1)
GROUPS_A = (2, 1, 1)
INACTIVE_TAG = 0x77000000
PRELOAD_TAG = 0x66000000


def root_blob_cs(*, extra_root_srv=False, flags=0):
    """DXBC RTS0 container:
    param 0 = 2 root constants (s4-s5),
    param 1 = root SRV descriptor (s[6:7]),
    optional param 2 = second root SRV descriptor (s[8:9]),
    final param = descriptor table with 1 UAV slot.
    """
    nparams = 4 if extra_root_srv else 3
    poff = 24
    coff = poff + 12 * nparams
    # Root descriptor payload in RTS0: (ShaderRegister, RegisterSpace) = 8 bytes.
    srv1_off = coff + 12
    srv2_off = srv1_off + 8
    toff = (srv2_off + 8) if extra_root_srv else (srv1_off + 8)
    params = [
        struct.pack('<3I', 1, 0, coff),       # 32BIT_CONSTANTS
        struct.pack('<3I', 3, 0, srv1_off),   # SRV root descriptor
    ]
    if extra_root_srv:
        params.append(struct.pack('<3I', 3, 0, srv2_off))
    params.append(struct.pack('<3I', 0, 0, toff))  # DESCRIPTOR_TABLE
    payload = struct.pack('<3I', 0, 0, 2) + struct.pack('<2I', 0, 0)
    if extra_root_srv:
        payload += struct.pack('<2I', 1, 0)
    payload += struct.pack('<2I', 0, toff + 8)
    rts0 = struct.pack('<6I', 2, nparams, poff, 0, toff + 8, flags) + b''.join(params) + payload
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 36 + len(part), 1) + struct.pack('<I', 36) + part


def encode_sop1(op, sdst, ssrc0, *, lit=None):
    w0 = 0xbe800000 | ((sdst & 0x7f) << 16) | ((op & 0xff) << 8) | (ssrc0 & 0xff)
    return struct.pack('<II', w0, lit & MASK) if lit is not None else struct.pack('<I', w0)


def encode_sop2(op, sdst, ssrc0, ssrc1, *, lit=None):
    w0 = 0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff)
    return struct.pack('<II', w0, lit & MASK) if lit is not None else struct.pack('<I', w0)


def encode_sopp(op, simm16=0):
    return struct.pack('<I', 0xbf800000 | ((op & 0x7f) << 16) | (simm16 & 0xffff))


def encode_smem(op, sdata, sbase, *, soffset=0x7d, imm=0):
    w0 = (0b111101 << 26) | ((op & 0xff) << 18) | ((sdata & 0x7f) << 6) | ((sbase >> 1) & 0x3f)
    w1 = ((soffset & 0x7f) << 25) | (imm & 0x1fffff)
    return struct.pack('<II', w0, w1)


def encode_vop1(op, vdst, src0, *, lit=None):
    w0 = 0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<II', w0, lit & MASK) if lit is not None else struct.pack('<I', w0)


def encode_vop2(op, vdst, src0, vsrc1, *, lit=None):
    w0 = ((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<II', w0, lit & MASK) if lit is not None else struct.pack('<I', w0)


def encode_vopc(op, src0, vsrc1):
    w0 = 0x7c000000 | ((op & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0)


def encode_vop3a(op, vdst, src0, src1, src2=0):
    w0 = (0b110101 << 26) | ((op & 0x3ff) << 16) | (vdst & 0xff)
    w1 = ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff)
    return struct.pack('<II', w0, w1)


def encode_mubuf(op, vdata, vaddr, srsrc, soffset_enc=0x80, *, offen=True, offset=0):
    w0 = 0xe0000000 | ((op & 0x7f) << 18) | (int(offen) << 12) | (offset & 0xfff)
    w1 = ((soffset_enc & 0xff) << 24) | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return struct.pack('<II', w0, w1)


def encode_global(op, vdst, addr, saddr, *, seg=2, data=0, offset=0, glc=0, slc=0, dlc=0, lds=0, nv=0, rsvd25=0):
    """Construct an 8-byte RDNA 2 FLAT / SCRATCH / GLOBAL instruction from its hardware fields."""
    w0 = (0xdc000000 | ((rsvd25 & 1) << 25) | ((op & 0x7f) << 18) | ((slc & 1) << 17)
          | ((glc & 1) << 16) | ((seg & 0x3) << 14) | ((lds & 1) << 13) | ((dlc & 1) << 12) | (offset & 0xfff))
    w1 = (((vdst & 0xff) << 24) | ((nv & 1) << 23) | ((saddr & 0x7f) << 16)
          | ((data & 0xff) << 8) | (addr & 0xff))
    return struct.pack('<II', w0, w1)


# Kernel A: 2 groups x 8 threads (16 lanes). Varies per-lane vaddr via runtime root constants (ud0=bias, ud1=stride)
# and exercises four global_load_dword forms across distinct scalar base states:
#   1. v5: direct entry root pointer s[6:7] with vdst == vaddr register aliasing (base + 0);
#   2. v6: out-of-place literal-advanced scalar base s[20:21] = s[6:7] + 48 (base + 48, +12 words);
#   3. v7: in-place second advance s[20:21] += 32 (base + 80) with negative signed immediate offset:-12 and
#          cache flags glc/slc/dlc (base + 68, +17 words);
#   4. v8: s_mov_b64 copy s[22:23] = s[6:7] with positive immediate offset:8 under partial EXEC (tid.x < 5 ->
#          base + 8, +2 words; inactive lanes tid.x in 5..7 keep INACTIVE_TAG | lane).
KERNEL_A_SOURCE = """\
s_mov_b32 s10, s8
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s12, s9, 3
v_add_nc_u32_e32 v3, s12, v0
v_mul_lo_u32 v4, v3, s5
v_add_nc_u32_e32 v4, s4, v4
v_and_b32_e32 v4, 63, v4
v_lshlrev_b32_e32 v4, 2, v4
v_mov_b32_e32 v5, v4
global_load_dword v5, v5, s[6:7]
s_add_u32 s20, s6, 48
s_addc_u32 s21, s7, 0
global_load_dword v6, v4, s[20:21]
s_add_u32 s20, s20, 32
s_addc_u32 s21, s21, 0
global_load_dword v7, v4, s[20:21] offset:-12 glc slc dlc
s_mov_b64 s[22:23], s[6:7]
v_mov_b32_e32 v8, 0x77000000
v_or_b32_e32 v8, v8, v3
v_mov_b32_e32 v9, v4
v_cmp_gt_u32_e32 vcc_lo, 5, v0
s_and_saveexec_b32 s24, vcc_lo
global_load_dword v8, v9, s[22:23] offset:8
s_mov_b32 exec_lo, s24
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v10, 4, v3
buffer_store_dwordx4 v[5:8], v10, s[16:19], 0 offen
s_endpgm
"""

KERNEL_A_BYTES = (
    encode_sop1(0x03, 10, 8)
    + encode_sop1(0x03, 11, 0x80)
    + encode_smem(0x02, 16, 10, soffset=0x7d, imm=0)
    + encode_sopp(0x0c, 0xc07f)
    + encode_sop2(0x1e, 12, 9, 0x80 + 3)                      # s_lshl_b32 s12, s9, 3
    + encode_vop2(0x25, 3, 12, 0)                             # v_add_nc_u32_e32 v3, s12, v0
    + encode_vop3a(0x169, 4, 0x100 + 3, 5)                    # v_mul_lo_u32 v4, v3, s5
    + encode_vop2(0x25, 4, 4, 4)                              # v_add_nc_u32_e32 v4, s4, v4
    + encode_vop2(0x1b, 4, 0x80 + 63, 4)                      # v_and_b32_e32 v4, 63, v4
    + encode_vop2(0x1a, 4, 0x80 + 2, 4)                       # v_lshlrev_b32_e32 v4, 2, v4
    + encode_vop1(0x01, 5, 0x100 + 4)                         # v_mov_b32_e32 v5, v4
    + encode_global(GLOBAL_LOAD_DWORD, 5, 5, 6)               # global_load_dword v5, v5, s[6:7]
    + encode_sop2(0x00, 20, 6, 0x80 + 48)                     # s_add_u32 s20, s6, 48
    + encode_sop2(0x04, 21, 7, 0x80)                          # s_addc_u32 s21, s7, 0
    + encode_global(GLOBAL_LOAD_DWORD, 6, 4, 20)              # global_load_dword v6, v4, s[20:21]
    + encode_sop2(0x00, 20, 20, 0x80 + 32)                    # s_add_u32 s20, s20, 32
    + encode_sop2(0x04, 21, 21, 0x80)                         # s_addc_u32 s21, s21, 0
    + encode_global(GLOBAL_LOAD_DWORD, 7, 4, 20, offset=-12, glc=1, slc=1, dlc=1)
    + encode_sop1(0x04, 22, 6)                                # s_mov_b64 s[22:23], s[6:7]
    + encode_vop1(0x01, 8, 0xff, lit=INACTIVE_TAG)            # v_mov_b32_e32 v8, 0x77000000
    + encode_vop2(0x1c, 8, 0x100 + 8, 3)                      # v_or_b32_e32 v8, v8, v3
    + encode_vop1(0x01, 9, 0x100 + 4)                         # v_mov_b32_e32 v9, v4
    + encode_vopc(0xc4, 0x80 + 5, 0)                          # v_cmp_gt_u32_e32 vcc_lo, 5, v0
    + encode_sop1(0x3c, 24, 0x6a)                             # s_and_saveexec_b32 s24, vcc_lo
    + encode_global(GLOBAL_LOAD_DWORD, 8, 9, 22, offset=8)    # global_load_dword v8, v9, s[22:23] offset:8
    + encode_sop1(0x03, 0x7e, 24)                             # s_mov_b32 exec_lo, s24
    + encode_sopp(0x0c, 0x3f70)                               # s_waitcnt vmcnt(0)
    + encode_vop2(0x1a, 10, 0x80 + 4, 3)                      # v_lshlrev_b32_e32 v10, 4, v3
    + encode_mubuf(0x1e, 5, 10, 16)                           # buffer_store_dwordx4 v[5:8], v10, s[16:19], 0 offen
    + encode_sopp(0x01, 0)                                    # s_endpgm
)

# Kernel B: unadvanced entry root pointer s[6:7] (base_off = 0) with negative 12-bit signed offset:-16
# (`v5 = (a >= 16u ? r0.Load(a - 16u) : 0u);`), verifying guarded subtraction when base_off + offset < 0.
KERNEL_B_SOURCE = """\
s_mov_b32 s10, s8
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s12, s9, 3
v_add_nc_u32_e32 v3, s12, v0
v_mul_lo_u32 v4, v3, s5
v_add_nc_u32_e32 v4, s4, v4
v_and_b32_e32 v4, 63, v4
v_add_nc_u32_e32 v4, 4, v4
v_lshlrev_b32_e32 v4, 2, v4
global_load_dword v5, v4, s[6:7] offset:-16
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v6, 2, v3
buffer_store_dword v5, v6, s[16:19], 0 offen
s_endpgm
"""

KERNEL_B_BYTES = (
    encode_sop1(0x03, 10, 8)
    + encode_sop1(0x03, 11, 0x80)
    + encode_smem(0x02, 16, 10, soffset=0x7d, imm=0)
    + encode_sopp(0x0c, 0xc07f)
    + encode_sop2(0x1e, 12, 9, 0x80 + 3)
    + encode_vop2(0x25, 3, 12, 0)
    + encode_vop3a(0x169, 4, 0x100 + 3, 5)
    + encode_vop2(0x25, 4, 4, 4)
    + encode_vop2(0x1b, 4, 0x80 + 63, 4)
    + encode_vop2(0x25, 4, 0x80 + 4, 4)
    + encode_vop2(0x1a, 4, 0x80 + 2, 4)
    + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=-16)
    + encode_sopp(0x0c, 0x3f70)
    + encode_vop2(0x1a, 6, 0x80 + 2, 3)
    + encode_mubuf(0x1c, 5, 6, 16)
    + encode_sopp(0x01, 0)
)

# Kernel C: positive-offset 64-bit sum overflow (`v4 = (ud0 & -4) + 4 * lane` across 0xfffffff4..0x100000004 and into
# the low range), negative-offset underflow (`v9 = (ud1 & -4) + 4 * lane` across 0..12 below 16), and proven-aligned
# low-bit byte offsets (`v11 = (ud1 + lane) & -4` stepping 1 raw byte per lane across distinct neighboring DWORDs)
# against a bounded root view populated with nonzero sentinels across all 320 low-range words:
#   1. v5: s[6:7] + offset:8 (total_off = +8; v4 = 0xfffffffc -> guest base + 0x100000004 -> 0, not wrapped byte 4);
#   2. v6: (s[6:7] + 48) + offset:-12 (total_off = +36; v4 = 0xfffffffc -> guest base + 0x100000020 -> 0, not byte 32);
#   3. v7: s[6:7] + offset:-16 (total_off = -16; v9 in 0..12 -> guest base - 16..-4 -> 0, v9 >= 16 -> in-range);
#   4. v8: (s[6:7] + 8) + offset:-12 (total_off = -4; raw `ud1 + lane` carries low bits 0..3 across neighboring DWORDs).
KERNEL_C_SOURCE = """\
s_mov_b32 s10, s8
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s12, s9, 3
v_add_nc_u32_e32 v3, s12, v0
v_lshlrev_b32_e32 v4, 2, v3
s_and_b32 s14, s4, -4
s_and_b32 s15, s5, -4
v_add_nc_u32_e32 v9, s15, v4
v_add_nc_u32_e32 v4, s14, v4
v_add_nc_u32_e32 v11, s5, v3
v_and_b32_e32 v11, -4, v11
v_mov_b32_e32 v5, 0x66000000
v_or_b32_e32 v5, v5, v3
v_mov_b32_e32 v6, v5
v_mov_b32_e32 v7, v5
v_mov_b32_e32 v8, v5
global_load_dword v5, v4, s[6:7] offset:8
s_add_u32 s20, s6, 48
s_addc_u32 s21, s7, 0
global_load_dword v6, v4, s[20:21] offset:-12
global_load_dword v7, v9, s[6:7] offset:-16
s_add_u32 s22, s6, 8
s_addc_u32 s23, s7, 0
global_load_dword v8, v11, s[22:23] offset:-12
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v10, 4, v3
buffer_store_dwordx4 v[5:8], v10, s[16:19], 0 offen
s_endpgm
"""

KERNEL_C_BYTES = (
    encode_sop1(0x03, 10, 8)
    + encode_sop1(0x03, 11, 0x80)
    + encode_smem(0x02, 16, 10, soffset=0x7d, imm=0)
    + encode_sopp(0x0c, 0xc07f)
    + encode_sop2(0x1e, 12, 9, 0x80 + 3)
    + encode_vop2(0x25, 3, 12, 0)
    + encode_vop2(0x1a, 4, 0x80 + 2, 3)
    + encode_sop2(0x0e, 14, 4, 192 + 4)
    + encode_sop2(0x0e, 15, 5, 192 + 4)
    + encode_vop2(0x25, 9, 15, 4)
    + encode_vop2(0x25, 4, 14, 4)
    + encode_vop2(0x25, 11, 5, 3)
    + encode_vop2(0x1b, 11, 192 + 4, 11)
    + encode_vop1(0x01, 5, 0xff, lit=PRELOAD_TAG)
    + encode_vop2(0x1c, 5, 0x100 + 5, 3)
    + encode_vop1(0x01, 6, 0x100 + 5)
    + encode_vop1(0x01, 7, 0x100 + 5)
    + encode_vop1(0x01, 8, 0x100 + 5)
    + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=8)
    + encode_sop2(0x00, 20, 6, 0x80 + 48)
    + encode_sop2(0x04, 21, 7, 0x80)
    + encode_global(GLOBAL_LOAD_DWORD, 6, 4, 20, offset=-12)
    + encode_global(GLOBAL_LOAD_DWORD, 7, 9, 6, offset=-16)
    + encode_sop2(0x00, 22, 6, 0x80 + 8)
    + encode_sop2(0x04, 23, 7, 0x80)
    + encode_global(GLOBAL_LOAD_DWORD, 8, 11, 22, offset=-12)
    + encode_sopp(0x0c, 0x3f70)
    + encode_vop2(0x1a, 10, 0x80 + 4, 3)
    + encode_mubuf(0x1e, 5, 10, 16)
    + encode_sopp(0x01, 0)
)

DECODE_CASES = (
    (encode_global(GLOBAL_LOAD_DWORD, 5, 5, 6), 'global_load_dword v5, v5, s[6:7]'),
    (encode_global(GLOBAL_LOAD_DWORD, 6, 4, 20), 'global_load_dword v6, v4, s[20:21]'),
    (encode_global(GLOBAL_LOAD_DWORD, 7, 4, 20, offset=-12, glc=1, slc=1, dlc=1),
     'global_load_dword v7, v4, s[20:21] offset:-12 glc slc dlc'),
    (encode_global(GLOBAL_LOAD_DWORD, 8, 9, 22, offset=8), 'global_load_dword v8, v9, s[22:23] offset:8'),
    (encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=-2048), 'global_load_dword v5, v4, s[6:7] offset:-2048'),
    (encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=2047), 'global_load_dword v5, v4, s[6:7] offset:2047'),
    (encode_global(GLOBAL_LOAD_DWORD, 5, 4, 0x7d), 'global_load_dword v5, v[4:5], off'),
    (encode_global(GLOBAL_LOAD_DWORD, 0, 4, 6, lds=1), 'global_load_dword v4, s[6:7] lds'),
)


def make_source_words(seed):
    """320 pairwise-distinct nonzero words so any low-range wrap or off-by-one shift mismatches immediately."""
    out = [((seed + (i + 1) * 0x9e3779b1) ^ ((i * 0x01010101) & MASK)) & MASK for i in range(WORDS)]
    assert len(set(out)) == WORDS and 0 not in out and SENTINEL not in out
    return out


def bounded_root_load(src_words, vaddr, total_off):
    """Exact 64-bit guest offset relative to the bound 320-dword root view (`0` when out of `[0, 1280)`)."""
    rel = (vaddr & MASK) + total_off
    if rel < 0 or rel >= WORDS * 4:
        return 0
    assert rel % 4 == 0
    return src_words[rel // 4]


def kernel_a_reference(src_words, ud0, ud1):
    want = [SENTINEL] * WORDS
    for grp in range(GROUPS_A[0]):
        for tid in range(GROUP_A[0]):
            lane = grp * GROUP_A[0] + tid
            idx = (lane * ud1 + ud0) & 63
            w0 = src_words[idx]
            w1 = src_words[idx + 12]       # +48 bytes
            w2 = src_words[idx + 17]       # +80 - 12 = +68 bytes
            w3 = src_words[idx + 2] if tid < 5 else (INACTIVE_TAG | lane)
            want[4 * lane:4 * lane + 4] = [w0, w1, w2, w3]
    return want


def kernel_b_reference(src_words, ud0, ud1):
    want = [SENTINEL] * WORDS
    for grp in range(GROUPS_A[0]):
        for tid in range(GROUP_A[0]):
            lane = grp * GROUP_A[0] + tid
            idx = ((lane * ud1 + ud0) & 63) + 4
            want[lane] = src_words[idx - 4]  # -16 bytes = -4 dwords
    return want


def kernel_c_reference(src_words, ud0, ud1):
    want = [SENTINEL] * WORDS
    for grp in range(GROUPS_A[0]):
        for tid in range(GROUP_A[0]):
            lane = grp * GROUP_A[0] + tid
            v4 = ((ud0 & 0xfffffffc) + 4 * lane) & MASK
            v9 = ((ud1 & 0xfffffffc) + 4 * lane) & MASK
            v11 = ((ud1 + lane) & 0xfffffffc) & MASK
            want[4 * lane:4 * lane + 4] = [
                bounded_root_load(src_words, v4, 8),
                bounded_root_load(src_words, v4, 36),
                bounded_root_load(src_words, v9, -16),
                bounded_root_load(src_words, v11, -4),
            ]
    return want


def cs_pipe(code=KERNEL_A_BYTES, *, root=None, rsrc2=RSRC2_A, group=GROUP_A):
    return G.ComputePipeline(code, root or root_blob_cs(), rsrc2, group)


def disassemble_words(code):
    lines = []
    off = 0
    while off < len(code):
        size, _name, asm, _comment = D.decode_instruction(code, off)
        lines.append(asm)
        off += size
    return '\n'.join(lines) + '\n'


def check_controls():
    # Low-level translator probes use the immutable empty alignment class fallback.
    from startup_gfx2hlsl_sop2_complement_check import Probe

    probe = Probe()
    probe.exec_known = True
    probe.one('s_wqm_b32', 's_wqm_b32 exec_lo, exec_lo')
    assert probe.wqm_exec

    for raw, expected_asm in DECODE_CASES:
        size, _name, asm, _ = D.decode_instruction(raw, 0)
        assert size == 8 and asm == expected_asm, (asm, expected_asm)

    assert disassemble_words(KERNEL_A_BYTES) == KERNEL_A_SOURCE, disassemble_words(KERNEL_A_BYTES)
    assert disassemble_words(KERNEL_B_BYTES) == KERNEL_B_SOURCE, disassemble_words(KERNEL_B_BYTES)
    assert disassemble_words(KERNEL_C_BYTES) == KERNEL_C_SOURCE, disassemble_words(KERNEL_C_BYTES)

    pipe_a = cs_pipe(KERNEL_A_BYTES)
    hlsl_a = pipe_a.hlsl_cs()
    manifest_a = pipe_a.manifest()
    expected_manifest_a = (
        'x2w-manifest 2\n'
        'stage cs\n'
        'group 8 1 1\n'
        'ud 2\n'
        'udsrc 0 rootconst 0 0\n'
        'udsrc 1 rootconst 0 1\n'
        'srv 0 buf root 1\n'
        'uav 0 table 2 0\n'
    )
    assert manifest_a == expected_manifest_a, manifest_a
    for snippet in (
        'ByteAddressBuffer r0 : register(t0);',
        'RWByteAddressBuffer u0 : register(u0);',
        'a = v5;\n    v5 = r0.Load(a);',
        'a = v4;\n    v6 = (a <= 0xffffffcfu ? r0.Load(a + 48u) : 0u);',
        'a = v4;\n    v7 = (a <= 0xffffffbbu ? r0.Load(a + 68u) : 0u);',
        'a = v9;\n    if (exec & 1) v8 = (a <= 0xfffffff7u ? r0.Load(a + 8u) : 0u);',
    ):
        assert snippet in hlsl_a, (snippet, hlsl_a)

    pipe_b = cs_pipe(KERNEL_B_BYTES)
    hlsl_b = pipe_b.hlsl_cs()
    assert 'a = v4;\n    v5 = (a >= 16u ? r0.Load(a - 16u) : 0u);' in hlsl_b, hlsl_b

    pipe_c = cs_pipe(KERNEL_C_BYTES)
    hlsl_c = pipe_c.hlsl_cs()
    for snippet in (
        'v11 = 0xfffffffcu & v11;',
        'a = v4;\n    v5 = (a <= 0xfffffff7u ? r0.Load(a + 8u) : 0u);',
        'a = v4;\n    v6 = (a <= 0xffffffdbu ? r0.Load(a + 36u) : 0u);',
        'a = v9;\n    v7 = (a >= 16u ? r0.Load(a - 16u) : 0u);',
        'a = v11;\n    v8 = (a >= 4u ? r0.Load(a - 4u) : 0u);',
    ):
        assert snippet in hlsl_c, (snippet, hlsl_c)

    # Multi-root SRV check: s[6:7] = root 1 -> r0, s[8:9] = root 2 -> r1 (table in s10, gid.x in s11).
    multi_code = (
        encode_sop1(0x03, 12, 10)
        + encode_sop1(0x03, 13, 0x80)
        + encode_smem(0x02, 16, 12, soffset=0x7d, imm=0)
        + encode_sopp(0x0c, 0xc07f)
        + encode_vop2(0x1a, 4, 0x80 + 2, 0)
        + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6)
        + encode_global(GLOBAL_LOAD_DWORD, 6, 4, 8, offset=16)
        + encode_sopp(0x0c, 0x3f70)
        + encode_vop2(0x25, 5, 0x100 + 5, 6)
        + encode_mubuf(0x1c, 5, 4, 16)
        + encode_sopp(0x01, 0)
    )
    multi_pipe = G.ComputePipeline(multi_code, root_blob_cs(extra_root_srv=True), (11 << 1) | TGID_X, (8, 1, 1))
    multi_hlsl = multi_pipe.hlsl_cs()
    multi_manifest = multi_pipe.manifest()
    assert 'srv 0 buf root 1\nsrv 1 buf root 2\nuav 0 table 3 0\n' in multi_manifest, multi_manifest
    assert 'v5 = r0.Load(a);' in multi_hlsl and 'v6 = (a <= 0xffffffefu ? r1.Load(a + 16u) : 0u);' in multi_hlsl, multi_hlsl

    # Boundary 4-byte-aligned signed 12-bit offsets (+2044 and -2048) lower cleanly with exact 32-bit guards.
    edge_code = (
        encode_sop1(0x03, 10, 8)
        + encode_sop1(0x03, 11, 0x80)
        + encode_smem(0x02, 16, 10, soffset=0x7d, imm=0)
        + encode_sopp(0x0c, 0xc07f)
        + encode_vop2(0x1a, 4, 0x80 + 2, 0)
        + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=2044)
        + encode_global(GLOBAL_LOAD_DWORD, 6, 4, 6, offset=-2048)
        + encode_sopp(0x0c, 0x3f70)
        + encode_mubuf(0x1d, 5, 4, 16)
        + encode_sopp(0x01, 0)
    )
    edge_hlsl = cs_pipe(edge_code).hlsl_cs()
    assert ('v5 = (a <= 0xfffff803u ? r0.Load(a + 2044u) : 0u);' in edge_hlsl
            and 'v6 = (a >= 2048u ? r0.Load(a - 2048u) : 0u);' in edge_hlsl), edge_hlsl

    # Encoding guards: reserved bit 25, nv (w1[23]), and non-zero data field (w1[15:8]).
    prefix = (
        encode_sop1(0x03, 10, 8)
        + encode_sop1(0x03, 11, 0x80)
        + encode_smem(0x02, 16, 10, soffset=0x7d, imm=0)
        + encode_sopp(0x0c, 0xc07f)
        + encode_vop2(0x1a, 4, 0x80 + 2, 0)
    )
    suffix = encode_sopp(0x0c, 0x3f70) + encode_mubuf(0x1c, 5, 4, 16) + encode_sopp(0x01, 0)
    refused('reserved bit 25', lambda: cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, rsvd25=1) + suffix))
    refused('global_load_dword nv', lambda: cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, nv=1) + suffix))
    refused('unused data field', lambda: cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, data=1) + suffix))

    # Binary refusals: off form, lds modifier, wider/sub-dword/store/atomic/flat/scratch opcodes.
    refused('without a scalar address pair',
            lambda: cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 0x7d) + suffix))
    refused('global load modifier lds',
            lambda: cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 0, 4, 6, lds=1) + suffix))
    for bad_name, bad_op, seg in (
        ('global_load_ubyte', GLOBAL_LOAD_UBYTE, 2),
        ('global_load_sbyte', GLOBAL_LOAD_SBYTE, 2),
        ('global_load_ushort', GLOBAL_LOAD_USHORT, 2),
        ('global_load_sshort', GLOBAL_LOAD_SSHORT, 2),
        ('global_load_dwordx2', GLOBAL_LOAD_DWORDX2, 2),
        ('global_load_dwordx3', GLOBAL_LOAD_DWORDX3, 2),
        ('global_load_dwordx4', GLOBAL_LOAD_DWORDX4, 2),
        ('global_store_dword', GLOBAL_STORE_DWORD, 2),
        ('global_atomic_add', GLOBAL_ATOMIC_ADD, 2),
        ('flat_load_dword', GLOBAL_LOAD_DWORD, 0),
        ('scratch_load_dword', GLOBAL_LOAD_DWORD, 1),
    ):
        refused(bad_name, lambda op=bad_op, s=seg: cs_pipe(prefix + encode_global(op, 5, 4, 6, seg=s) + suffix))

    # Provenance & pointer-pair refusals on binary programs.
    # 1. Incomplete pointer add (s_add_u32 s20, s6, 16 without s_addc_u32).
    refused('not the matching pair', lambda: cs_pipe(
        prefix + encode_sop2(0x00, 20, 6, 0x80 + 16) + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 20) + suffix))
    # 2. Clobbered entry high word s7.
    refused('not the matching pair', lambda: cs_pipe(
        prefix + encode_sop1(0x03, 7, 0x80) + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6) + suffix))
    # 3. High word taken from a second root pointer (s9 from root 2 instead of s7 from root 1).
    refused('not the matching pair', lambda: G.ComputePipeline(
        encode_sop1(0x03, 12, 10)
        + encode_sop1(0x03, 13, 0x80)
        + encode_smem(0x02, 16, 12, soffset=0x7d, imm=0)
        + encode_sopp(0x0c, 0xc07f)
        + encode_vop2(0x1a, 4, 0x80 + 2, 0)
        + encode_sop1(0x03, 20, 6)
        + encode_sop1(0x03, 21, 9)
        + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 20)
        + suffix,
        root_blob_cs(extra_root_srv=True), (11 << 1) | TGID_X, (8, 1, 1)))
    # 4. Table pointer s[8:9] (not a 64-bit root buffer pointer).
    refused("global load through ('table', 2)", lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 8) + suffix))
    # 5. Loaded V# descriptor words s[16:17].
    refused('holding (\'desc\'', lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 16) + suffix))
    # 6. Numeric SGPR pair s[4:5] (root constants, not a pointer).
    refused('holding (\'ud\'', lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 4) + suffix))
    # 7. Special saddr encodings (vcc=106, ttmp[0:1]=108).
    refused('global load base vcc', lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 0x6a) + suffix))
    refused('global load base ttmp[0:1]', lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 0x6c) + suffix))

    # 8. Low-bit 4-byte alignment refusals on binary programs (vaddr, immediate offset, scalar advance, partial EXEC).
    for unaligned_vaddr in (1, 2, 3):
        refused('not proven 4-byte aligned', lambda c=unaligned_vaddr: cs_pipe(
            prefix + encode_vop1(0x01, 7, 0x80 + c) + encode_global(GLOBAL_LOAD_DWORD, 5, 7, 6) + suffix))
    # Unshifted tid.x (v0), shift by 1 (`v0 << 1`), mask `-2` (`0xfffffffe`), and adding unmasked root constant `s4`.
    refused('not proven 4-byte aligned', lambda: cs_pipe(
        prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 0, 6) + suffix))
    refused('not proven 4-byte aligned', lambda: cs_pipe(
        prefix + encode_vop2(0x1a, 7, 0x80 + 1, 0) + encode_global(GLOBAL_LOAD_DWORD, 5, 7, 6) + suffix))
    refused('not proven 4-byte aligned', lambda: cs_pipe(
        prefix + encode_vop2(0x1b, 7, 192 + 2, 0) + encode_global(GLOBAL_LOAD_DWORD, 5, 7, 6) + suffix))
    refused('not proven 4-byte aligned', lambda: cs_pipe(
        prefix + encode_vop2(0x25, 7, 4, 4) + encode_global(GLOBAL_LOAD_DWORD, 5, 7, 6) + suffix))
    # Dropping `v_and_b32_e32 v11, -4, v11` from Kernel C leaves `v11 = s5 + v3` unproven and refuses translation.
    mask_v11_bytes = encode_vop2(0x1b, 11, 192 + 4, 11)
    assert KERNEL_C_BYTES.count(mask_v11_bytes) == 1
    refused('vaddr v11 is not proven 4-byte aligned',
            lambda: cs_pipe(KERNEL_C_BYTES.replace(mask_v11_bytes, b'')))
    # Partial-EXEC write of an aligned value over an odd VGPR (`v9 = 1`) leaves inactive lanes odd and is refused.
    refused('vaddr v9 is not proven 4-byte aligned', lambda: cs_pipe(
        prefix
        + encode_vop1(0x01, 9, 0x80 + 1)
        + encode_vopc(0xc4, 0x80 + 5, 0)
        + encode_sop1(0x3c, 24, 0x6a)
        + encode_vop1(0x01, 9, 0x100 + 4)
        + encode_sop1(0x03, 0x7e, 24)
        + encode_global(GLOBAL_LOAD_DWORD, 5, 9, 6)
        + suffix))
    # Unaligned immediate offsets and unaligned literal scalar-base advances are refused.
    for bad_imm in (1, 2, 3, -1, -2, -3, 2047, -2047):
        refused(f'unaligned immediate offset {bad_imm}', lambda imm=bad_imm: cs_pipe(
            prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6, offset=imm) + suffix))
    for bad_adv in (1, 2, 3):
        refused(f'unaligned pointer offset {bad_adv}', lambda adv=bad_adv: cs_pipe(
            prefix
            + encode_sop2(0x00, 20, 6, 0x80 + adv)
            + encode_sop2(0x04, 21, 7, 0x80)
            + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 20)
            + suffix))

    # Direct Stage handler checks for arity, operand shapes, offset range, untracked branch add, and non-cs stages.
    stage = cs_pipe(prefix + encode_global(GLOBAL_LOAD_DWORD, 5, 4, 6) + suffix).cs
    refused('expected 3', lambda: stage.op_global_load_dword([], []))
    refused('expected 3', lambda: stage.op_global_load_dword(['v5'], []))
    refused('expected 3', lambda: stage.op_global_load_dword(['v5', 'v4'], []))
    refused('expected 3', lambda: stage.op_global_load_dword(['v5', 'v4', 's[6:7]', '0'], []))
    refused('global load modifier lds', lambda: stage.op_global_load_dword(['v4', 's[6:7]'], ['lds']))
    refused('global load modifier tfe', lambda: stage.op_global_load_dword(['v5', 'v4', 's[6:7]'], ['tfe']))
    refused('global load offset 2048', lambda: stage.op_global_load_dword(['v5', 'v4', 's[6:7]'], ['offset:2048']))
    refused('global load offset -2049', lambda: stage.op_global_load_dword(['v5', 'v4', 's[6:7]'], ['offset:-2049']))
    refused('global load destination v[5:6]', lambda: stage.op_global_load_dword(['v[5:6]', 'v4', 's[6:7]'], []))
    refused('global load destination s4', lambda: stage.op_global_load_dword(['s4', 'v4', 's[6:7]'], []))
    refused('global load vaddr v[4:5]', lambda: stage.op_global_load_dword(['v5', 'v[4:5]', 's[6:7]'], []))
    refused('global load vaddr off', lambda: stage.op_global_load_dword(['v5', 'off', 's[6:7]'], []))
    refused('global load base s6', lambda: stage.op_global_load_dword(['v5', 'v4', 's6'], []))
    refused('global load base s[5:6]', lambda: stage.op_global_load_dword(['v5', 'v4', 's[5:6]'], []))
    refused('global load base s[106:107]', lambda: stage.op_global_load_dword(['v5', 'v4', 's[106:107]'], []))

    stage.sym[30] = ('ptr', ('root', 1), 'added')
    stage.sym[31] = ('ptrhi', ('root', 1))
    refused('untracked offset', lambda: stage.op_global_load_dword(['v5', 'v4', 's[30:31]'], []))

    stage.sym[30] = ('ptr', ('root', 1), -4)
    stage.sym[31] = G.ptr_high_symbol(('root', 1), -4)
    refused('invalid pointer byte offset', lambda: stage.op_global_load_dword(['v5', 'v4', 's[30:31]'], []))

    stage.sym[30] = ('ptr', ('root', 1), 0xfffffffc)
    stage.sym[31] = G.ptr_high_symbol(('root', 1), 0xfffffffc)
    refused('global load byte offset 0x100000000',
            lambda: stage.op_global_load_dword(['v5', 'v4', 's[30:31]'], ['offset:4']))

    for other_kind in ('vs', 'ps', 'ms', None):
        stage.kind = other_kind
        refused('global_load_dword', lambda: stage.op_global_load_dword(['v5', 'v4', 's[6:7]'], []))
    stage.kind = 'cs'

    # Unlowered negative control: temporarily removing op_global_load_dword restores the baseline refusal.
    saved = G.Stage.op_global_load_dword
    try:
        delattr(G.Stage, 'op_global_load_dword')
        refused('global_load_dword', lambda: cs_pipe(KERNEL_A_BYTES))
    finally:
        G.Stage.op_global_load_dword = saved

    print('startup_compute_global_load controls: ok (3 kernels, multi-root SRV, 64-bit sum & 4-byte alignment guards, refusals)')


def check_binary(llvm_mc):
    assert assemble(llvm_mc, KERNEL_A_SOURCE) == KERNEL_A_BYTES
    assert assemble(llvm_mc, KERNEL_B_SOURCE) == KERNEL_B_BYTES
    assert assemble(llvm_mc, KERNEL_C_SOURCE) == KERNEL_C_BYTES
    for raw, asm in DECODE_CASES:
        assert assemble(llvm_mc, asm + '\n') == raw, asm
    print(f'startup_compute_global_load binary ({llvm_mc}): ok ({len(DECODE_CASES) + 3} assemblies exact)')


def check_decode(libllvm_path):
    from pixel_aot import disassemble
    blob = b''.join(raw for raw, _ in DECODE_CASES)
    llvm_lines = disassemble(blob, libllvm_path)
    ours = [expected_asm for _, expected_asm in DECODE_CASES]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'startup_compute_global_load decode ({libllvm_path}): ok ({len(DECODE_CASES)} GLOBAL instructions exact)')


def write_store_case(path, groups, expected, ud0, ud1, src_words):
    consts = [ud0 & MASK, ud1 & MASK] + [0] * 14
    groups_word = groups[0] | (groups[1] << 8) | (groups[2] << 16)
    path.write_bytes(struct.pack('<I320I16I320I', groups_word, *expected, *consts, *src_words))


def run_case(directory, compiler, renderer, adapter, name, hlsl, groups, want, ud0, ud1, src_words):
    src = directory / f'{name}.hlsl'
    cso = directory / f'{name}.cso'
    case = directory / f'{name}.case'
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([compiler, '-Gis', '-WX', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    write_store_case(case, groups, want, ud0, ud1, src_words)
    # Bind t0 as a bounded 320-dword raw buffer SRV view (matching runtime root-buffer view bounds).
    return subprocess.run([renderer, adapter, '--compute-store-sized-case', str(cso), str(case)],
                          capture_output=True, text=True)


def dispatch(directory, compiler, renderer, adapter):
    hlsl_a = cs_pipe(KERNEL_A_BYTES).hlsl_cs()
    hlsl_b = cs_pipe(KERNEL_B_BYTES).hlsl_cs()
    hlsl_c = cs_pipe(KERNEL_C_BYTES).hlsl_cs()

    # 4 parameterizations of Kernel A varying stride (including 1 = strictly adjacent words), bias, and source words.
    runs_a = (
        ('a_adjacent_stride1', 0x10203040, 0, 1),
        ('a_biased_adjacent', 0x55aa1234, 9, 1),
        ('a_stride3', 0x89abcdef, 3, 3),
        ('a_stride7', 0x31415926, 5, 7),
    )
    for name, seed, ud0, ud1 in runs_a:
        src_words = make_source_words(seed)
        want = kernel_a_reference(src_words, ud0, ud1)
        res = run_case(directory, compiler, renderer, adapter, name, hlsl_a, GROUPS_A, want, ud0, ud1, src_words)
        if res.returncode == 77:
            return 77
        assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (name, res.stdout, res.stderr)
        assert 'debug_errors=0 debug_warnings=0' in res.stdout or 'debug_layer=unavailable' in res.stdout, res.stdout

    # 2 parameterizations of Kernel B verifying negative immediate subtraction in range.
    runs_b = (
        ('b_neg16_adjacent', 0x27182818, 0, 1),
        ('b_neg16_stride5', 0xdead10cc, 7, 5),
    )
    for name, seed, ud0, ud1 in runs_b:
        src_words = make_source_words(seed)
        want = kernel_b_reference(src_words, ud0, ud1)
        res = run_case(directory, compiler, renderer, adapter, name, hlsl_b, GROUPS_A, want, ud0, ud1, src_words)
        assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (name, res.stdout, res.stderr)

    # 3 parameterizations of Kernel C verifying positive-offset 64-bit sum overflow (vaddr=0xfffffffc + 8 -> guest base
    # + 0x100000004, which would wrap to byte 4 in the low range), negative-offset underflow (vaddr in 0..12 below 16),
    # and low-bit inputs (ud0=1, ud1=5 stepping 1 raw byte per lane across distinct neighboring DWORDs) with nonzero
    # sentinels across all 320 low-range words.
    runs_c = (
        ('c_pos_overflow_neg_underflow', 0x13579bdf, 0xfffffffc, 0),
        ('c_pos_neg_boundary_sweep', 0x2468ace0, 0xffffffdc, 0xfffffff4),
        ('c_low_bits_neighbor_dwords', 0x97531eca, 1, 5),
    )
    for name, seed, ud0, ud1 in runs_c:
        src_words = make_source_words(seed)
        want = kernel_c_reference(src_words, ud0, ud1)
        res = run_case(directory, compiler, renderer, adapter, name, hlsl_c, GROUPS_A, want, ud0, ud1, src_words)
        assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (name, res.stdout, res.stderr)
        assert 'debug_errors=0 debug_warnings=0' in res.stdout or 'debug_layer=unavailable' in res.stdout, res.stdout

    # Reference negative controls on unmutated Kernel C: expecting the would-be wrapped low-range nonzero sentinel on
    # positive overflow (word 0: 0xfffffffc + 8 -> byte 4 = src_c[1]), negative underflow (word 2: 0 - 16 -> src_c[0]),
    # or an unaligned byte-gather / upper neighboring DWORD on low-bit input (word 3: raw 5 - 4 = byte 1 between
    # distinct neighboring src_low[0] and src_low[1]) must each fail at that exact word.
    src_c = make_source_words(0x13579bdf)
    want_c = kernel_c_reference(src_c, 0xfffffffc, 0)
    assert want_c[0] == 0 and want_c[1] == 0 and want_c[2] == 0 and want_c[3] == 0
    for ref_name, bad_word, wrapped_sentinel in (
        ('ref_pos_overflow_wrapped_low', 0, src_c[1]),
        ('ref_neg_underflow_wrapped_low', 2, src_c[0]),
    ):
        wrong_want = list(want_c)
        wrong_want[bad_word] = wrapped_sentinel
        res = run_case(directory, compiler, renderer, adapter, ref_name, hlsl_c, GROUPS_A, wrong_want, 0xfffffffc, 0, src_c)
        expected_err = f'word={bad_word} expected={wrapped_sentinel:08x} actual=00000000'
        assert res.returncode == 1 and expected_err in res.stderr, (ref_name, res.returncode, expected_err, res.stderr)

    src_low = make_source_words(0x97531eca)
    want_low = kernel_c_reference(src_low, 1, 5)
    unaligned_byte1 = ((src_low[0] >> 8) | (src_low[1] << 24)) & MASK
    assert src_low[0] != src_low[1] and unaligned_byte1 not in (src_low[0], src_low[1])
    assert want_low[3] == src_low[0] and want_low[15] == src_low[1]
    for ref_name, wrong_val in (
        ('ref_low_bit_unaligned_gather', unaligned_byte1),
        ('ref_low_bit_upper_neighbor', src_low[1]),
    ):
        wrong_want = list(want_low)
        wrong_want[3] = wrong_val
        res = run_case(directory, compiler, renderer, adapter, ref_name, hlsl_c, GROUPS_A, wrong_want, 1, 5, src_low)
        expected_err = f'word=3 expected={wrong_val:08x} actual={src_low[0]:08x}'
        assert res.returncode == 1 and expected_err in res.stderr, (ref_name, res.returncode, expected_err, res.stderr)

    # 10 GPU HLSL negative mutation controls: each must fail at the exact predicted word and value.
    src_a = make_source_words(0x10203040)
    want_a = kernel_a_reference(src_a, 0, 1)
    src_b = make_source_words(0x27182818)
    want_b = kernel_b_reference(src_b, 0, 1)

    mutations = (
        ('mut_wrong_vaddr', hlsl_a, 'v5 = r0.Load(a);', 'v5 = r0.Load(a + 4u);', want_a, 0, 1, src_a, 0, src_a[1]),
        ('mut_dropped_scalar_base', hlsl_a, 'v6 = (a <= 0xffffffcfu ? r0.Load(a + 48u) : 0u);', 'v6 = r0.Load(a);',
         want_a, 0, 1, src_a, 1, src_a[0]),
        ('mut_dropped_neg_imm', hlsl_a, 'v7 = (a <= 0xffffffbbu ? r0.Load(a + 68u) : 0u);',
         'v7 = (a <= 0xffffffafu ? r0.Load(a + 80u) : 0u);', want_a, 0, 1, src_a, 2, src_a[20]),
        ('mut_dropped_exec_guard', hlsl_a, 'if (exec & 1) v8 = (a <= 0xfffffff7u ? r0.Load(a + 8u) : 0u);',
         'v8 = (a <= 0xfffffff7u ? r0.Load(a + 8u) : 0u);', want_a, 0, 1, src_a, 23, src_a[5 + 2]),
        ('mut_flipped_neg16_sign', hlsl_b, 'v5 = (a >= 16u ? r0.Load(a - 16u) : 0u);', 'v5 = r0.Load(a + 16u);',
         want_b, 0, 1, src_b, 0, src_b[8]),
        ('mut_dropped_pos8_overflow_guard', hlsl_c, 'v5 = (a <= 0xfffffff7u ? r0.Load(a + 8u) : 0u);',
         'v5 = r0.Load(a + 8u);', want_c, 0xfffffffc, 0, src_c, 0, src_c[1]),
        ('mut_dropped_pos36_overflow_guard', hlsl_c, 'v6 = (a <= 0xffffffdbu ? r0.Load(a + 36u) : 0u);',
         'v6 = r0.Load(a + 36u);', want_c, 0xfffffffc, 0, src_c, 1, src_c[8]),
        ('mut_clamped_neg16_underflow_low', hlsl_c, 'v7 = (a >= 16u ? r0.Load(a - 16u) : 0u);',
         'v7 = r0.Load(a >= 16u ? a - 16u : 0u);', want_c, 0xfffffffc, 0, src_c, 2, src_c[0]),
        ('mut_wrapped_neg4_underflow_low', hlsl_c, 'v8 = (a >= 4u ? r0.Load(a - 4u) : 0u);',
         'v8 = r0.Load((a - 4u) & 0x7fu);', want_c, 0xfffffffc, 0, src_c, 3, src_c[31]),
        ('mut_low_bits_unaligned_gather', hlsl_c.replace('v11 = 0xfffffffcu & v11;', ''),
         'v8 = (a >= 4u ? r0.Load(a - 4u) : 0u);', 'v8 = ldu(r0, a - 4u);',
         want_low, 1, 5, src_low, 3, unaligned_byte1),
    )
    for name, base_hlsl, old, new, want, ud0, ud1, src_words, bad_word, bad_actual in mutations:
        assert base_hlsl.count(old) == 1, (name, old)
        res = run_case(directory, compiler, renderer, adapter, name, base_hlsl.replace(old, new),
                       GROUPS_A, want, ud0, ud1, src_words)
        expected_err = f'word={bad_word} expected={want[bad_word]:08x} actual={bad_actual:08x}'
        assert res.returncode == 1 and expected_err in res.stderr, (name, res.returncode, expected_err, res.stdout, res.stderr)

    print(f'startup_compute_global_load {adapter}: ok (9 dispatches + 4 reference controls + 10 negative mutations caught)')
    return 0


IMAGE_TABLE_BASE = 320
IMAGE_JUMP_TARGET = 24
IMAGE_MODEL_BASE = 0x12ffffff00
# The final data dword deliberately matches the END encoding.
IMAGE_TABLE = tuple(0x49a10203 ^ (index * 0x01050913) for index in range(15)) + (0xbf810000,)


def image_program(mode='global', control=None):
    """An original 32-lane kernel with a GETPC-relative table and a forward SETPC."""
    from startup_gfx2hlsl_add_sub_co_ci_check import vop3b

    def carry_op(*args):
        return struct.pack('<II', *vop3b(*args))

    jump_delta = IMAGE_JUMP_TARGET - 4
    table_delta = IMAGE_TABLE_BASE - IMAGE_JUMP_TARGET
    code = (
        encode_sop1(0x1f, 34 if mode == 'flat' else 36, 0)
        + (encode_sop1(0x04, 36, 34) if mode == 'flat' else encode_sopp(0))
        + encode_sop2(0x00, 36, 36, 128 + jump_delta)
        + encode_sop2(0x04, 37, 37, 128)
        + encode_sop1(0x20, 0, 36)                           # s_setpc_b64 s[36:37]
        + encode_sopp(0x01)                                 # unreachable early termination
        + encode_vop2(0x1b, 12, 128 + (63 if control == 'outside' else 15), 0)
        + (encode_vop1(0x01, 12, 9) if control == 'unbounded' else encode_vop2(0x1a, 12, 130, 12))
    )
    assert IMAGE_JUMP_TARGET == 24
    if mode == 'saddr':
        # Negative displacement tests sign-extension on the scalar high add.
        delta = -4 if control == 'overlap' else table_delta + 4
        code += encode_sop2(0x00, 38, 36, 255, lit=delta)
        code += encode_sop2(0x04, 39, 37, 193 if delta < 0 else 128)
        code += encode_global(GLOBAL_LOAD_DWORD, 15, 12, 38, offset=-4)
    else:
        code += (
            (encode_sop1(0x03, 38, 37) if mode == 'flat' else encode_sopp(0))
            + encode_sop1(0x04, 40, 128)                   # known zero for the dropped-carry mutant
            + (encode_vop1(0x01, 21, 38) if mode == 'flat' else encode_sopp(0))
            + carry_op(0x30f, 20, 106, 36, 255, 0) + struct.pack('<I', table_delta)
            + (encode_vop2(0x28, 21, 128, 21) if mode == 'flat' else
               carry_op(0x128, 21, 40, 37, 128, 40 if control == 'carry' else 106))
            + carry_op(0x30f, 20, 42, 0x100 + 20, 0x100 + 12, 0)
            + carry_op(0x128, 21, 44, 0x100 + 21, 128, 42)
        )
        seg, saddr = (0 if mode == 'flat' else 2), 0x7d
        if control == 'store':
            code += encode_global(GLOBAL_STORE_DWORD, 0, 20, saddr, seg=seg, data=12)
        else:
            code += encode_global(GLOBAL_LOAD_DWORD, 15, 20, saddr, seg=seg)
    if control == 'early-exit':
        code += encode_sopp(0x08, 9)                        # s_cbranch_execz to final s_endpgm
    code += (
        encode_sopp(0x0c, 0x3f70)                            # s_waitcnt vmcnt(0)
        + encode_sop1(0x03, 10, 8)
        + encode_sop1(0x03, 11, 128)
        + encode_smem(0x02, 16, 10)
        + encode_sopp(0x0c, 0xc07f)                          # s_waitcnt lgkmcnt(0)
        + encode_vop2(0x1a, 13, 130, 0)
        + encode_mubuf(0x1c, 15, 13, 16)
        + encode_sopp(0x01)
    )
    assert len(code) <= IMAGE_TABLE_BASE, len(code)
    return code.ljust(IMAGE_TABLE_BASE, b'\0') + struct.pack('<16I', *IMAGE_TABLE)


def image_reference():
    # The displacement from the jump target forces low-word overflow in
    # the model. A high word that ignores carry points 4 GiB before the table.
    pc = IMAGE_MODEL_BASE + IMAGE_JUMP_TARGET
    lo = (pc & MASK) + IMAGE_TABLE_BASE - IMAGE_JUMP_TARGET
    assert lo > MASK
    table = (((pc >> 32) + (lo >> 32)) << 32) | (lo & MASK)
    assert table == IMAGE_MODEL_BASE + IMAGE_TABLE_BASE
    assert ((pc & ~MASK) | (lo & MASK)) == table - (1 << 32)
    return [IMAGE_TABLE[lane & 15] for lane in range(32)] + [SENTINEL] * (WORDS - 32)


def check_image_controls():
    for mode in ('global', 'flat', 'saddr'):
        pipe = cs_pipe(image_program(mode), group=(32, 1, 1))
        assert pipe.cs.calculated_pc[16] == IMAGE_JUMP_TARGET
        assert 'srv ' not in pipe.manifest()
        assert tuple(pipe.cs.image_tables.values()) == (IMAGE_TABLE,)
    for mode, control, reason in (
            ('global', 'unbounded', 'load from the program image with an unbounded offset'),
            ('flat', 'outside', 'load from the program image outside the image'),
            ('saddr', 'overlap', 'load from the program image overlaps executed code'),
            ('global', 'store', 'store to the program image'),
            ('global', 'carry', 'missing matching high-word carry for program image address')):
        refused(reason, lambda: cs_pipe(image_program(mode, control), group=(32, 1, 1)))
    early = cs_pipe(image_program(control='early-exit'), group=(32, 1, 1))
    assert tuple(early.cs.image_tables.values()) == (IMAGE_TABLE,)
    image_reference()
    print('program image controls: 3 load forms, early exit and 5 refusals')


def check_image_binary(llvm_mc):
    for mode in ('global', 'flat', 'saddr'):
        program = image_program(mode)
        end = cs_pipe(program, group=(32, 1, 1)).cs.ins[-1][0] + 4
        assert assemble(llvm_mc, disassemble_words(program[:end])) == program[:end], mode
    print('program image binary: all 3 assemblies exact')


def dispatch_image(directory, compiler, renderer, adapter):
    want = image_reference()
    source = image_program()
    src_words = [0] * WORDS
    groups = (1, 1, 1)
    for mode in ('global', 'flat', 'saddr'):
        program = image_program(mode)
        hlsl = cs_pipe(program, group=(32, 1, 1)).hlsl_cs()
        result = run_case(directory, compiler, renderer, adapter, 'image_' + mode, hlsl,
                          groups, want, 0, 0, src_words)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (
            mode, result.stdout, result.stderr)
        assert 'debug_errors=0 debug_warnings=0' in result.stdout or 'debug_layer=unavailable' in result.stdout
        print(f'program image {mode}, OS {platform.version()}')
        print(result.stdout.strip())

    hlsl = cs_pipe(source, group=(32, 1, 1)).hlsl_cs()
    old = ', '.join('0x%08xu' % value for value in IMAGE_TABLE)
    wrong = struct.unpack_from('<16I', source)
    new = ', '.join('0x%08xu' % value for value in wrong)
    assert hlsl.count(old) == 1 and wrong[0] != want[0]
    result = run_case(directory, compiler, renderer, adapter, 'image_wrong_table', hlsl.replace(old, new),
                      groups, want, 0, 0, src_words)
    assert result.returncode == 1 and f'word=0 expected={want[0]:08x} actual={wrong[0]:08x}' in result.stderr, (
        result.stdout, result.stderr)
    refused('missing matching high-word carry for program image address',
            lambda: cs_pipe(image_program(control='carry'), group=(32, 1, 1)))
    print(f'program image {adapter}: 3 readbacks, wrong table and missing carry caught')
    return 0


def main(argv):
    if not argv or argv == ['controls']:
        check_controls()
        check_image_controls()
        return 0
    if len(argv) == 2 and argv[0] in ('binary', '--llvm-mc'):
        check_controls()
        check_binary(argv[1])
        check_image_binary(argv[1])
        return 0
    if len(argv) == 2 and argv[0] == 'decode':
        check_decode(argv[1])
        return 0
    if len(argv) == 4 and argv[0] in ('dispatch', '--dispatch', 'image-dispatch'):
        out = ROOT / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            run = dispatch_image if argv[0] == 'image-dispatch' else dispatch
            return run(Path(tmp), *argv[1:])
    raise SystemExit(__doc__)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
