#!/usr/bin/env python3
"""Verify COMPUTE_PGM_RSRC2.TG_SIZE_EN (#2274 Part A) in tools/startup/gfx2hlsl.py.

  (no arguments)                        pure-Python translation, uniformity and refusal controls
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                        translate, compile with DXC cs_6_0, dispatch and compare all 320 words
  control <dropped-base|swapped-mask> <dxc> <offscreen_check> <warp|hardware>
                                        mutated translation must mismatch on readback
  waves-dispatch <dxc> <offscreen_check> <warp|hardware>
                                        64 threads (two wave32 waves): every thread stores its wave-ID bits and the scc select
  waves-control <wave-id-zero|wave-width-16> <dxc> <offscreen_check> <warp|hardware>

  fields-dispatch <dxc> <offscreen_check> warp
                                        compare both fields with the CPU for six guest group/wave configurations
  fields-control <wave32-shift|wave64-shift|drop-wave-count> <dxc> <offscreen_check> warp

The entry SGPR follows USER_SGPR and enabled group-ID registers. Only the wave-ID and
wave-count masks are supported. Unknown wave size still permits the legacy small-group
wave-ID fold because its result is independent of the guest wave size.
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
from startup_gfx2hlsl_compute_check import (  # noqa: E402
    SENTINEL,
    TGID_X,
    TGID_Y,
    TGID_Z,
    TG_SIZE_EN,
    USER7,
    WORDS,
    refused,
    root_blob,
    tidig,
)


def le32(word):
    return struct.pack('<I', word & 0xffffffff)


def sop1(op, sdst, ssrc0, lit=None):
    b = le32(0xbe800000 | ((sdst & 0x7f) << 16) | ((op & 0xff) << 8) | (ssrc0 & 0xff))
    return b if lit is None else b + le32(lit)


def sop2(op, sdst, ssrc0, ssrc1, lit=None):
    b = le32(0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))
    return b if lit is None else b + le32(lit)


def sopk(op, sdst, simm16):
    return le32(0xb0000000 | ((op & 0x1f) << 23) | ((sdst & 0x7f) << 16) | (simm16 & 0xffff))


def sopc(op, ssrc0, ssrc1, lit=None):
    b = le32(0xbf000000 | ((op & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))
    return b if lit is None else b + le32(lit)


def sopp(op, simm=0):
    return le32(0xbf800000 | ((op & 0x7f) << 16) | (simm & 0xffff))


def vop1(op, vdst, src0, lit=None):
    b = le32(0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def vop2(op, vdst, src0, vsrc1, lit=None):
    b = le32(((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff))
    return b if lit is None else b + le32(lit)


def vop3(op, vdst, src0, src1, src2=0):
    w0 = 0xd4000000 | ((op & 0x3ff) << 16) | (vdst & 0xff)
    w1 = ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff)
    return le32(w0) + le32(w1)


S_MOV_B32 = 0x03
S_MOV_B64 = 0x04
S_ADD_U32 = 0x00
S_CSELECT_B32 = 0x0a
S_AND_B32 = 0x0e
S_OR_B32 = 0x10
S_LSHL_B32 = 0x1e
S_LSHR_B32 = 0x1f
S_ADDK_I32 = 0x0f
S_CMP_EQ_U32 = 0x00
V_MOV_B32 = 0x01
V_LSHLREV_B32 = 0x1a
V_ADD_NC_U32 = 0x25
V_MAD_U32_U24 = 0x143
V_MBCNT_LO_U32_B32 = 0x365
V_MBCNT_HI_U32_B32 = 0x366

INLINE_0 = 128
INLINE_2 = 130
INLINE_3 = 131
INLINE_4 = 132
INLINE_6 = 134
INLINE_17 = 128 + 17
INLINE_42 = 128 + 42
INLINE_NEG1 = 193
M0 = 124
EXEC_LO = 126
VCC_LO = 106
LIT = 255

S_LOAD_X8_S12_S10 = bytes.fromhex('05030cf4000000fa')
S_WAITCNT_LGKM0 = bytes.fromhex('7fc08cbf')
BUFFER_STORE_X4_V8_V4_S16 = bytes.fromhex('001078e004080480')
S_ENDPGM = sopp(0x01, 0)

MASK_A = 0x85241249
MASK_B = 0x0f0f0f0f
GROUP = (4, 2, 2)
DISPATCH_GROUPS = (2, 1, 1)
RSRC2 = USER7 | TGID_X | TG_SIZE_EN | tidig(3)

# Two-group compute shader with TGID_X (s7) and TG_SIZE_EN (s8).
# Each thread computes:
#   s20 = s8 & 0xfc0 (0u; scc = false)
#   s23 = scc ? 17 : 42 (42, verifying scc = false)
#   v8  = v_mbcnt_lo_u32_b32(MASK_A, s20)
#   v9  = v_mbcnt_lo_u32_b32(MASK_B, v8)
#   v10 = s23
#   v11 = gid.x * 16 + (tid.x + 4 * (tid.y + 2 * tid.z))
# and stores uint4(v8, v9, v10, v11) at record index v11.
MAIN_PROGRAM = b''.join([
    sop1(S_MOV_B32, 10, 6),
    sop1(S_MOV_B32, 11, INLINE_0),
    S_LOAD_X8_S12_S10,
    sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
    sop2(S_CSELECT_B32, 23, INLINE_17, INLINE_42),
    sop1(S_MOV_B32, 21, LIT, MASK_A),
    sop1(S_MOV_B32, 22, LIT, MASK_B),
    vop3(V_MAD_U32_U24, 3, 256 + 2, INLINE_2, 256 + 1),
    vop3(V_MAD_U32_U24, 3, 256 + 3, INLINE_4, 256 + 0),
    sop2(S_LSHL_B32, 24, 7, INLINE_4),
    vop2(V_ADD_NC_U32, 11, 24, 3),
    vop2(V_LSHLREV_B32, 4, INLINE_4, 11),
    vop3(V_MBCNT_LO_U32_B32, 8, 21, 20),
    vop3(V_MBCNT_LO_U32_B32, 9, 22, 256 + 8),
    vop1(V_MOV_B32, 10, 23),
    S_WAITCNT_LGKM0,
    BUFFER_STORE_X4_V8_V4_S16,
    S_ENDPGM,
])

EXPECTED_MAIN_ASM = [
    's_mov_b32 s10, s6',
    's_mov_b32 s11, 0',
    's_load_dwordx8 s[12:19], s[10:11], 0x0',
    's_and_b32 s20, s8, 0xfc0',
    's_cselect_b32 s23, 17, 42',
    's_mov_b32 s21, 0x85241249',
    's_mov_b32 s22, 0xf0f0f0f',
    'v_mad_u32_u24 v3, v2, 2, v1',
    'v_mad_u32_u24 v3, v3, 4, v0',
    's_lshl_b32 s24, s7, 4',
    'v_add_nc_u32_e32 v11, s24, v3',
    'v_lshlrev_b32_e32 v4, 4, v11',
    'v_mbcnt_lo_u32_b32 v8, s21, s20',
    'v_mbcnt_lo_u32_b32 v9, s22, v8',
    'v_mov_b32_e32 v10, s23',
    's_waitcnt lgkmcnt(0)',
    'buffer_store_dwordx4 v[8:11], v4, s[16:19], 0 offen',
    's_endpgm',
]


# Multi-wave variant: 64 threads per group (two wave32 waves), TG_SIZE[11:6] is the wave ID. The program is stated to be wave32
# by the caller (wave_size=32), exactly as the AOT path does. Each thread stores uint4(wave bits, 17 or 42 by scc, wave bits, index).
WAVES_GROUP = (8, 4, 2)
WAVES_DISPATCH_GROUPS = (1, 1, 1)
INLINE_8 = 136
WAVES_PROGRAM = b''.join([
    sop1(S_MOV_B32, 10, 6),
    sop1(S_MOV_B32, 11, INLINE_0),
    S_LOAD_X8_S12_S10,
    sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
    sop2(S_CSELECT_B32, 23, INLINE_17, INLINE_42),
    vop3(V_MAD_U32_U24, 3, 256 + 2, INLINE_4, 256 + 1),
    vop3(V_MAD_U32_U24, 3, 256 + 3, INLINE_8, 256 + 0),
    sop2(S_LSHL_B32, 24, 7, INLINE_6),
    vop2(V_ADD_NC_U32, 11, 24, 3),
    vop2(V_LSHLREV_B32, 4, INLINE_4, 11),
    vop1(V_MOV_B32, 8, 20),
    vop1(V_MOV_B32, 9, 23),
    vop1(V_MOV_B32, 10, 20),
    S_WAITCNT_LGKM0,
    BUFFER_STORE_X4_V8_V4_S16,
    S_ENDPGM,
])


# Each guest thread writes two words through u0. The 320-word readback buffer
# covers all 128 threads and leaves sentinel words to detect stray writes.
FIELD_CASES = (
    ((33, 1, 1), 32),
    ((8, 3, 2), 32),
    ((8, 4, 2), 32),
    ((8, 6, 2), 32),
    ((8, 4, 2), 64),
    ((16, 4, 2), 64),
)
BUFFER_STORE_X2_V8_V4_S16 = bytes.fromhex('001074e004080480')


def fields_program(group):
    x, y, _ = group
    return b''.join([
        sop1(S_MOV_B32, 10, 6),
        sop1(S_MOV_B32, 11, INLINE_0),
        S_LOAD_X8_S12_S10,
        sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
        sop2(S_AND_B32, 21, LIT, 8, 0x3f),
        vop3(V_MAD_U32_U24, 3, 256 + 2, 128 + y, 256 + 1),
        vop3(V_MAD_U32_U24, 3, 256 + 3, 128 + x, 256 + 0),
        vop2(V_LSHLREV_B32, 4, INLINE_3, 3),
        vop1(V_MOV_B32, 8, 20),
        vop1(V_MOV_B32, 9, 21),
        S_WAITCNT_LGKM0,
        BUFFER_STORE_X2_V8_V4_S16,
        S_ENDPGM,
    ])


def fields_expected_words(group, wave):
    threads = group[0] * group[1] * group[2]
    wave_count = len(range(0, threads, wave))
    out = [SENTINEL] * WORDS
    for wave_id, start in enumerate(range(0, threads, wave)):
        for thread in range(start, min(start + wave, threads)):
            out[2 * thread:2 * thread + 2] = [wave_id * 64, wave_count]
    return out


def waves_expected_words():
    out = [SENTINEL] * WORDS
    for lane in range(64):
        wave_bits = (lane >> 5) << 6
        out[4 * lane:4 * lane + 4] = [wave_bits, 17 if wave_bits else 42, wave_bits, lane]
    return out


def translate(code=MAIN_PROGRAM, rsrc2=RSRC2, group=GROUP, wave_size=None):
    return G.ComputePipeline(code, root_blob(), rsrc2, group, wave_size=wave_size)


def expected_words():
    out = [SENTINEL] * WORDS
    gx_count, gy_count, gz_count = DISPATCH_GROUPS
    lx, ly, lz = GROUP
    assert gy_count == 1 and gz_count == 1
    for gx in range(gx_count):
        for tz in range(lz):
            for ty in range(ly):
                for tx in range(lx):
                    lane = tx + lx * (ty + ly * tz)
                    below = (1 << lane) - 1
                    w8 = (MASK_A & below).bit_count()
                    w9 = (MASK_B & below).bit_count() + w8
                    w10 = 42
                    w11 = gx * (lx * ly * lz) + lane
                    out[4 * w11:4 * w11 + 4] = [w8, w9, w10, w11]
    return out


def check_controls():
    assert [asm for _, _, _, asm in G.decode(MAIN_PROGRAM)] == EXPECTED_MAIN_ASM

    # 1. Placement of TG_SIZE SGPR across TGID combinations (USER7 -> s7..s10).
    for rsrc2, tg_reg in (
        (USER7 | TG_SIZE_EN, 7),
        (USER7 | TGID_X | TG_SIZE_EN, 8),
        (USER7 | TGID_X | TGID_Z | TG_SIZE_EN, 9),
        (USER7 | TGID_X | TGID_Y | TGID_Z | TG_SIZE_EN, 10),
    ):
        pipe = translate(code=S_ENDPGM, rsrc2=rsrc2, group=(8, 4, 1))
        assert pipe.tg_size_sgpr == tg_reg, (hex(rsrc2), pipe.tg_size_sgpr, tg_reg)
        assert pipe.cs.sym.get(tg_reg) == ('tg_size',), (hex(rsrc2), pipe.cs.sym)
        assert not pipe.cs.uniform_source('s%d' % tg_reg)
        assert pipe.cs.is_const(tg_reg) is None
        assert pipe.cs.const_u32('s%d' % tg_reg) is None

    # 2. Standalone s_and_b32 R, TG, 0xfc0 (both operand orders and in-place R == TG).
    for and_insn in (
        sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
        sop2(S_AND_B32, 20, LIT, 8, 0xfc0),
    ):
        pipe = translate(code=and_insn + S_ENDPGM)
        hlsl = pipe.hlsl_cs()
        assert '    s20 = 0u;\n    scc = false;' in hlsl, hlsl
        assert 'uint s8' not in hlsl, hlsl
        assert pipe.cs.sym.get(20) == ('const', 0)
        assert pipe.cs.sym.get(8) == ('tg_size',)
        assert pipe.cs.scc_uniform is True

    # In-place overwrite `s_and_b32 s8, s8, 0xfc0` lowers to 0u and turns s8 into ('const', 0).
    inplace = translate(code=sop2(S_AND_B32, 8, 8, LIT, 0xfc0) + sop1(S_MOV_B32, 20, 8) + S_ENDPGM)
    inplace_hlsl = inplace.hlsl_cs()
    assert '    s8 = 0u;\n    scc = false;\n    s20 = 0x00000000u;' in inplace_hlsl, inplace_hlsl
    assert inplace.cs.sym.get(8) == ('const', 0)

    # 3. Full program: s_and_b32 R, TG, 0xfc0 feeds R into v_mbcnt_lo_u32_b32, plus standalone v_mbcnt_lo/hi.
    main_hlsl = translate().hlsl_cs()
    assert '    s20 = 0u;\n    scc = false;' in main_hlsl, main_hlsl
    assert 'countbits(0x85241249u & ((1u << (tid.x + 4u * (tid.y + 2u * tid.z))) - 1u)) + 0x00000000u' in main_hlsl, main_hlsl
    assert 'countbits(0x0f0f0f0fu & ((1u << (tid.x + 4u * (tid.y + 2u * tid.z))) - 1u)) + v8' in main_hlsl, main_hlsl

    hi_prog = b''.join([
        sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
        vop3(V_MBCNT_HI_U32_B32, 8, INLINE_NEG1, 20),
        S_ENDPGM,
    ])
    hi_hlsl = translate(code=hi_prog).hlsl_cs()
    assert '    v8 = 0x00000000u;' in hi_hlsl, hi_hlsl

    # 4. Groups above 32 threads require a known guest wave size.
    for bad_group in ((8, 4, 2), (33, 1, 1), (64, 1, 1)):
        refused('TG_SIZE_EN requires at most 32 threads per group',
                lambda g=bad_group: translate(code=S_ENDPGM, group=g))
    wave_and = sop2(S_AND_B32, 20, 8, LIT, 0xfc0) + S_ENDPGM
    refused('TG_SIZE_EN requires at most 32 threads per group', lambda: translate(code=wave_and, group=(8, 4, 2)))  # no wave size known
    waves_hlsl = translate(code=WAVES_PROGRAM, group=WAVES_GROUP, wave_size=32).hlsl_cs()
    assert '    s20 = (((tid.x + 8u * (tid.y + 4u * tid.z)) >> 5u) << 6u);\n    scc = s20 != 0u;' in waves_hlsl, waves_hlsl
    waves_pipe = translate(code=wave_and, group=(8, 4, 2), wave_size=32)
    assert not waves_pipe.cs.uniform_source('s20') and waves_pipe.cs.scc_uniform is False
    # A program that writes exec_lo with a 32-bit op proves wave32; a 64-bit exec form refuses; no exec op states nothing.
    assert G.wave32_exec_proof([(0, 4, 's_mov_b32', 's_mov_b32 exec_lo, -1')])
    assert G.wave32_exec_proof([(0, 4, 's_and_saveexec_b32', 's_and_saveexec_b32 s4, vcc_lo')])
    assert not G.wave32_exec_proof([(0, 4, 's_mov_b32', 's_mov_b32 s4, exec_lo')])  # a read of exec_lo states nothing
    assert not G.wave32_exec_proof([(0, 4, 's_mov_b64', 's_mov_b64 exec, -1')])
    assert not G.wave32_exec_proof([(0, 4, 's_mov_b32', 's_mov_b32 exec_lo, -1'), (4, 4, 's_mov_b64', 's_mov_b64 exec, -1')])
    assert not G.wave32_exec_proof([(0, 4, 's_endpgm', 's_endpgm')])

    field_hlsl = {(group, wave): check_fields_translation(group, wave) for group, wave in FIELD_CASES}
    for control, (case, old, new, _) in FIELD_MUTANTS.items():
        mutate(field_hlsl[case], control, {control: (old, new)})
    low_and = sop2(S_AND_B32, 21, 8, LIT, 0x3f) + S_ENDPGM
    refused(G.TG_SIZE_WAVES_REFUSAL, lambda: translate(code=low_and))
    for wave, group, copy_exec in (
        (32, (33, 1, 1), sop1(S_MOV_B32, EXEC_LO, EXEC_LO)),
        (64, (128, 1, 1), sop1(S_MOV_B64, 24, EXEC_LO)),
    ):
        code = copy_exec + fields_program(group)
        assert translate(code=code, group=group).cs.tg_size_wave == wave

    # The existing lane-operation refusal still applies to a partial guest wave64.
    mbcnt = vop3(V_MBCNT_LO_U32_B32, 8, INLINE_NEG1, INLINE_0) + S_ENDPGM
    refused('in a larger group requires a proven wave32 program',
            lambda: translate(code=mbcnt, group=(48, 1, 1), wave_size=64))

    # 5. Disallowed reads of TG_SIZE are refused.
    bad_tg_reads = [
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, 20, 8, LIT, 0xfff)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, 20, 8, LIT, 0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, 20, 8, 8)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, M0, 8, LIT, 0xfc0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, EXEC_LO, 8, LIT, 0xfc0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, VCC_LO, 8, LIT, 0xfc0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_AND_B32, 106, 8, LIT, 0xfc0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop1(S_MOV_B32, 20, 8)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_OR_B32, 20, 8, INLINE_0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_LSHR_B32, 20, 8, INLINE_6)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop2(S_LSHL_B32, 20, 8, INLINE_2)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sop1(S_MOV_B64, 20, 8)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', sopc(S_CMP_EQ_U32, 8, INLINE_0)),
        ('s_addk_i32 on tg_size data', sopk(S_ADDK_I32, 8, 1)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', vop2(V_ADD_NC_U32, 3, 8, 0)),
        ('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', vop1(V_MOV_B32, 3, 8)),
        ('v_mbcnt_lo_u32_b32 requires a VGPR or known base', vop3(V_MBCNT_LO_U32_B32, 8, INLINE_NEG1, 8)),
        ('v_mbcnt_lo_u32_b32 requires a known mask', vop3(V_MBCNT_LO_U32_B32, 8, 8, INLINE_0)),
    ]
    for msg, insn in bad_tg_reads:
        refused(msg, lambda c=insn + S_ENDPGM: translate(code=c).hlsl_cs())

    # Modified s_and_b32 (e.g. non-empty mods on unit probe) is also refused.
    probe = G.Stage.__new__(G.Stage)
    probe.sym = {8: ('tg_size',)}
    probe.used_s = set()
    probe.lines = []
    probe.exec_known = True
    refused('TG_SIZE register s8 read outside s_and_b32 with 0xfc0',
            lambda: probe.scalar('s_and_b32', ['s20', 's8', '0xfc0'], ['clamp']))

    # Descriptor-path protection: mixing descriptor and tg_size in s_add_u32 refuses tg_size instead of silently dropping it.
    desc_mix = b''.join([
        sop1(S_MOV_B32, 10, 6),
        sop1(S_MOV_B32, 11, INLINE_0),
        S_LOAD_X8_S12_S10,
        sop2(S_ADD_U32, 12, 12, 8),
        S_ENDPGM,
    ])
    refused('TG_SIZE register s8 read outside s_and_b32 with 0xfc0', lambda: translate(code=desc_mix).hlsl_cs())

    # 6. Overwriting TG first restores ordinary scalar behavior.
    overwrite_const = b''.join([
        sop1(S_MOV_B32, 8, 128 + 5),
        sop2(S_AND_B32, 20, 8, LIT, 0x3f),
        vop2(V_ADD_NC_U32, 3, 8, 0),
        S_ENDPGM,
    ])
    ov_const_hlsl = translate(code=overwrite_const).hlsl_cs()
    assert '    s8 = 0x00000005u;\n    s20 = 0x00000005u & 0x0000003fu;' in ov_const_hlsl, ov_const_hlsl
    assert '    v3 = 0x00000005u + v0;' in ov_const_hlsl, ov_const_hlsl

    overwrite_runtime = b''.join([
        sop1(S_MOV_B32, 8, 7),
        sop2(S_AND_B32, 20, 8, LIT, 0x3f),
        vop2(V_ADD_NC_U32, 3, 8, 0),
        S_ENDPGM,
    ])
    ov_rt_hlsl = translate(code=overwrite_runtime).hlsl_cs()
    assert '    s8 = s7;\n    s20 = s8 & 0x0000003fu;' in ov_rt_hlsl, ov_rt_hlsl
    assert '    v3 = s8 + v0;' in ov_rt_hlsl, ov_rt_hlsl

    # 7. Branch / execz interactions: s_and_b32 inside s_cbranch_execz live-out is refused; dead-out is accepted.
    execz_live = b''.join([
        sopp(0x08, 2),  # s_cbranch_execz +2 dwords (skips 8-byte s_and_b32 with literal 0xfc0)
        sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
        vop1(V_MOV_B32, 3, 20),
        S_ENDPGM,
    ])
    refused('definition of s20, live where the block joins',
            lambda: translate(code=execz_live, group=(128, 1, 1), wave_size=32).hlsl_cs())

    execz_dead = b''.join([
        sopp(0x08, 2),  # s_cbranch_execz +2 dwords
        sop2(S_AND_B32, 20, 8, LIT, 0xfc0),
        S_ENDPGM,
    ])
    translate(code=execz_dead).hlsl_cs()

    # Overwriting TG on one branch arm and reading it at join refuses (never leaks ('tg_size',) across a join).
    join_leak = b''.join([
        sopc(S_CMP_EQ_U32, 7, INLINE_0),
        sopp(0x04, 2),  # s_cbranch_scc0 +2 dwords
        sop2(S_AND_B32, 8, 8, LIT, 0xfc0),
        sop1(S_MOV_B32, 20, 8),
        S_ENDPGM,
    ])
    refused('holding joined at a branch target', lambda: translate(code=join_leak).hlsl_cs())

    print('tg_size_en: entry layout, both fields in six group/wave cases, mutation sites and refusals pass')


CONTROLS = {
    'dropped-base': (' + v8', ''),
    'swapped-mask': ('v8 = (countbits(0x85241249u', 'v8 = (countbits(0x0f0f0f0fu'),
}

CONTROL_MISMATCH = {
    'dropped-base': 'word=5 expected=00000002 actual=00000001',
    'swapped-mask': 'word=8 expected=00000001 actual=00000002',
}

# Multi-wave variant (64 threads = two wave32 waves, TG_SIZE[11:6] is the wave ID): the wave ID of a thread is its flat index / 32.
WAVE_ID_LINE = 's20 = (((tid.x + 8u * (tid.y + 4u * tid.z)) >> 5u) << 6u);'
WAVES_CONTROLS = {
    # every thread gets wave 0: thread 32's record shows 0 where wave bits 0x40 belong
    'wave-id-zero': (WAVE_ID_LINE, 's20 = 0u;'),
    # waves of 16 threads instead of 32: thread 16 is wrongly in wave 1
    'wave-width-16': ('>> 5u) << 6u);', '>> 4u) << 6u);'),
}
WAVES_MISMATCH = {
    'wave-id-zero': 'word=128 expected=00000040 actual=00000000',
    'wave-width-16': 'word=64 expected=00000000 actual=00000040',
}


def mutate(hlsl, control, controls=None):
    old, new = (controls or CONTROLS)[control]
    assert hlsl.count(old) == 1, (control, hlsl)
    return hlsl.replace(old, new)


def check_fields_translation(group, wave):
    pipe = translate(code=fields_program(group), group=group, wave_size=wave)
    hlsl = pipe.hlsl_cs()
    expected = fields_expected_words(group, wave)
    assert pipe.cs.sym[21] == ('const', expected[1])
    assert pipe.cs.uniform_source('s21') and pipe.cs.scc_uniform
    assert 's21 = %du;\n    scc = true;' % expected[1] in hlsl
    if expected[1] > 1:
        assert '>> %du) << 6u);' % (5 if wave == 32 else 6) in hlsl
    else:
        assert 's20 = 0u;\n    scc = false;' in hlsl
    return hlsl


FIELD_MUTANTS = {
    'wave32-shift': (FIELD_CASES[0], '>> 5u) << 6u);', '>> 6u) << 6u);',
                     'word=64 expected=00000040 actual=00000000'),
    'wave64-shift': (FIELD_CASES[-1], '>> 6u) << 6u);', '>> 5u) << 6u);',
                     'word=64 expected=00000000 actual=00000040'),
    'drop-wave-count': (FIELD_CASES[0], 'v9 = 0x00000002u;', 'v9 = 0u;',
                        'word=1 expected=00000002 actual=00000000'),
}


def check_fields_dispatch(dxc, runner, adapter, control=None):
    if control is not None:
        (group, wave), old, new, mismatch = FIELD_MUTANTS[control]
        hlsl = mutate(check_fields_translation(group, wave), control, {control: (old, new)})
        return run_dispatch(dxc, runner, adapter, hlsl, (1, 1, 1), group,
                            fields_expected_words(group, wave), control, mismatch)
    for group, wave in FIELD_CASES:
        print('guest wave%d' % wave)
        status = run_dispatch(dxc, runner, adapter, check_fields_translation(group, wave),
                              (1, 1, 1), group, fields_expected_words(group, wave))
        if status:
            return status
    return 0


def check_dispatch(dxc, runner, adapter, control=None, waves=False):
    controls, mismatch = (WAVES_CONTROLS, WAVES_MISMATCH) if waves else (CONTROLS, CONTROL_MISMATCH)
    groups, group, expected = (WAVES_DISPATCH_GROUPS, WAVES_GROUP, waves_expected_words()) if waves else (
        DISPATCH_GROUPS, GROUP, expected_words())
    hlsl = (translate(code=WAVES_PROGRAM, group=WAVES_GROUP, wave_size=32) if waves else translate()).hlsl_cs()
    if control:
        if control not in controls:
            raise SystemExit(f'unknown control {control!r}')
        hlsl = mutate(hlsl, control, controls)
    return run_dispatch(dxc, runner, adapter, hlsl, groups, group, expected,
                        control, mismatch.get(control))


def run_dispatch(dxc, runner, adapter, hlsl, groups, group, expected, control=None, mismatch=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        d = Path(tmp)
        src, cso, case = d / 'tg_size.hlsl', d / 'tg_size.cso', d / 'tg_size.case'
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        gx, gy, gz = groups
        constants = [0] * 16
        case.write_bytes(
            struct.pack('<I', gx | (gy << 8) | (gz << 16))
            + struct.pack('<%dI' % WORDS, *expected)
            + struct.pack('<16I', *constants)
        )
        run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                             capture_output=True, text=True)
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
            first = diagnostics[:1]
            if (run.returncode != 1 or not any('expected=' in line for line in diagnostics)
                    or first != [mismatch]):
                raise AssertionError(
                    'control %s: unexpected result (exit %d) %s' % (control, run.returncode, run.stderr)
                )
            print('control %s: rejected at %s' % (control, first[0]))
        elif run.returncode or diagnostics:
            raise AssertionError('tg_size dispatch: %s' % (run.stderr or 'exit %d' % run.returncode))
        else:
            print('tg_size dispatch: all %d words match (%dx%dx%d groups x %dx%dx%d threads)' % (
                WORDS, gx, gy, gz, *group))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2]))
    elif len(sys.argv) == 5 and sys.argv[1] == 'waves-dispatch':
        sys.exit(check_dispatch(*sys.argv[2:], waves=True))
    elif len(sys.argv) == 6 and sys.argv[1] == 'waves-control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2], waves=True))
    elif len(sys.argv) == 5 and sys.argv[1] == 'fields-dispatch':
        sys.exit(check_fields_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'fields-control':
        sys.exit(check_fields_dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit(__doc__)
