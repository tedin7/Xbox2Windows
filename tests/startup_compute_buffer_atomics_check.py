"""Compute 32-bit MUBUF raw buffer atomics in tools/startup/gfx2hlsl.py (Refs #1545).

  (no arguments)                              translation, encoding and refusal controls, pure Python
  binary <llvm-mc>                            reassemble every fixture source and compare machine words
  dispatch <dxc> <offscreen_check> <adapter>  translate, compile DXC cs_6_0, dispatch and verify GPU readback
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))

from startup_gfx2hlsl_compute_check import G, MASK, SENTINEL, USER7, WORDS, assemble, refused, root_blob  # noqa: E402

ATOMIC_OPCODES = {
    'swap': (0x30, 1, 'Exchange'),
    'cmpswap': (0x31, 2, 'CompareExchange'),
    'add': (0x32, 1, 'Add'),
    'sub': (0x33, 1, 'Add'),
    'smin': (0x35, 1, 'Min'),
    'umin': (0x36, 1, 'Min'),
    'smax': (0x37, 1, 'Max'),
    'umax': (0x38, 1, 'Max'),
    'and': (0x39, 1, 'And'),
    'or': (0x3a, 1, 'Or'),
    'xor': (0x3b, 1, 'Xor'),
}
UNSUPPORTED_ATOMICS = {
    'inc': 0x3c,
    'dec': 0x3d,
    'fcmpswap': 0x3e,
    'fmin': 0x3f,
    'fmax': 0x40,
    'add_x2': 0x52,
}
INPUT_BASE = 0x2434c810

HEAD = bytes.fromhex(
    '06038abe'          # s_mov_b32 s10, s6
    '80038bbe'          # s_mov_b32 s11, 0
    '050408f4200000fa'  # s_load_dwordx4 s[16:19], s[10:11], 32
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
)


def s_mov_imm(reg, value):
    if 0 <= value <= 64:
        return struct.pack('<I', 0xbe800380 | ((reg & 0x7f) << 16) | value)
    return struct.pack('<II', 0xbe8003ff | ((reg & 0x7f) << 16), value & MASK)


def v_mov_imm(reg, value):
    head = 0x7e000200 | ((reg & 0xff) << 17)
    if 0 <= value <= 64:
        return struct.pack('<I', head | (0x80 + value))
    return struct.pack('<II', head | 0xff, value & MASK)


def vop2(op, vdst, src0, vsrc1, *, imm=None):
    word = (op << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<II', word, imm & MASK) if imm is not None else struct.pack('<I', word)


def mubuf(op, vdata, vaddr, srsrc, soffset_enc, *, offen=True, idxen=False, offset=0, glc=False, slc=False, dlc=False):
    w0 = (0xe0000000 | (op << 18) | (int(dlc) << 15) | (int(glc) << 14)
          | (int(idxen) << 13) | (int(offen) << 12) | (offset & 0xfff))
    w1 = (((soffset_enc & 0xff) << 24) | (int(slc) << 22)
          | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff))
    return struct.pack('<II', w0, w1)


def cs_pipe(code, group=(1, 1, 1), rsrc2=USER7):
    return G.ComputePipeline(code, root_blob(), rsrc2, group)


# Case 1: 32 contended threads (2 groups x 16 lanes) running all 9 reduction ops in sequence at distinct slots.
REDUCE_SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 32
s_mov_b32 s13, 32
s_waitcnt lgkmcnt(0)
s_lshl_b32 s12, s7, 4
v_add_nc_u32 v1, s12, v0
v_xor_b32 v2, 0x2434c810, v1
v_and_b32 v3, 3, v1
v_lshlrev_b32 v3, 30, v3
v_xor_b32 v2, v3, v2
v_mov_b32 v0, 0
buffer_atomic_add v2, v0, s[16:19], 0 offen
buffer_atomic_sub v2, v0, s[16:19], 0 offen offset:4
buffer_atomic_smin v2, v0, s[16:19], 0 offen offset:8
buffer_atomic_umin v2, v0, s[16:19], 0 offen offset:12
buffer_atomic_smax v2, v0, s[16:19], 0 offen offset:16
buffer_atomic_umax v2, v0, s[16:19], 0 offen offset:20
buffer_atomic_and v2, v0, s[16:19], 0 offen offset:24
buffer_atomic_or v2, v0, s[16:19], 0 offen offset:28
buffer_atomic_xor v2, v0, s[16:19], s13 offen
s_endpgm
"""
REDUCE_BYTES = (
    HEAD[:16]
    + s_mov_imm(13, 32)
    + HEAD[16:]
    + bytes.fromhex('07840c8f')           # s_lshl_b32 s12, s7, 4
    + vop2(0x25, 1, 12, 0)                # v_add_nc_u32 v1, s12, v0
    + vop2(0x1d, 2, 0xff, 1, imm=INPUT_BASE)  # v_xor_b32 v2, 0x2434c810, v1
    + vop2(0x1b, 3, 0x83, 1)              # v_and_b32 v3, 3, v1
    + vop2(0x1a, 3, 0x80 + 30, 3)         # v_lshlrev_b32 v3, 30, v3
    + vop2(0x1d, 2, 0x103, 2)             # v_xor_b32 v2, v3, v2
    + v_mov_imm(0, 0)                     # v_mov_b32 v0, 0
    + mubuf(0x32, 2, 0, 16, 0x80, offset=0)
    + mubuf(0x33, 2, 0, 16, 0x80, offset=4)
    + mubuf(0x35, 2, 0, 16, 0x80, offset=8)
    + mubuf(0x36, 2, 0, 16, 0x80, offset=12)
    + mubuf(0x37, 2, 0, 16, 0x80, offset=16)
    + mubuf(0x38, 2, 0, 16, 0x80, offset=20)
    + mubuf(0x39, 2, 0, 16, 0x80, offset=24)
    + mubuf(0x3a, 2, 0, 16, 0x80, offset=28)
    + mubuf(0x3b, 2, 0, 16, 13, offset=0)
    + bytes.fromhex('000081bf')
)

# Case 2: 8 lanes (1 group x 8 lanes) testing EXEC mask, glc pre-atomic return, swap, cmpswap order
# (replacement before compare, glc overwrites only vdata[0]), signed glc return, and vdata==vaddr aliasing.
EXCHANGE_SOURCE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 32
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v1, 2, v0
v_add_nc_u32 v2, 16, v0
v_mov_b32 v3, 5
v_mov_b32 v4, 0xaa000000
v_or_b32 v4, v4, v0
v_mov_b32 v14, 0x80000001
v_cmp_gt_u32 vcc_lo, 6, v0
s_and_saveexec_b32 s0, vcc_lo
buffer_atomic_add v2, v1, s[16:19], 0 offen glc
buffer_atomic_sub v3, v1, s[16:19], 0 offen offset:64 glc
buffer_atomic_smin v14, v1, s[16:19], 0 offen offset:96 glc
buffer_atomic_swap v4, v1, s[16:19], 0 offen offset:128 glc
s_or_b32 exec_lo, exec_lo, s0
s_waitcnt vmcnt(0)
buffer_store_dword v2, v1, s[16:19], 0 offen offset:32
buffer_store_dword v3, v1, s[16:19], 0 offen offset:160
buffer_store_dword v14, v1, s[16:19], 0 offen offset:192
buffer_store_dword v4, v1, s[16:19], 0 offen offset:224
v_add_nc_u32 v5, 0x11223344, v0
v_mov_b32 v6, 0xc5c5c5c5
v_mov_b32 v7, 0x55667788
v_mov_b32 v8, 0xc5c5c5c5
v_cmp_eq_u32 vcc_lo, 1, v0
v_cndmask_b32 v8, v8, 0, vcc_lo
buffer_atomic_cmpswap v[5:6], v1, s[16:19], 0 offen offset:256
buffer_atomic_cmpswap v[7:8], v1, s[16:19], 0 offen offset:288 glc
s_waitcnt vmcnt(0)
v_lshlrev_b32 v9, 3, v0
buffer_store_dwordx2 v[7:8], v9, s[16:19], 0 offen offset:320
v_add_nc_u32 v10, 384, v1
buffer_atomic_add v10, v10, s[16:19], 0 offen glc
s_waitcnt vmcnt(0)
buffer_store_dword v10, v1, s[16:19], 0 offen offset:416
s_endpgm
"""


def s32(word):
    return word - 0x100000000 if word & 0x80000000 else word


# Case 2 assembly matching EXCHANGE_BYTES with verified VOP2 compare + v_cndmask_b32 encodings.
EXCHANGE_FIXED_SOURCE = EXCHANGE_SOURCE.replace(
    'v_cmp_eq_u32 vcc_lo, 1, v0\nv_cndmask_b32 v8, v8, 0, vcc_lo',
    'v_cmp_gt_u32 vcc_lo, 1, v0\nv_cndmask_b32 v8, 0, v8, vcc_lo',
)


def exchange_bytes():
    code = bytearray(HEAD)
    code += vop2(0x1a, 1, 0x82, 0)                      # v_lshlrev_b32 v1, 2, v0
    code += vop2(0x25, 2, 0x90, 0)                      # v_add_nc_u32 v2, 16, v0
    code += v_mov_imm(3, 5)                             # v_mov_b32 v3, 5
    code += v_mov_imm(4, 0xaa000000)                    # v_mov_b32 v4, 0xaa000000
    code += vop2(0x1c, 4, 0x104, 0)                     # v_or_b32 v4, v4, v0
    code += v_mov_imm(14, 0x80000001)                   # v_mov_b32 v14, 0x80000001
    code += bytes.fromhex('8600887d')                   # v_cmp_gt_u32 vcc_lo, 6, v0
    code += bytes.fromhex('6a3c80be')                   # s_and_saveexec_b32 s0, vcc_lo
    code += mubuf(0x32, 2, 1, 16, 0x80, offset=0, glc=True)
    code += mubuf(0x33, 3, 1, 16, 0x80, offset=64, glc=True)
    code += mubuf(0x35, 14, 1, 16, 0x80, offset=96, glc=True)
    code += mubuf(0x30, 4, 1, 16, 0x80, offset=128, glc=True)
    code += bytes.fromhex('7e007e88')                   # s_or_b32 exec_lo, exec_lo, s0
    code += bytes.fromhex('703f8cbf')                   # s_waitcnt vmcnt(0)
    code += mubuf(0x1c, 2, 1, 16, 0x80, offset=32)      # buffer_store_dword v2, ... offset:32
    code += mubuf(0x1c, 3, 1, 16, 0x80, offset=160)     # buffer_store_dword v3, ... offset:160
    code += mubuf(0x1c, 14, 1, 16, 0x80, offset=192)    # buffer_store_dword v14, ... offset:192
    code += mubuf(0x1c, 4, 1, 16, 0x80, offset=224)     # buffer_store_dword v4, ... offset:224
    code += vop2(0x25, 5, 0xff, 0, imm=0x11223344)      # v_add_nc_u32 v5, 0x11223344, v0
    code += v_mov_imm(6, SENTINEL)
    code += v_mov_imm(7, 0x55667788)
    code += v_mov_imm(8, SENTINEL)
    code += bytes.fromhex('8100887d')                   # v_cmp_gt_u32 vcc_lo, 1, v0
    code += vop2(0x01, 8, 0x80, 8)                      # v_cndmask_b32_e32 v8, 0, v8, vcc_lo (VOP2 op 1)
    code += mubuf(0x31, 5, 1, 16, 0x80, offset=256, glc=False)
    code += mubuf(0x31, 7, 1, 16, 0x80, offset=288, glc=True)
    code += bytes.fromhex('703f8cbf')                   # s_waitcnt vmcnt(0)
    code += vop2(0x1a, 9, 0x83, 0)                      # v_lshlrev_b32 v9, 3, v0
    code += mubuf(0x1d, 7, 9, 16, 0x80, offset=320)     # buffer_store_dwordx2 v[7:8], v9, ... offset:320
    code += vop2(0x25, 10, 0xff, 1, imm=384)            # v_add_nc_u32 v10, 384, v1
    code += mubuf(0x32, 10, 10, 16, 0x80, offset=0, glc=True)
    code += bytes.fromhex('703f8cbf')                   # s_waitcnt vmcnt(0)
    code += mubuf(0x1c, 10, 1, 16, 0x80, offset=416)    # buffer_store_dword v10, v1, ... offset:416
    code += bytes.fromhex('000081bf')                   # s_endpgm
    return bytes(code)


EXCHANGE_BYTES = exchange_bytes()

# Two single-lane groups address separate counters 32 bytes apart. Each stores
# the GLC return with bit zero flipped so a missing observer store differs
# from the initial sentinel in the word 64 bytes past its base.
SOFFSET_SOURCE = '''\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 32
s_waitcnt lgkmcnt(0)
s_lshl_b32 s13, s7, 5
v_mov_b32 v1, s13
v_add_nc_u32 v4, 7, v1
buffer_atomic_add v4, off, s[16:19], s13{immediate} glc
s_waitcnt vmcnt(0)
v_xor_b32 v4, 1, v4
buffer_store_dword v4, v1, s[16:19], 0 offen offset:64
s_endpgm
'''


def soffset_code(immediate):
    return (HEAD
            + bytes.fromhex('07850d8f 0d02027e')  # s_lshl_b32 s13, s7, 5; v_mov_b32 v1, s13
            + vop2(0x25, 4, 0x87, 1)            # v_add_nc_u32 v4, 7, v1
            + mubuf(0x32, 4, 0, 16, 13, offen=False, offset=immediate, glc=True)
            + bytes.fromhex('703f8cbf')          # s_waitcnt vmcnt(0)
            + vop2(0x1d, 4, 0x81, 4)            # v_xor_b32 v4, 1, v4
            + mubuf(0x1c, 4, 1, 16, 0x80, offset=64)
            + bytes.fromhex('000081bf'))


def soffset_pipe(immediate):
    return cs_pipe(soffset_code(immediate), rsrc2=USER7 | (1 << 7))


def soffset_reference(immediate):
    out = [SENTINEL] * WORDS
    for soffset in (0, 32):
        out[(soffset + immediate) // 4] = (SENTINEL + 7 + soffset) & MASK
        out[(soffset + 64) // 4] = SENTINEL ^ 1
    return out


def reduce_reference():
    out = [SENTINEL] * WORDS
    vals = [INPUT_BASE ^ gid ^ ((gid & 3) << 30) for gid in range(32)]
    out[0] = (SENTINEL + sum(vals)) & MASK
    out[1] = (SENTINEL - sum(vals)) & MASK
    out[2] = min([SENTINEL] + vals, key=s32)
    out[3] = min([SENTINEL] + vals)
    out[4] = max([SENTINEL] + vals, key=s32)
    out[5] = max([SENTINEL] + vals)
    a, o, x = SENTINEL, SENTINEL, SENTINEL
    for val in vals:
        a &= val
        o |= val
        x ^= val
    out[6], out[7], out[8] = a, o, x
    return out


def exchange_reference():
    out = [SENTINEL] * WORDS
    for lane in range(8):
        if lane < 6:
            out[lane] = (SENTINEL + 16 + lane) & MASK
            out[8 + lane] = SENTINEL
            out[16 + lane] = (SENTINEL - 5) & MASK
            out[40 + lane] = SENTINEL
            out[24 + lane] = 0x80000001
            out[48 + lane] = SENTINEL
            out[32 + lane] = 0xaa000000 | lane
            out[56 + lane] = SENTINEL
        else:
            out[8 + lane] = (16 + lane) & MASK
            out[40 + lane] = 5
            out[48 + lane] = 0x80000001
            out[56 + lane] = 0xaa000000 | lane
        # Unmasked cmpswap without glc (always expects SENTINEL):
        out[64 + lane] = (0x11223344 + lane) & MASK
        # Unmasked cmpswap with glc: only lane 0 has expected==SENTINEL; lanes 1..7 keep memory SENTINEL,
        # and all 8 lanes write old SENTINEL into v7 while keeping v8 untouched (SENTINEL on 0, 0 on 1..7).
        out[72 + lane] = 0x55667788 if lane == 0 else SENTINEL
        out[80 + 2 * lane] = SENTINEL
        out[80 + 2 * lane + 1] = SENTINEL if lane == 0 else 0
        # vdata == vaddr aliasing with glc: byte 384 + 4*lane (dword 96 + lane).
        out[96 + lane] = (SENTINEL + 384 + 4 * lane) & MASK
        out[104 + lane] = SENTINEL
    return out


def single_atomic_code(op_name, *, glc=False, offen=True, idxen=False, slc=False, dlc=False,
                       offset=0, vdata=2, vaddr=0, srsrc=16, soffset_enc=0x80, prefix=HEAD):
    opcode, _, _ = ATOMIC_OPCODES[op_name]
    body = v_mov_imm(0, 0) + v_mov_imm(2, 7) + v_mov_imm(3, SENTINEL)
    insn = mubuf(opcode, vdata, vaddr, srsrc, soffset_enc, offen=offen, idxen=idxen,
                 offset=offset, glc=glc, slc=slc, dlc=dlc)
    return prefix + body + insn + bytes.fromhex('000081bf')


def controls(llvm_mc):
    # Verify multi-op reduction and exchange/glc/EXEC/cmpswap fixtures.
    if llvm_mc:
        assert REDUCE_BYTES == assemble(llvm_mc, REDUCE_SOURCE)
        assert EXCHANGE_BYTES == assemble(llvm_mc, EXCHANGE_FIXED_SOURCE)
    reduce_hlsl = cs_pipe(REDUCE_BYTES, group=(16, 1, 1), rsrc2=USER7 | (1 << 7)).hlsl_cs()
    assert 'u0.InterlockedAdd(a, v2);' in reduce_hlsl
    assert 'u0.InterlockedAdd(a, (0u - v2));' in reduce_hlsl
    assert 'u0.InterlockedMin(a, asint(v2));' in reduce_hlsl
    assert 'u0.InterlockedMin(a, v2);' in reduce_hlsl
    assert 'u0.InterlockedMax(a, asint(v2));' in reduce_hlsl
    assert 'u0.InterlockedMax(a, v2);' in reduce_hlsl
    assert 'u0.InterlockedAnd(a, v2);' in reduce_hlsl
    assert 'u0.InterlockedOr(a, v2);' in reduce_hlsl
    assert 'u0.InterlockedXor(a, v2);' in reduce_hlsl
    assert 'ao = 0;' not in reduce_hlsl and 'ai = 0;' not in reduce_hlsl, reduce_hlsl

    exch_hlsl = cs_pipe(EXCHANGE_BYTES, group=(8, 1, 1)).hlsl_cs()
    assert 'uint ao = 0;' in exch_hlsl and 'int ai = 0;' in exch_hlsl, exch_hlsl
    assert 'if (exec & 1) u0.InterlockedAdd(a, v2, ao);' in exch_hlsl
    assert 'if (exec & 1) u0.InterlockedAdd(a, (0u - v3), ao);' in exch_hlsl
    assert 'if (exec & 1) u0.InterlockedMin(a, asint(v14), ai);' in exch_hlsl
    assert 'if (exec & 1) v14 = asuint(ai);' in exch_hlsl
    assert 'if (exec & 1) u0.InterlockedExchange(a, v4, ao);' in exch_hlsl
    assert 'u0.InterlockedCompareStore(a, v6, v5);' in exch_hlsl
    assert 'u0.InterlockedCompareExchange(a, v8, v7, ao);' in exch_hlsl
    assert 'v7 = ao;' in exch_hlsl and 'v8 = ao;' not in exch_hlsl, exch_hlsl

    for immediate in (0, 4):
        pipe = soffset_pipe(immediate)
        if llvm_mc:
            modifier = f' offset:{immediate}' if immediate else ''
            assert soffset_code(immediate) == assemble(llvm_mc, SOFFSET_SOURCE.format(immediate=modifier))
        hlsl = pipe.hlsl_cs()
        assert f'a = (s13 + {immediate}) & 0xfffffffcu;' in hlsl
        assert 'u0.InterlockedAdd(a, v4, ao);' in hlsl and 'v4 = ao;' in hlsl
        assert pipe.manifest().splitlines()[-1] == 'uav 0 table 1 32', pipe.manifest()

    # Individual opcodes with and without glc, plus llvm-mc byte checks.
    for op_name, (opcode, width, method) in ATOMIC_OPCODES.items():
        data_tok = 'v[2:3]' if width == 2 else 'v2'
        for glc in (False, True):
            code = single_atomic_code(op_name, glc=glc)
            if llvm_mc:
                src = (f's_mov_b32 s10, s6\ns_mov_b32 s11, 0\n'
                       f's_load_dwordx4 s[16:19], s[10:11], 32\ns_waitcnt lgkmcnt(0)\n'
                       f'v_mov_b32 v0, 0\nv_mov_b32 v2, 7\nv_mov_b32 v3, 0xc5c5c5c5\n'
                       f'buffer_atomic_{op_name} {data_tok}, v0, s[16:19], 0 offen'
                       f'{" glc" if glc else ""}\ns_endpgm\n')
                assert code == assemble(llvm_mc, src), (op_name, glc)
            pipe = cs_pipe(code)
            hlsl = pipe.hlsl_cs()
            expected_method = ('CompareExchange' if (op_name == 'cmpswap' and glc)
                               else 'CompareStore' if op_name == 'cmpswap' else method)
            expected_glc = 'v2 = asuint(ai);' if op_name in ('smin', 'smax') else 'v2 = ao;'
            assert f'u0.Interlocked{expected_method}(' in hlsl, (op_name, glc, hlsl)
            assert (expected_glc in hlsl) == glc, (op_name, glc, hlsl)
            assert pipe.manifest().splitlines()[-1] == 'uav 0 table 1 32', pipe.manifest()

    # Non-zero aligned literal soffset + aligned offset:2 + soffset 2 works, unaligned offset refuses.
    assert 'a = (v0 + 4) & 0xfffffffcu;' in cs_pipe(single_atomic_code('add', offset=2, soffset_enc=0x82)).hlsl_cs()
    refused('unaligned immediate offset', lambda: cs_pipe(single_atomic_code('add', offset=2)))
    refused('unaligned immediate offset', lambda: cs_pipe(single_atomic_code('add', offset=1)))

    # The off variant is limited to add + glc + a scalar register, with the same offset and modifier checks.
    refused('without the offen address form', lambda: cs_pipe(single_atomic_code('add', offen=False)))
    refused('without the offen address form', lambda: cs_pipe(single_atomic_code('sub', offen=False, glc=True, soffset_enc=13)))
    refused('off requires a scalar register soffset', lambda: cs_pipe(single_atomic_code('add', offen=False, glc=True)))
    refused('unaligned immediate offset', lambda: soffset_pipe(2))
    refused('buffer atomic modifier idxen', lambda: cs_pipe(single_atomic_code('add', offen=False, idxen=True, glc=True, soffset_enc=13)))
    refused('buffer soffset vcc_lo', lambda: cs_pipe(single_atomic_code('add', offen=False, glc=True, soffset_enc=0x6a)))
    refused('buffer atomic modifier idxen', lambda: cs_pipe(single_atomic_code('add', idxen=True)))
    refused('buffer atomic modifier idxen', lambda: cs_pipe(single_atomic_code('add', idxen=True, offen=True)))
    refused('buffer atomic modifier slc', lambda: cs_pipe(single_atomic_code('add', slc=True)))
    refused('buffer atomic modifier dlc', lambda: cs_pipe(single_atomic_code('add', dlc=True)))
    refused('buffer soffset vcc_lo', lambda: cs_pipe(single_atomic_code('add', soffset_enc=0x6a)))
    refused('buffer atomic data v[255:256]', lambda: cs_pipe(single_atomic_code('cmpswap', vdata=255)))

    # Direct Stage tests for operand-width & non-cs & negative soffset refusals.
    stage = cs_pipe(single_atomic_code('add')).cs
    refused('buffer atomic data', lambda: stage.buffer_atomic('buffer_atomic_add', ['v[2:3]', 'v0', 's[16:19]', '0'], ['offen']))
    refused('buffer atomic data', lambda: stage.buffer_atomic('buffer_atomic_cmpswap', ['v2', 'v0', 's[16:19]', '0'], ['offen']))
    refused('buffer atomic data', lambda: stage.buffer_atomic('buffer_atomic_cmpswap', ['v[2:4]', 'v0', 's[16:19]', '0'], ['offen']))
    refused('buffer soffset -4', lambda: stage.buffer_atomic('buffer_atomic_add', ['v2', 'v0', 's[16:19]', '-4'], ['offen']))
    stage.kind = 'ps'
    refused('buffer_atomic_add', lambda: stage.buffer_atomic('buffer_atomic_add', ['v2', 'v0', 's[16:19]', '0'], ['offen']))

    # Modified and aliased descriptor refusals.
    mod_prefix = HEAD + bytes.fromhex('11ff118800000400')  # s_or_b32 s17, s17, 0x40000
    refused('modified descriptor', lambda: cs_pipe(single_atomic_code('add', prefix=mod_prefix)))
    refused('holding None', lambda: cs_pipe(single_atomic_code('add', srsrc=20)))
    refused('holding', lambda: cs_pipe(single_atomic_code('add')).cs.buffer_atomic(
        'buffer_atomic_add', ['v2', 'v0', 's[17:20]', '0'], ['offen']))
    alias_prefix = HEAD + bytes.fromhex('001030e000050480')  # buffer_load_dword v5, v0, s[16:19], 0 offen
    refused('used for a load and a store', lambda: cs_pipe(single_atomic_code('add', prefix=alias_prefix)))

    # Out-of-scope atomics (inc/dec/float/64-bit) stay strictly refused by name.
    for op_name, opcode in UNSUPPORTED_ATOMICS.items():
        refused(f'buffer_atomic_{op_name}', lambda op=opcode: cs_pipe(
            HEAD + v_mov_imm(0, 0) + v_mov_imm(2, 1) + mubuf(op, 2, 0, 16, 0x80) + bytes.fromhex('000081bf')
        ))
    print('buffer_atomic controls (11 ops, scalar byte offset, glc/non-glc, alignment, modifiers, descriptors, refusals) passed')


def run_case(directory, compiler, renderer, adapter, name, hlsl, groups, want):
    src = directory / f'{name}.hlsl'
    cso = directory / f'{name}.cso'
    case = directory / f'{name}.case'
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([compiler, '-Gis', '-WX', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    case.write_bytes(struct.pack('<321I', groups[0] | (groups[1] << 8) | (groups[2] << 16), *want))
    return subprocess.run([renderer, adapter, '--compute-store-case', str(cso), str(case)],
                          capture_output=True, text=True)


def dispatch(directory, compiler, renderer, adapter):
    reduce_pipe = cs_pipe(REDUCE_BYTES, group=(16, 1, 1), rsrc2=USER7 | (1 << 7))
    reduce_hlsl = reduce_pipe.hlsl_cs()
    reduce_want = reduce_reference()
    res = run_case(directory, compiler, renderer, adapter, 'reduce_32threads', reduce_hlsl, (2, 1, 1), reduce_want)
    if res.returncode == 77:
        return 77
    assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (res.stdout, res.stderr)

    exch_pipe = cs_pipe(EXCHANGE_BYTES, group=(8, 1, 1))
    exch_hlsl = exch_pipe.hlsl_cs()
    exch_want = exchange_reference()
    res = run_case(directory, compiler, renderer, adapter, 'exchange_glc_exec_cmpswap', exch_hlsl, (1, 1, 1), exch_want)
    assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (res.stdout, res.stderr)

    for immediate in (0, 4):
        hlsl = soffset_pipe(immediate).hlsl_cs()
        want = soffset_reference(immediate)
        res = run_case(directory, compiler, renderer, adapter, f'soffset_add_{immediate}', hlsl, (2, 1, 1), want)
        assert res.returncode == 0 and f'compared_words={WORDS} mismatches=0' in res.stdout, (res.stdout, res.stderr)

    hlsl = soffset_pipe(4).hlsl_cs()
    want = soffset_reference(4)
    mutations = (
        ('dropped_soffset', 'a = (s13 + 4)', 'a = 4', 1, (SENTINEL + 7 + 39) & MASK),
        ('dropped_immediate', 'a = (s13 + 4)', 'a = (s13 + 0)', 0, (SENTINEL + 7) & MASK),
        ('dropped_soffset_glc_return', 'v4 = ao;', '', 16, 7 ^ 1),
        ('dropped_soffset_observer_store', 'u0.Store(a, v4);', '', 16, SENTINEL),
    )
    for name, original, replacement, word, value in mutations:
        assert hlsl.count(original) == 1
        result = run_case(directory, compiler, renderer, adapter, name, hlsl.replace(original, replacement), (2, 1, 1), want)
        assert result.returncode == 1 and f'word={word} expected={want[word]:08x} actual={value:08x}' in result.stderr, (
            name, result.returncode, result.stdout, result.stderr)

    # Negative controls on GPU: every mutated shader must fail at the predicted first mismatch word.
    wrong_sub_want = reduce_want.copy()
    vals = [INPUT_BASE ^ gid ^ ((gid & 3) << 30) for gid in range(32)]
    wrong_sub_want[1] = (SENTINEL + sum(vals)) & MASK
    wrong_smin_want = reduce_want.copy()
    wrong_smin_want[2] = min([SENTINEL] + vals)
    dropped_atomic_want = reduce_want.copy()
    dropped_atomic_want[0] = SENTINEL

    for name, src, groups, expected, actual_ref in (
        ('dropped_atomic', reduce_hlsl.replace('u0.InterlockedAdd(a, v2);', ''), (2, 1, 1), reduce_want, dropped_atomic_want),
        ('wrong_sub_sign', reduce_hlsl.replace('u0.InterlockedAdd(a, (0u - v2));', 'u0.InterlockedAdd(a, v2);'),
         (2, 1, 1), reduce_want, wrong_sub_want),
        ('unsigned_smin', reduce_hlsl.replace('u0.InterlockedMin(a, asint(v2));', 'u0.InterlockedMin(a, v2);'),
         (2, 1, 1), reduce_want, wrong_smin_want),
        ('swapped_cmpswap', exch_hlsl.replace('u0.InterlockedCompareExchange(a, v8, v7, ao);',
                                              'u0.InterlockedCompareExchange(a, v7, v8, ao);'),
         (1, 1, 1), exch_want, exch_want[:72] + [SENTINEL] + exch_want[73:]),
        ('dropped_glc_return', exch_hlsl.replace('if (exec & 1) v2 = ao;', ''),
         (1, 1, 1), exch_want, exch_want[:8] + [16] + exch_want[9:]),
        ('dropped_exec_guard', exch_hlsl.replace('if (exec & 1) u0.InterlockedAdd(a, v2, ao);',
                                                 'u0.InterlockedAdd(a, v2, ao);'),
         (1, 1, 1), exch_want, exch_want[:6] + [(SENTINEL + 22) & MASK, (SENTINEL + 23) & MASK] + exch_want[8:]),
    ):
        r = run_case(directory, compiler, renderer, adapter, name, src, groups, expected)
        first = next(i for i, (a, b) in enumerate(zip(expected, actual_ref)) if a != b)
        assert r.returncode == 1 and f'word={first} expected={expected[first]:08x} actual={actual_ref[first]:08x}' in r.stderr, (
            name, first, r.returncode, r.stdout, r.stderr)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    args = parser.parse_args()
    if args.dispatch:
        out = ROOT / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls(args.llvm_mc)
